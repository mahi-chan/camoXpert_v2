"""
Command-line training script with gradient accumulation & checkpointing
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import json
from tqdm import tqdm
import numpy as np
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.camoxpert import CamoXpert, count_parameters
from models.camoxpert_cod import CamoXpertCOD
from models.camoxpert_sparse_moe import CamoXpertSparseMoE
from data.dataset import COD10KDataset
from losses.advanced_loss import AdvancedCODLoss, CODSpecializedLoss
from metrics.cod_metrics import CODMetrics
from models.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='CamoXpert SOTA Training with Memory Optimization')

    parser.add_argument('command', type=str, choices=['train'])
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                        help='Backbone: pvt_v2_b2 (SOTA for COD, recommended), edgenext_base (mobile), swin_tiny, convnext_tiny')
    parser.add_argument('--num-experts', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--stage1-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--deep-supervision', action='store_true', default=False)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False)
    parser.add_argument('--use-ema', action='store_true', default=False)
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate (default: 0.999, use 0.9999 for very stable training)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage2-batch-size', type=int, default=None,
                        help='Batch size for stage 2 (default: same as --batch-size)')
    parser.add_argument('--progressive-unfreeze', action='store_true', default=False,
                        help='Gradually unfreeze backbone layers in stage 2')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--skip-stage1', action='store_true', default=False,
                        help='Skip stage 1 and go directly to stage 2 (use with --resume-from)')
    parser.add_argument('--stage2-lr', type=float, default=None,
                        help='Learning rate for Stage 2 (default: same as --lr)')
    parser.add_argument('--scheduler', type=str, choices=['onecycle', 'cosine', 'cosine_restart', 'none'],
                        default='onecycle', help='Learning rate scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine schedulers')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of warmup epochs for Stage 2')
    parser.add_argument('--t-mult', type=int, default=2,
                        help='T_mult for cosine_restart scheduler')
    parser.add_argument('--no-amp', action='store_true', default=False,
                        help='Disable mixed precision (use FP32 instead of FP16) - more stable but slower')
    parser.add_argument('--use-cod-specialized', action='store_true', default=False,
                        help='Use 100%% COD-specialized architecture (recommended for best results)')

    # Sparse MoE arguments
    parser.add_argument('--use-sparse-moe', action='store_true', default=False,
                        help='Use sparse MoE routing instead of dense experts (35-40%% faster)')
    parser.add_argument('--moe-num-experts', type=int, default=6,
                        help='Number of experts in MoE pool (default: 6)')
    parser.add_argument('--moe-top-k', type=int, default=2,
                        help='Number of experts to select per input (default: 2)')

    # DDP arguments
    parser.add_argument('--use-ddp', action='store_true', default=False,
                        help='Use DistributedDataParallel for multi-GPU training (recommended over DataParallel)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (auto-set by torch.distributed.launch)')

    return parser.parse_args()


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def setup_ddp(args):
    """
    Initialize DistributedDataParallel

    NOTE: For multi-GPU training with DDP, you must launch with torchrun:
        torchrun --nproc_per_node=2 train_ultimate.py train --use-ddp ...
    """
    # Set default MASTER_ADDR and MASTER_PORT for single-node training if not set
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Launched with torchrun or torch.distributed.launch
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_PROCID' in os.environ:
        # Running on SLURM cluster
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        # Not launched with torchrun - print helpful error message
        print("\n" + "="*70)
        print("⚠️  ERROR: DDP requires launching with torchrun")
        print("="*70)
        print(f"You have {torch.cuda.device_count()} GPUs available.")
        print("\nTo use DDP with multiple GPUs, launch with:\n")
        print(f"  torchrun --nproc_per_node={torch.cuda.device_count()} train_ultimate.py train --use-ddp [other args]")
        print("\nExample:")
        print(f"  torchrun --nproc_per_node=2 train_ultimate.py train \\")
        print(f"    --dataset-path /path/to/dataset \\")
        print(f"    --use-ddp --batch-size 8 --epochs 200")
        print("\nAlternatively, remove --use-ddp flag to train on single GPU.")
        print("="*70 + "\n")
        raise RuntimeError("DDP requires torchrun. See error message above for instructions.")

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                           world_size=args.world_size, rank=args.rank)
    dist.barrier()
    return args.rank == 0  # is_main_process


def cleanup_ddp():
    """Cleanup DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def clear_gpu_memory():
    """Clear GPU memory cache and collect garbage"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved")


def enable_gradient_checkpointing(model):
    print("🔧 Enabling gradient checkpointing...")
    checkpointed = 0

    def make_checkpointed_forward(original_forward):
        """Factory function to create checkpointed forward with proper closure"""
        def checkpointed_forward(self, *args, **kwargs):
            def custom_forward(*inputs):
                return original_forward(*inputs, **kwargs)

            return gradient_checkpoint(custom_forward, *args, use_reentrant=False)

        return checkpointed_forward

    # Checkpoint memory-intensive modules
    # For COD: expert modules, decoder blocks, refinement modules, contrast enhancement
    checkpoint_patterns = ['expert', 'decoder', 'refinement', 'contrast', 'moe', 'sdta']

    for name, module in model.named_modules():
        # Check if module name contains any checkpoint pattern
        should_checkpoint = any(pattern in name.lower() for pattern in checkpoint_patterns)

        if should_checkpoint and hasattr(module, 'forward'):
            # Skip if it's a container module (Sequential, ModuleList, etc.)
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue

            original_forward = module.forward
            module.forward = make_checkpointed_forward(original_forward).__get__(module, type(module))
            checkpointed += 1

    print(f"✓ Checkpointed {checkpointed} modules")
    return model


def get_actual_model(model):
    """Get the actual model, unwrapping DataParallel or DistributedDataParallel if needed"""
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def progressive_unfreeze_backbone(model, stage):
    """
    Progressively unfreeze backbone layers
    stage 0: freeze all
    stage 1: unfreeze last layer
    stage 2: unfreeze last 2 layers
    stage 3: unfreeze all
    """
    # Get actual model (unwrap DataParallel if needed)
    actual_model = get_actual_model(model)

    # First freeze everything
    for param in actual_model.backbone.parameters():
        param.requires_grad = False

    if stage == 0:
        return

    # Get backbone layers
    backbone_children = list(actual_model.backbone.children())
    num_layers = len(backbone_children)

    if stage >= 3:
        # Unfreeze all
        for param in actual_model.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone: All layers unfrozen")
    else:
        # Unfreeze last 'stage' layers
        layers_to_unfreeze = backbone_children[-stage:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"✓ Backbone: Last {stage}/{num_layers} layers unfrozen")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable/1e6:.1f}M")


def create_scheduler(optimizer, scheduler_type, total_steps=None, total_epochs=None,
                    max_lr=None, min_lr=1e-6, warmup_epochs=0, t_mult=2):
    """Create learning rate scheduler"""

    if scheduler_type == 'onecycle':
        if total_steps is None:
            raise ValueError("total_steps required for onecycle scheduler")
        return OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.1)

    elif scheduler_type == 'cosine':
        if total_epochs is None:
            raise ValueError("total_epochs required for cosine scheduler")
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    elif scheduler_type == 'cosine_restart':
        if total_epochs is None:
            raise ValueError("total_epochs required for cosine_restart scheduler")
        return CosineAnnealingWarmRestarts(optimizer, T_0=total_epochs//4, T_mult=t_mult, eta_min=min_lr)

    else:  # 'none'
        return None


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Apply learning rate warmup"""
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor
        return True
    return False


def train_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps, ema, epoch, total_epochs,
                use_deep_sup, use_amp=True, use_sparse_moe=False, use_cod_specialized=False):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    # Calculate router warmup factor for Sparse MoE
    # Gradually increase load balance loss over FULL Stage 1 (40 epochs)
    # CRITICAL: Changed from 20 to 40 epochs after explosion at epoch 10
    router_warmup_epochs = 40
    if use_sparse_moe and use_cod_specialized and epoch < router_warmup_epochs:
        router_warmup_factor = (epoch + 1) / router_warmup_epochs
    else:
        router_warmup_factor = 1.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}/{total_epochs}")

    for batch_idx, (images, masks) in pbar:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        # Conditionally use mixed precision
        if use_amp:
            context = torch.amp.autocast('cuda')
        else:
            context = torch.cuda.amp.autocast(enabled=False)

        with context:
            # Pass warmup factor to model ONLY if using CamoXpertSparseMoE (requires both flags)
            if use_sparse_moe and use_cod_specialized:
                pred, aux_or_dict, deep = model(images, return_deep_supervision=use_deep_sup,
                                                warmup_factor=router_warmup_factor)
            else:
                pred, aux_or_dict, deep = model(images, return_deep_supervision=use_deep_sup)

            # Debug: Check for NaN/Inf in model outputs BEFORE loss
            if not torch.isfinite(pred).all():
                print(f"\n❌ NaN/Inf in model prediction output at batch {batch_idx}!")
                print(f"   Pred min: {pred.min().item()}, max: {pred.max().item()}")
                raise ValueError("Model produced NaN/Inf outputs")

            # Handle different model outputs
            if isinstance(aux_or_dict, dict):
                # COD-specialized model returns dict with auxiliary outputs
                uncertainty = aux_or_dict.get('uncertainty', None)
                fg_map = aux_or_dict.get('fg_map', None)
                search_map = aux_or_dict.get('search_map', None)
                refinements = aux_or_dict.get('refinements', None)
                load_balance_loss = aux_or_dict.get('load_balance_loss', 0.0)
                aux_loss = None
                loss, _ = criterion(pred, masks, aux_loss, deep,
                                  uncertainty=uncertainty, fg_map=fg_map,
                                  refinements=refinements, search_map=search_map)
                # Add load balance loss if using sparse MoE
                if load_balance_loss != 0.0:
                    loss = loss + load_balance_loss
            else:
                # Standard model returns aux_loss
                aux_loss = aux_or_dict
                if aux_loss is not None and not torch.isfinite(aux_loss).all():
                    print(f"\n❌ NaN/Inf in aux_loss at batch {batch_idx}!")
                    print(f"   Aux loss: {aux_loss}")
                    raise ValueError("Aux loss is NaN/Inf")
                loss, _ = criterion(pred, masks, aux_loss, deep)

            loss = loss / accumulation_steps

        # Check for NaN/Inf in loss
        if not torch.isfinite(loss):
            print(f"\n❌ NaN/Inf detected in loss at batch {batch_idx}!")
            print(f"   Loss value: {loss.item()}")
            print(f"   Skipping this batch and stopping training for safety.")
            print(f"   Please reduce learning rate and restart from last good checkpoint.")
            raise ValueError("Training stopped due to NaN/Inf loss. See error message above.")

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            # Extra aggressive clipping for router parameters if using sparse MoE
            if use_sparse_moe:
                router_params = []
                other_params = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Check if this is a router parameter
                        if 'router' in name or 'gate' in name:
                            router_params.append(param)
                        else:
                            other_params.append(param)

                # Clip router gradients more aggressively (0.1 vs 0.5)
                if router_params:
                    torch.nn.utils.clip_grad_norm_(router_params, 0.1)

            # Clip all gradients MORE aggressively (0.5 instead of 1.0) to prevent explosions
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Check for NaN/Inf in gradients
            if not torch.isfinite(grad_norm):
                print(f"\n❌ NaN/Inf detected in gradients at batch {batch_idx}!")
                print(f"   Gradient norm: {grad_norm.item()}")
                print(f"   This indicates gradient explosion.")
                print(f"   Please reduce learning rate and restart from last good checkpoint.")
                raise ValueError("Training stopped due to NaN/Inf gradients. See error message above.")

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update()

        epoch_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    if len(loader) % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        # Clip gradients MORE aggressively (0.5 instead of 1.0) to prevent explosions
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Check for NaN/Inf in final gradients
        if not torch.isfinite(grad_norm):
            print(f"\n❌ NaN/Inf detected in final gradients!")
            print(f"   Gradient norm: {grad_norm.item()}")
            raise ValueError("Training stopped due to NaN/Inf gradients in final step.")

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, metrics, use_ddp=False):
    """
    Validation function with DDP support

    Computes metrics on local data, then synchronizes across all GPUs
    to get global average over the full validation set.
    """
    # Unwrap DataParallel/DDP for validation
    actual_model = get_actual_model(model)
    actual_model.eval()

    all_metrics = []
    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images, masks = images.cuda(), masks.cuda()
        pred, _, _ = actual_model(images)
        pred = torch.sigmoid(pred)
        all_metrics.append(metrics.compute_all(pred, masks))

    # Compute local metrics (for this GPU's subset)
    local_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    # Synchronize metrics across all GPUs if using DDP
    if use_ddp and dist.is_initialized():
        # Convert to tensors for all_reduce
        metric_tensor = torch.tensor(
            [local_metrics['IoU'], local_metrics['Dice_Score']],
            device='cuda'
        )

        # Average across all GPUs
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)

        # Update with synchronized values
        local_metrics['IoU'] = metric_tensor[0].item()
        local_metrics['Dice_Score'] = metric_tensor[1].item()

    return local_metrics


def train(args):
    # Enable PyTorch memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    set_seed(args.seed)
    device = torch.device(args.device)

    # Set stage 2 batch size (default to half of stage 1 for memory efficiency)
    if args.stage2_batch_size is None:
        args.stage2_batch_size = max(1, args.batch_size // 2)

    effective_batch_s1 = args.batch_size * args.accumulation_steps
    effective_batch_s2 = args.stage2_batch_size * args.accumulation_steps

    print("\n" + "=" * 70)
    print("CAMOXPERT ULTIMATE TRAINING")
    print("=" * 70)
    print(f"Backbone:         {args.backbone}")
    print(f"Experts:          {args.num_experts}")
    print(f"Resolution:       {args.img_size}px")
    print(f"Stage 1 Batch:    {args.batch_size} × {args.accumulation_steps} = {effective_batch_s1} effective")
    print(f"Stage 2 Batch:    {args.stage2_batch_size} × {args.accumulation_steps} = {effective_batch_s2} effective")
    print(f"Epochs:           {args.epochs}")
    print(f"Deep Supervision: {args.deep_supervision}")
    print(f"Grad Checkpoint:  {args.gradient_checkpointing}")
    print(f"Progressive:      {args.progressive_unfreeze}")
    print(f"EMA:              {args.use_ema}")
    print(f"\n🎯 Target: IoU ≥ 0.72")
    print("=" * 70 + "\n")

    print_gpu_memory()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize DDP FIRST if enabled
    n_gpus = torch.cuda.device_count()
    is_main_process = True

    if args.use_ddp and n_gpus > 1:
        if is_main_process:
            print(f"🚀 Initializing DistributedDataParallel with {n_gpus} GPUs!\n")
        is_main_process = setup_ddp(args)

    # Datasets
    train_data = COD10KDataset(args.dataset_path, 'train', args.img_size, augment=True)
    val_data = COD10KDataset(args.dataset_path, 'val', args.img_size, augment=False)

    # Create samplers for DDP if enabled (AFTER setup_ddp())
    train_sampler = None
    val_sampler = None
    if args.use_ddp and n_gpus > 1:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        val_sampler = DistributedSampler(val_data, shuffle=False)

    train_loader = DataLoader(train_data, args.batch_size,
                              shuffle=(train_sampler is None),  # Don't shuffle if using sampler
                              sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=True if args.num_workers > 0 else False,
                              prefetch_factor=3,
                              multiprocessing_context='fork' if args.num_workers > 0 else None)
    val_loader = DataLoader(val_data, args.batch_size,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True if args.num_workers > 0 else False,
                            prefetch_factor=2,
                            multiprocessing_context='fork' if args.num_workers > 0 else None)

    if is_main_process:
        print(f"Train: {len(train_data)} | Val: {len(val_data)}\n")

    # Model
    if args.use_cod_specialized:
        if args.use_sparse_moe:
            if is_main_process:
                print("🎯 Using 100% COD-Specialized Architecture with Sparse MoE Routing")
                print(f"   MoE Experts: {args.moe_num_experts}")
                print(f"   Top-k Selection: {args.moe_top_k}")
                print(f"   Sparsity: {100 * args.moe_top_k / args.moe_num_experts:.0f}% active")
            model = CamoXpertSparseMoE(3, 1, pretrained=True, backbone=args.backbone,
                                       num_experts=args.moe_num_experts, top_k=args.moe_top_k).cuda()
        else:
            if is_main_process:
                print("🎯 Using 100% COD-Specialized Architecture")
            model = CamoXpertCOD(3, 1, pretrained=True, backbone=args.backbone).cuda()
    else:
        if is_main_process:
            print("📦 Using Standard CamoXpert Architecture")
        model = CamoXpert(3, 1, pretrained=True, backbone=args.backbone, num_experts=args.num_experts).cuda()

    # Wrap model with DDP if enabled
    # Determine if we need find_unused_parameters before wrapping with DDP
    # Stage 1: backbone frozen → need find_unused_parameters=True
    # Stage 2: all params active → find_unused_parameters=False (better performance)
    # Check if we'll run Stage 1 or go straight to Stage 2
    will_run_stage1 = True
    if args.resume_from and os.path.exists(args.resume_from):
        # Peek at checkpoint to see if we're past Stage 1
        checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        resume_epoch = checkpoint.get('epoch', -1) + 1
        will_run_stage1 = resume_epoch < args.stage1_epochs

    if args.use_ddp and n_gpus > 1:
        if is_main_process:
            for i in range(n_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        # Set find_unused_parameters based on which stage we'll run
        use_find_unused = will_run_stage1
        if is_main_process:
            print(f"   DDP find_unused_parameters: {use_find_unused} ({'Stage 1' if will_run_stage1 else 'Stage 2'})")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                   find_unused_parameters=use_find_unused)
        if is_main_process:
            print(f"   Batch per GPU: {args.batch_size}")
            print(f"   Total batch: {args.batch_size * n_gpus}\n")
    else:
        if n_gpus > 1:
            print(f"⚠️  Multiple GPUs detected but DDP not enabled.")
            print(f"   Use --use-ddp flag to enable multi-GPU training")
            print(f"   Using single GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}\n")

        # Automatic batch size adjustment for single GPU
        if args.batch_size > 32:
            original_batch = args.batch_size
            args.batch_size = 24  # Safe for Tesla T4
            # Increase gradient accumulation to maintain effective batch size
            args.accumulation_steps = max(args.accumulation_steps, (original_batch + 23) // 24)
            if is_main_process:
                print(f"⚠️  Adjusted for single GPU:")
                print(f"   Batch size: {original_batch} → {args.batch_size}")
                print(f"   Gradient accumulation: {args.accumulation_steps} steps")
                print(f"   Effective batch: {args.batch_size * args.accumulation_steps}\n")

    if args.gradient_checkpointing:
        model = enable_gradient_checkpointing(model)

    total, trainable = count_parameters(model)
    print(f"Model: {total / 1e6:.1f}M params\n")

    if args.use_cod_specialized:
        print("Using COD-Specialized Loss Function")
        # Reduced loss weights for AMP stability (prevent FP16 overflow)
        criterion = CODSpecializedLoss(bce_weight=2.0, iou_weight=1.5, edge_weight=1.0,
                                      boundary_weight=1.5, uncertainty_weight=0.3,
                                      reverse_attention_weight=0.8, aux_weight=0.1)
    else:
        # Reduced loss weights for AMP stability
        criterion = AdvancedCODLoss(bce_weight=2.0, iou_weight=1.5, edge_weight=1.0, aux_weight=0.1)
    metrics = CODMetrics()
    # Very conservative GradScaler for initial stability (will grow automatically)
    scaler = torch.cuda.amp.GradScaler(init_scale=512, growth_interval=2000)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if args.use_ema:
        print(f"✨ EMA enabled with decay: {args.ema_decay}")

    best_iou = 0.0
    history = []
    start_epoch = 0

    # Load checkpoint if resuming
    if args.resume_from:
        print(f"\n{'='*70}")
        print(f"LOADING CHECKPOINT: {args.resume_from}")
        print(f"{'='*70}")
        if not os.path.exists(args.resume_from):
            print(f"❌ ERROR: Checkpoint not found at {args.resume_from}")
            return

        checkpoint = torch.load(args.resume_from, map_location='cuda', weights_only=False)

        # Handle DataParallel state_dict loading
        state_dict = checkpoint['model_state_dict']
        if n_gpus > 1 and not any(k.startswith('module.') for k in state_dict.keys()):
            # Checkpoint was saved without DataParallel, but we're using DataParallel now
            # Add 'module.' prefix to all keys
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif n_gpus == 1 and any(k.startswith('module.') for k in state_dict.keys()):
            # Checkpoint was saved with DataParallel, but we're not using it now
            # Remove 'module.' prefix from all keys
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        if ema and checkpoint.get('ema_state_dict'):
            ema.shadow = checkpoint['ema_state_dict']

        best_iou = checkpoint.get('best_iou', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"✓ Best IoU so far: {best_iou:.4f}")
        print(f"✓ Resuming from epoch {start_epoch}")

        if args.skip_stage1 and start_epoch < args.stage1_epochs:
            print(f"⚠️  WARNING: Checkpoint is from epoch {checkpoint.get('epoch', 0)} (Stage 1)")
            print(f"   You requested --skip-stage1, jumping to epoch {args.stage1_epochs}")
            start_epoch = args.stage1_epochs

        print(f"{'='*70}\n")

    # Stage 1
    if start_epoch < args.stage1_epochs:
        print("=" * 70)
        print("STAGE 1: DECODER TRAINING")
        print("=" * 70)

        actual_model = get_actual_model(model)
        for param in actual_model.backbone.parameters():
            param.requires_grad = False

        # Separate learning rates for router vs other parameters (if using sparse MoE)
        if args.use_sparse_moe:
            router_params = []
            other_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Router/gate parameters get 0.01× learning rate for stability
                    if 'router' in name or 'gate' in name:
                        router_params.append(param)
                    else:
                        other_params.append(param)

            # CRITICAL: Router LR = 0.01× main LR for stability at 416px
            optimizer = AdamW([
                {'params': other_params, 'lr': args.lr},
                {'params': router_params, 'lr': args.lr * 0.01}  # 100× slower for router
            ], weight_decay=args.weight_decay)

            if is_main_process:
                print(f"🎯 Sparse MoE: Router LR = {args.lr * 0.01:.6f} (0.01× main LR)")
                print(f"   Other params LR = {args.lr:.6f}")
        else:
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay)

        # Create scheduler for Stage 1
        total_steps = len(train_loader) * args.stage1_epochs // args.accumulation_steps
        total_epochs = args.stage1_epochs
        scheduler = create_scheduler(optimizer, args.scheduler,
                                     total_steps=total_steps,
                                     total_epochs=total_epochs,
                                     max_lr=args.lr,
                                     min_lr=args.min_lr,
                                     t_mult=args.t_mult)

        # Advance scheduler to correct position if resuming during Stage 1
        if start_epoch > 0 and scheduler is not None and args.scheduler != 'onecycle':
            if is_main_process:
                print(f"   Advancing Stage 1 scheduler by {start_epoch} steps to sync with resumed epoch\n")
            for _ in range(start_epoch):
                scheduler.step()

        for epoch in range(start_epoch, args.stage1_epochs):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                     args.accumulation_steps, ema, epoch, args.stage1_epochs, args.deep_supervision,
                                     use_amp=not args.no_amp, use_sparse_moe=args.use_sparse_moe,
                                     use_cod_specialized=args.use_cod_specialized)

            # Step scheduler if it exists
            if scheduler is not None:
                if args.scheduler == 'onecycle':
                    for _ in range(len(train_loader) // args.accumulation_steps):
                        scheduler.step()
                else:
                    scheduler.step()

            if ema:
                ema.apply_shadow()
            val_metrics = validate(model, val_loader, metrics, use_ddp=args.use_ddp)
            if ema:
                ema.restore()

            print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}")

            # Monitor Sparse MoE router health
            if args.use_sparse_moe and is_main_process:
                # Get a batch to check router behavior
                sample_imgs, _ = next(iter(val_loader))
                sample_imgs = sample_imgs.cuda()
                actual_model = get_actual_model(model)
                with torch.no_grad():
                    _, aux, _ = actual_model(sample_imgs, return_deep_supervision=False)
                    if isinstance(aux, dict):
                        lb_loss = aux.get('load_balance_loss', 0.0)
                        warmup = aux.get('router_warmup_factor', 1.0)

                        # Check router health
                        if lb_loss > 0:
                            lb_value = lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss
                            print(f"   Router LB Loss: {lb_value:.6f} | Warmup: {warmup:.2f}")

                            # Warning thresholds
                            if epoch >= 20:  # After warmup
                                if lb_value < 0.00005:
                                    print(f"   ⚠️  WARNING: Load balance loss very low - router may have collapsed!")
                                    print(f"   ⚠️  All images might be using same experts (no specialization)")
                                elif lb_value > 0.005:
                                    print(f"   ⚠️  WARNING: Load balance loss very high - router unstable!")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                # Save unwrapped model state dict (portable between single/multi-GPU)
                actual_model = get_actual_model(model)
                if is_main_process:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': actual_model.state_dict(),
                        'ema_state_dict': ema.shadow if ema else None,
                        'best_iou': best_iou,
                        'args': vars(args)
                    }, f"{args.checkpoint_dir}/best_model.pth")
                    print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

            history.append({'epoch': epoch, 'stage': 1, 'train_loss': train_loss, **val_metrics})

        print(f"\n✓ Stage 1 Complete. Best IoU: {best_iou:.4f}\n")
    else:
        print(f"\n⏩ Skipping Stage 1 (resuming from epoch {start_epoch})\n")

    # ========================================
    # MEMORY CLEANUP BEFORE STAGE 2
    # ========================================
    if start_epoch < args.stage1_epochs:
        # Only cleanup if we just finished Stage 1
        print("🧹 Cleaning up memory before Stage 2...")
        del optimizer, scheduler

        # Recreate DDP wrapper with find_unused_parameters=False for Stage 2
        if args.use_ddp and n_gpus > 1:
            # Unwrap model from DDP
            actual_model = model.module
            # Recreate DDP with optimized settings for Stage 2 (no unused params)
            model = DDP(actual_model, device_ids=[args.local_rank], output_device=args.local_rank,
                       find_unused_parameters=False)
            if is_main_process:
                print("   ✓ Recreated DDP wrapper with find_unused_parameters=False")

        clear_gpu_memory()
        print_gpu_memory()

    # ========================================
    # Stage 2: Full Fine-tuning with reduced batch size
    # ========================================
    print("\n" + "=" * 70)
    print("STAGE 2: FULL FINE-TUNING")
    print("=" * 70)

    # Create new dataloader with reduced batch size for Stage 2
    if args.stage2_batch_size != args.batch_size:
        print(f"🔧 Reducing batch size: {args.batch_size} → {args.stage2_batch_size}")
        # Recreate samplers with Stage 2 batch size for DDP
        if args.use_ddp and n_gpus > 1:
            train_sampler = DistributedSampler(train_data, shuffle=True)
            val_sampler = DistributedSampler(val_data, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(train_data, args.stage2_batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                  persistent_workers=True if args.num_workers > 0 else False,
                                  prefetch_factor=3,
                                  multiprocessing_context='fork' if args.num_workers > 0 else None)
        val_loader = DataLoader(val_data, args.stage2_batch_size,
                                shuffle=False,
                                sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=True if args.num_workers > 0 else False,
                                prefetch_factor=2,
                                multiprocessing_context='fork' if args.num_workers > 0 else None)

    if args.progressive_unfreeze:
        print("📈 Using progressive unfreezing strategy")
        progressive_unfreeze_backbone(model, stage=1)
    else:
        print("🔓 Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {trainable/1e6:.1f}M")

    print()
    print_gpu_memory()

    # Use stage2-specific LR if provided, otherwise use default
    stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr

    print(f"Stage 2 Learning Rate: {stage2_lr}")
    if args.warmup_epochs > 0:
        print(f"Warmup epochs: {args.warmup_epochs}")
    if args.scheduler != 'none':
        print(f"Scheduler: {args.scheduler}")
        if args.scheduler in ['cosine', 'cosine_restart']:
            print(f"  Min LR: {args.min_lr}")

    # Create optimizer with unwrapped model (DataParallel compatible)
    # In Stage 2, backbone is unfrozen
    actual_model = get_actual_model(model)

    # Separate learning rates for router (if using sparse MoE)
    if args.use_sparse_moe:
        router_params = []
        other_params = []
        for name, param in actual_model.named_parameters():
            # Router/gate parameters get 0.01× learning rate for stability
            if 'router' in name or 'gate' in name:
                router_params.append(param)
            else:
                other_params.append(param)

        # CRITICAL: Router LR = 0.01× main LR for stability
        optimizer = AdamW([
            {'params': other_params, 'lr': stage2_lr},
            {'params': router_params, 'lr': stage2_lr * 0.01}  # 100× slower for router
        ], weight_decay=args.weight_decay)

        if is_main_process:
            print(f"🎯 Sparse MoE: Router LR = {stage2_lr * 0.01:.6f} (0.01× main LR)")
            print(f"   Other params LR = {stage2_lr:.6f}")
    else:
        optimizer = AdamW(actual_model.parameters(), lr=stage2_lr, weight_decay=args.weight_decay)

    # Create scheduler for Stage 2
    total_steps = len(train_loader) * (args.epochs - args.stage1_epochs) // args.accumulation_steps
    total_epochs = args.epochs - args.stage1_epochs
    scheduler = create_scheduler(optimizer, args.scheduler,
                                 total_steps=total_steps,
                                 total_epochs=total_epochs,
                                 max_lr=stage2_lr,
                                 min_lr=args.min_lr,
                                 t_mult=args.t_mult)

    stage2_start = max(start_epoch, args.stage1_epochs)
    if stage2_start > args.stage1_epochs:
        print(f"📍 Resuming Stage 2 from epoch {stage2_start}")
        # Advance scheduler to correct position when resuming
        if scheduler is not None and args.scheduler != 'onecycle':
            steps_to_skip = stage2_start - args.stage1_epochs
            if is_main_process:
                print(f"   Advancing scheduler by {steps_to_skip} steps to sync with resumed epoch\n")
            for _ in range(steps_to_skip):
                scheduler.step()
        else:
            print()

    for epoch in range(stage2_start, args.epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        # Progressive unfreezing: gradually unfreeze more layers
        if args.progressive_unfreeze:
            stage2_progress = epoch - args.stage1_epochs
            total_stage2 = args.epochs - args.stage1_epochs
            if stage2_progress == total_stage2 // 3:
                print("\n📈 Progressive unfreeze: Stage 2/3")
                progressive_unfreeze_backbone(model, stage=2)
                # Recreate optimizer with new parameters
                actual_model = get_actual_model(model)
                optimizer = AdamW([
                    {'params': actual_model.backbone.parameters(), 'lr': stage2_lr * 0.1},
                    {'params': [p for n, p in actual_model.named_parameters() if 'backbone' not in n], 'lr': stage2_lr}
                ], weight_decay=args.weight_decay)
                clear_gpu_memory()
                print_gpu_memory()
                print()
            elif stage2_progress == 2 * total_stage2 // 3:
                print("\n📈 Progressive unfreeze: Stage 3/3 (Full)")
                progressive_unfreeze_backbone(model, stage=3)
                # Recreate optimizer with new parameters
                actual_model = get_actual_model(model)
                optimizer = AdamW([
                    {'params': actual_model.backbone.parameters(), 'lr': stage2_lr * 0.1},
                    {'params': [p for n, p in actual_model.named_parameters() if 'backbone' not in n], 'lr': stage2_lr}
                ], weight_decay=args.weight_decay)
                clear_gpu_memory()
                print_gpu_memory()
                print()

        # Apply warmup if in warmup period
        stage2_epoch = epoch - args.stage1_epochs
        in_warmup = warmup_lr(optimizer, stage2_epoch, args.warmup_epochs, stage2_lr)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                 args.accumulation_steps, ema, epoch, args.epochs, args.deep_supervision,
                                 use_amp=not args.no_amp, use_sparse_moe=args.use_sparse_moe,
                                 use_cod_specialized=args.use_cod_specialized)

        # Step scheduler if not in warmup and scheduler exists
        if not in_warmup and scheduler is not None:
            if args.scheduler == 'onecycle':
                for _ in range(len(train_loader) // args.accumulation_steps):
                    scheduler.step()
            else:
                scheduler.step()

        if ema:
            ema.apply_shadow()
        val_metrics = validate(model, val_loader, metrics, use_ddp=args.use_ddp)
        if ema:
            ema.restore()

        # Print training metrics with LR
        current_lr = optimizer.param_groups[0]['lr']
        lr_info = f" | LR: {current_lr:.6f}" if args.scheduler != 'none' or in_warmup else ""
        print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}{lr_info}")

        # Monitor Sparse MoE router health (Stage 2)
        if args.use_sparse_moe and is_main_process:
            sample_imgs, _ = next(iter(val_loader))
            sample_imgs = sample_imgs.cuda()
            actual_model = get_actual_model(model)
            with torch.no_grad():
                _, aux, _ = actual_model(sample_imgs, return_deep_supervision=False)
                if isinstance(aux, dict):
                    lb_loss = aux.get('load_balance_loss', 0.0)
                    warmup = aux.get('router_warmup_factor', 1.0)

                    if lb_loss > 0:
                        lb_value = lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss
                        print(f"   Router LB Loss: {lb_value:.6f} | Warmup: {warmup:.2f}")

                        # Warning thresholds (stricter in Stage 2)
                        if lb_value < 0.0001:
                            print(f"   ⚠️  WARNING: Load balance loss very low - router may have collapsed!")
                            print(f"   ⚠️  All images might be using same experts (no specialization)")
                        elif lb_value > 0.01:
                            print(f"   ⚠️  WARNING: Load balance loss very high - router unstable!")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            # Save unwrapped model state dict (portable between single/multi-GPU)
            actual_model = get_actual_model(model)
            if is_main_process:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': actual_model.state_dict(),
                    'ema_state_dict': ema.shadow if ema else None,
                    'best_iou': best_iou,
                    'args': vars(args)
                }, f"{args.checkpoint_dir}/best_model.pth")
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

        history.append({'epoch': epoch, 'stage': 2, 'train_loss': train_loss, **val_metrics})

    with open(f"{args.checkpoint_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Target:   0.72")
    print(f"Status:   {'✅ SOTA!' if best_iou >= 0.72 else f'Gap: {0.72 - best_iou:.4f}'}")
    print("=" * 70)

    # Cleanup DDP if used
    if args.use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    args = parse_args()
    if args.command == 'train':
        try:
            train(args)
        except Exception as e:
            # Ensure DDP cleanup even on error
            if hasattr(args, 'use_ddp') and args.use_ddp:
                cleanup_ddp()
            raise e