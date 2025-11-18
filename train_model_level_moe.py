"""
3-Stage Training for Model-Level MoE Ensemble

Target: 0.80-0.81 IoU (beat SOTA at 0.78-0.79)

Training Strategy:
  STAGE 1 (40 epochs): Train each expert independently
    - Expert 1 (SINet): 40 epochs
    - Expert 2 (PraNet): 40 epochs
    - Expert 3 (ZoomNet): 40 epochs
    - Expert 4 (UJSC): 40 epochs
    - Goal: Each expert reaches 0.73-0.76 IoU individually

  STAGE 2 (30 epochs): Train router with frozen experts
    - Freeze all experts
    - Train router to select best expert per image
    - Goal: Learn optimal routing strategy

  STAGE 3 (80 epochs): Fine-tune everything together
    - Unfreeze all parameters
    - Low learning rate
    - Goal: Ensemble reaches 0.80-0.81 IoU
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.model_level_moe import ModelLevelMoE, count_parameters
from data.dataset import COD10KDataset
from losses.advanced_loss import CODSpecializedLoss
from metrics.cod_metrics import CODMetrics
from models.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Model-Level MoE 3-Stage Training')

    # Data
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--num-workers', type=int, default=4)

    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=2)

    # Training stages
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='Training stage: 1=Experts, 2=Router, 3=Full')
    parser.add_argument('--expert-id', type=int, default=None,
                        help='For stage 1: which expert to train (0-3, or None for all)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override default epochs for stage')

    # Optimization
    parser.add_argument('--lr', type=float, default=None,
                        help='Override default LR for stage')
    parser.add_argument('--weight-decay', type=float, default=0.0001)

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_moe')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--load-experts-from', type=str, default=None,
                        help='For stage 2/3: path to trained experts')

    # System
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-ddp', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    # torchrun sets LOCAL_RANK as environment variable
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def setup_ddp(args):
    """Initialize DDP"""
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()

        # Set device based on local_rank
        print(f"[Rank {args.rank}] local_rank from args: {args.local_rank}")
        torch.cuda.set_device(args.local_rank)
        print(f"[Rank {args.rank}] torch.cuda.current_device(): {torch.cuda.current_device()}")
    else:
        args.world_size = 1
        args.rank = 0


def is_main_process(args):
    return args.rank == 0


def get_model(model):
    """Get the actual model from DDP wrapper if needed"""
    return model.module if hasattr(model, 'module') else model


def print_stage_info(stage, args):
    """Print information about current training stage"""
    if not is_main_process(args):
        return

    print("\n" + "="*70)
    if stage == 1:
        if args.expert_id is not None:
            print(f"STAGE 1: TRAINING EXPERT {args.expert_id}")
            expert_names = ["SINet-Style", "PraNet-Style", "ZoomNet-Style", "UJSC-Style"]
            print(f"  Expert: {expert_names[args.expert_id]}")
        else:
            print("STAGE 1: TRAINING ALL EXPERTS SEQUENTIALLY")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  Goal: Each expert reaches 0.73-0.76 IoU")
    elif stage == 2:
        print("STAGE 2: TRAINING ROUTER")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  Experts: FROZEN")
        print(f"  Goal: Learn optimal expert selection")
    elif stage == 3:
        print("STAGE 3: FINE-TUNING FULL ENSEMBLE")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  All parameters: TRAINABLE")
        print(f"  Goal: Ensemble reaches 0.80-0.81 IoU")
    print("="*70 + "\n")


def train_expert(expert_id, model, train_loader, val_loader, criterion, metrics, args):
    """Train a single expert (Stage 1)"""

    actual_model = get_model(model)

    # Freeze everything except the target expert
    for name, param in model.named_parameters():
        if f'expert_models.{expert_id}' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Also need backbone trainable
    for param in actual_model.backbone.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        print(f"Trainable parameters: {trainable/1e6:.1f}M")

    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    # LR Scheduler: Cosine annealing for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()

            # Forward: only this expert produces prediction
            features = actual_model.backbone(images)
            pred = actual_model.expert_models[expert_id](features)

            # Loss
            loss, _ = criterion(pred, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks = masks.cuda()

                features = actual_model.backbone(images)
                pred = actual_model.expert_models[expert_id](features)
                pred = torch.sigmoid(pred)

                metrics.update(pred, masks)

        val_metrics = metrics.compute()
        metrics.reset()

        if is_main_process(args):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f} | LR: {current_lr:.6f}")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

                # Save expert checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, f'expert_{expert_id}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'expert_id': expert_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou
                }, checkpoint_path)

        # Step LR scheduler
        scheduler.step()

    if is_main_process(args):
        print(f"\nExpert {expert_id} training complete. Best IoU: {best_iou:.4f}")

    return best_iou


def train_router(model, train_loader, val_loader, criterion, metrics, args):
    """Train router with frozen experts (Stage 2)"""

    # Freeze experts and backbone, train only router
    for name, param in model.named_parameters():
        if 'router' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        print(f"Trainable parameters (router only): {trainable/1e6:.1f}M")

    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()

            # Forward: router selects experts
            pred, routing_info = model(images, return_routing_info=True)

            # Segmentation loss
            seg_loss, _ = criterion(pred, masks)

            # CRITICAL: Router diversity loss (encourages expert specialization)
            # We want experts to be used roughly equally (prevents collapse to one expert)
            expert_probs = routing_info['expert_probs']  # [B, num_experts]
            avg_expert_usage = expert_probs.mean(dim=0)  # [num_experts]
            uniform_usage = torch.ones_like(avg_expert_usage) / args.num_experts
            diversity_loss = F.kl_div(
                avg_expert_usage.log(),
                uniform_usage,
                reduction='batchmean'
            )

            # Total loss: segmentation + diversity
            loss = seg_loss + 0.01 * diversity_loss  # Small weight for diversity

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks = masks.cuda()

                pred, routing_info = model(images, return_routing_info=True)
                pred = torch.sigmoid(pred)

                metrics.update(pred, masks)

        val_metrics = metrics.compute()
        routing_stats = routing_info['routing_stats'] if 'routing_stats' in routing_info else {}
        metrics.reset()

        if is_main_process(args):
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"IoU: {val_metrics['IoU']:.4f} | "
                  f"Entropy: {routing_stats.get('entropy', 0):.3f}")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

                # Save checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, 'router_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou
                }, checkpoint_path)

    if is_main_process(args):
        print(f"\nRouter training complete. Best IoU: {best_iou:.4f}")

    return best_iou


def train_full_ensemble(model, train_loader, val_loader, criterion, metrics, args):
    """Fine-tune full ensemble (Stage 3)"""

    actual_model = get_model(model)

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        print(f"Trainable parameters (full model): {trainable/1e6:.1f}M")

    # Optimizer with different LRs for different components
    optimizer = AdamW([
        {'params': actual_model.backbone.parameters(), 'lr': args.lr * 0.1},  # Lower LR for backbone
        {'params': actual_model.router.parameters(), 'lr': args.lr},
        {'params': actual_model.expert_models.parameters(), 'lr': args.lr * 0.5}  # Medium LR for experts
    ], weight_decay=args.weight_decay)

    # Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()

            # Forward
            pred = model(images)

            # Loss
            loss, _ = criterion(pred, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks = masks.cuda()

                pred, routing_info = model(images, return_routing_info=True)
                pred = torch.sigmoid(pred)

                metrics.update(pred, masks)

        val_metrics = metrics.compute()
        routing_stats = routing_info['routing_stats']
        metrics.reset()

        current_lr = optimizer.param_groups[0]['lr']

        if is_main_process(args):
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f} | "
                  f"LR: {current_lr:.6f}")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

                # Save checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, 'ensemble_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou
                }, checkpoint_path)

    if is_main_process(args):
        print(f"\nEnsemble training complete. Best IoU: {best_iou:.4f}")

    return best_iou


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_ddp(args)

    # Set default hyperparameters per stage
    if args.stage == 1:
        args.epochs = args.epochs or 40
        args.lr = args.lr or 0.0003
    elif args.stage == 2:
        args.epochs = args.epochs or 30
        args.lr = args.lr or 0.0002
    elif args.stage == 3:
        args.epochs = args.epochs or 80
        args.lr = args.lr or 0.00005  # Much lower for fine-tuning

    print_stage_info(args.stage, args)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    # DDP-aware caching: Each process only caches images it will actually use
    # Rank 0: caches indices [0, 2, 4, ...] → 3000 train + 400 val
    # Rank 1: caches indices [1, 3, 5, ...] → 3000 train + 400 val
    # Total: 6800 images (same as single GPU!) instead of 13,600

    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size,
        cache_in_memory=True,
        rank=args.rank if args.use_ddp else 0,
        world_size=args.world_size if args.use_ddp else 1
    )
    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='val',  # Use 'val' split (800 images) instead of 'test' (3200 images)
        img_size=args.img_size,
        cache_in_memory=True,
        rank=args.rank if args.use_ddp else 0,
        world_size=args.world_size if args.use_ddp else 1
    )

    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    target_device = args.local_rank if args.use_ddp else 0
    if is_main_process(args) or args.use_ddp:
        print(f"[Rank {args.rank}] Creating model on cuda:{target_device}")

    model = ModelLevelMoE(
        backbone=args.backbone,
        num_experts=args.num_experts,
        top_k=args.top_k,
        pretrained=True
    ).cuda(target_device)

    if is_main_process(args) or args.use_ddp:
        print(f"[Rank {args.rank}] Model created, checking device...")
        # Verify model is on correct device
        first_param_device = next(model.parameters()).device
        print(f"[Rank {args.rank}] Model parameters are on: {first_param_device}")

    # Load checkpoints if specified
    device = f'cuda:{args.local_rank if args.use_ddp else 0}'
    if args.load_experts_from and args.stage >= 2:
        if is_main_process(args):
            print(f"\nLoading trained experts from: {args.load_experts_from}")
        checkpoint = torch.load(args.load_experts_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.resume_from:
        if is_main_process(args):
            print(f"\nResuming from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # DDP
    if args.use_ddp:
        print(f"[Rank {args.rank}] Wrapping model with DDP on device_ids=[{args.local_rank}]")
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        print(f"[Rank {args.rank}] DDP wrapping successful")

    # Loss and metrics
    criterion = CODSpecializedLoss(
        bce_weight=2.0,
        iou_weight=1.5,
        edge_weight=1.0,
        boundary_weight=1.5,
        uncertainty_weight=0.3,
        reverse_attention_weight=0.8,
        aux_weight=0.0
    ).cuda()

    metrics = CODMetrics()

    # Train based on stage
    if args.stage == 1:
        if args.expert_id is not None:
            train_expert(args.expert_id, model, train_loader, val_loader, criterion, metrics, args)
        else:
            # Train all experts sequentially
            for expert_id in range(args.num_experts):
                if is_main_process(args):
                    print(f"\n{'='*70}")
                    print(f"Training Expert {expert_id}")
                    print(f"{'='*70}\n")
                train_expert(expert_id, model, train_loader, val_loader, criterion, metrics, args)

    elif args.stage == 2:
        train_router(model, train_loader, val_loader, criterion, metrics, args)

    elif args.stage == 3:
        train_full_ensemble(model, train_loader, val_loader, criterion, metrics, args)

    if args.use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
