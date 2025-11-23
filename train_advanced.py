"""
Advanced Training Script with All New Modules Integrated

Integrates:
1. OptimizedTrainer - Advanced training framework
2. CompositeLoss - Multi-component loss system
3. Enhanced experts with new modules
4. All optimizations from new components

Usage:
    torchrun --nproc_per_node=2 train_advanced.py \
        --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
        --epochs 100 \
        --batch-size 16 \
        --use-ddp
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from pathlib import Path

# Import new modules
from trainers.optimized_trainer import OptimizedTrainer
from losses.composite_loss import CompositeLossSystem
from data.dataset import COD10KDataset
from metrics.cod_metrics import CODMetrics
from models.model_level_moe import ModelLevelMoE
from models.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Advanced COD Training with New Modules')

    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to COD10K-v3 dataset')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')

    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                        choices=['pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5'],
                        help='Backbone architecture (default: pvt_v2_b2)')
    parser.add_argument('--num-experts', type=int, default=4,
                        help='Number of experts in MoE (default: 4)')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Top-k experts to use (default: 2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone weights')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Train backbone from scratch')
    parser.add_argument('--deep-supervision', action='store_true', default=True,
                        help='Enable deep supervision in model')
    parser.add_argument('--no-deep-supervision', action='store_false', dest='deep_supervision',
                        help='Disable deep supervision')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                        help='Gradient accumulation steps')

    # OptimizedTrainer settings
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs for cosine scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                        help='Disable AMP')

    # Progressive augmentation
    parser.add_argument('--enable-progressive-aug', action='store_true', default=True,
                        help='Enable progressive augmentation')
    parser.add_argument('--aug-transition-epoch', type=int, default=20,
                        help='Epoch to start increasing augmentation')

    # CompositeLoss settings
    parser.add_argument('--loss-scheme', type=str, default='progressive',
                        choices=['progressive', 'full'],
                        help='Loss weighting scheme (default: progressive)')
    parser.add_argument('--boundary-lambda-start', type=float, default=0.5,
                        help='Starting weight for boundary loss (default: 0.5)')
    parser.add_argument('--boundary-lambda-end', type=float, default=2.0,
                        help='Ending weight for boundary loss (default: 2.0)')
    parser.add_argument('--frequency-weight', type=float, default=1.5,
                        help='Weight for frequency loss (default: 1.5)')
    parser.add_argument('--scale-small-weight', type=float, default=2.0,
                        help='Weight for small object scale loss (default: 2.0)')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.5,
                        help='Threshold for uncertainty loss (default: 0.5)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_advanced',
                        help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint')

    # Distributed
    parser.add_argument('--use-ddp', action='store_true', default=False,
                        help='Use DistributedDataParallel')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for DDP')

    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def setup_ddp(args):
    """Setup distributed training."""
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()
    else:
        args.world_size = 1

    return args.local_rank == 0  # is_main_process


def create_dataloaders(args, is_main_process):
    """Create train and validation dataloaders."""

    # Training dataset
    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size,
        augment=True,  # Built-in augmentations
        cache_in_memory=False  # Don't cache in Kaggle (limited RAM)
    )

    # Validation dataset
    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='test',
        img_size=args.img_size,
        augment=False,
        cache_in_memory=False
    )

    # Samplers for DDP
    if args.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if is_main_process:
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")

    return train_loader, val_loader, train_sampler


def create_model(args, device, is_main_process):
    """Create model with all enhancements."""

    model = ModelLevelMoE(
        backbone=args.backbone,
        num_experts=args.num_experts,
        top_k=args.top_k,
        pretrained=args.pretrained,
        use_deep_supervision=args.deep_supervision
    )

    model = model.to(device)

    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True  # For MoE models
        )

    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created: {args.backbone}")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

    return model


def create_optimizer_and_criterion(model, args, is_main_process):
    """Create optimizer and advanced loss."""

    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Advanced CompositeLoss (progressive weighting is always enabled)
    criterion = CompositeLossSystem(
        total_epochs=args.epochs,
        use_boundary=True,
        use_frequency=True,
        use_scale_adaptive=True,
        use_uncertainty=True,
        boundary_lambda_start=args.boundary_lambda_start,
        boundary_lambda_end=args.boundary_lambda_end,
        frequency_weight=args.frequency_weight,
        scale_small_weight=args.scale_small_weight,
        uncertainty_threshold=args.uncertainty_threshold
    )

    if is_main_process:
        print(f"✓ Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
        print(f"✓ Loss: CompositeLoss ({args.loss_scheme} scheme)")

    return optimizer, criterion


def compute_metrics(predictions, targets):
    """Compute validation metrics."""
    metrics = CODMetrics()

    # Threshold predictions
    preds_binary = (torch.sigmoid(predictions) > 0.5).float()

    # Compute metrics
    mae = torch.abs(preds_binary - targets).mean()
    iou = metrics.compute_iou(preds_binary, targets)
    f_measure = metrics.compute_f_measure(preds_binary, targets)

    return {
        'val_mae': mae.item(),
        'val_iou': iou.item(),
        'val_f_measure': f_measure.item()
    }


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    is_main_process = setup_ddp(args)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print("=" * 80)
        print(" " * 20 + "ADVANCED COD TRAINING")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Data root: {args.data_root}")
        print(f"  Batch size: {args.batch_size} (per GPU)")
        print(f"  Accumulation steps: {args.accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.accumulation_steps * args.world_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Image size: {args.img_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Mixed precision: {args.use_amp}")
        print(f"  Progressive augmentation: {args.enable_progressive_aug}")
        print(f"  DDP: {args.use_ddp} (world size: {args.world_size})")
        print()

    # Create checkpoint directory
    if is_main_process:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, train_sampler = create_dataloaders(args, is_main_process)

    # Model
    model = create_model(args, device, is_main_process)

    # Optimizer and criterion
    optimizer, criterion = create_optimizer_and_criterion(model, args, is_main_process)

    # OptimizedTrainer with all features
    trainer = OptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.use_amp,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        max_lr=args.lr,
        num_experts=args.num_experts if args.num_experts > 1 else None,
        enable_load_balancing=True if args.num_experts > 1 else False,
        enable_collapse_detection=True if args.num_experts > 1 else False,
        enable_progressive_aug=args.enable_progressive_aug,
        aug_transition_epoch=args.aug_transition_epoch
    )

    if is_main_process:
        print("✓ OptimizedTrainer initialized with:")
        print(f"  - Cosine annealing with {args.warmup_epochs}-epoch warmup")
        print(f"  - Mixed precision: {args.use_amp}")
        print(f"  - Gradient accumulation: {args.accumulation_steps} steps")
        print(f"  - Progressive augmentation: {args.enable_progressive_aug}")
        if args.num_experts > 1:
            print(f"  - MoE load balancing: Enabled")
            print(f"  - Expert collapse detection: Enabled")
        print()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_iou = 0.0

    if args.resume_from and os.path.exists(args.resume_from):
        if is_main_process:
            print(f"Resuming from: {args.resume_from}")
        start_epoch = trainer.load_checkpoint(args.resume_from)
        start_epoch += 1

    # Training loop
    if is_main_process:
        print("=" * 80)
        print("Starting training...")
        print("=" * 80)
        print()

    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if args.use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Update CompositeLoss for current epoch
        criterion.update_epoch(epoch, args.epochs)

        # Train one epoch
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch=epoch,
            log_interval=20
        )

        # Validate
        val_metrics = trainer.validate(
            val_loader,
            metrics_fn=compute_metrics
        )

        # Print results (main process only)
        if is_main_process:
            print(f"\nEpoch [{epoch}/{args.epochs}] Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val IoU: {val_metrics['val_iou']:.4f}")
            print(f"  Val F-measure: {val_metrics['val_f_measure']:.4f}")
            print(f"  Val MAE: {val_metrics['val_mae']:.4f}")
            print(f"  Learning Rate: {train_metrics['lr']:.6f}")

            if args.enable_progressive_aug:
                print(f"  Aug Strength: {trainer.augmentation.current_strength:.3f}")

            # MoE statistics
            if 'load_balance_loss' in train_metrics:
                print(f"  Load Balance Loss: {train_metrics['load_balance_loss']:.6f}")

            if 'collapse_collapsed' in train_metrics and train_metrics['collapse_collapsed']:
                print(f"  ⚠ Expert collapse detected!")

            print()

        # Save checkpoints (main process only)
        if is_main_process:
            # Save best model
            current_iou = val_metrics['val_iou']
            if current_iou > best_iou:
                best_iou = current_iou
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                trainer.save_checkpoint(best_path, epoch, val_metrics)
                print(f"✓ Saved best model (IoU: {best_iou:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % args.save_interval == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth')
                trainer.save_checkpoint(ckpt_path, epoch, val_metrics)
                print(f"✓ Saved checkpoint: {ckpt_path}")

            # Always save latest
            latest_path = os.path.join(args.checkpoint_dir, 'latest.pth')
            trainer.save_checkpoint(latest_path, epoch, val_metrics)

    # Final summary
    if is_main_process:
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)
        print(f"Best validation IoU: {best_iou:.4f}")
        print(f"Checkpoints saved to: {args.checkpoint_dir}")

        summary = trainer.get_training_summary()
        print(f"\nFinal Summary:")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")

    # Cleanup
    if args.use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
