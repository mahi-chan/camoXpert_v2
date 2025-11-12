"""
CamoXpert - Main training script
Run with: python main.py train --dataset-path /path/to/data --batch-size 4 ...
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.camoxpert import CamoXpert, count_parameters
from data.dataset import COD10KDataset
from losses.advanced_loss import AdvancedCODLoss
from metrics.cod_metrics import CODMetrics
from models.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='CamoXpert Training')

    # Paths
    parser.add_argument('command', type=str, choices=['train'], help='Command to run')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to COD10K dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')

    # Model
    parser.add_argument('--backbone', type=str, default='edgenext_small',
                        choices=['edgenext_small', 'edgenext_base', 'edgenext_base_usi'],
                        help='Backbone architecture')
    parser.add_argument('--num-experts', type=int, default=3, help='Number of MoE experts (3-7)')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained backbone')

    # Training
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--img-size', type=int, default=192, help='Input image size')
    parser.add_argument('--epochs', type=int, default=80, help='Total epochs')
    parser.add_argument('--stage1-epochs', type=int, default=20, help='Stage 1 epochs (frozen backbone)')

    # Optimization
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.00005, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')

    # Features
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable mixed precision')
    parser.add_argument('--deep-supervision', action='store_true', default=False, help='Use deep supervision')

    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=2, help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-interval', type=int, default=10, help='Save checkpoint every N epochs')

    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, grad_clip, deep_supervision):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0

    pbar = tqdm(loader, desc="Training")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                pred, aux_loss, deep = model(images, return_deep_supervision=deep_supervision)
                loss, loss_dict = criterion(pred, masks, aux_loss, deep)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, aux_loss, deep = model(images, return_deep_supervision=deep_supervision)
            loss, loss_dict = criterion(pred, masks, aux_loss, deep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, metrics, device):
    """Validate model"""
    model.eval()
    all_metrics = []

    pbar = tqdm(loader, desc="Validation", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        pred, _, _ = model(images)
        batch_metrics = metrics.compute_all(pred, masks)
        all_metrics.append(batch_metrics)

    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0]
    }

    return avg_metrics


def force_num_experts(model, target_num_experts):
    """Force model to use specified number of experts"""
    print(f"\n🔧 Forcing model to use {target_num_experts} experts...")

    patched = 0
    for name, module in model.named_modules():
        if hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList):
            current = len(module.experts)

            if current != target_num_experts:
                if current > target_num_experts:
                    module.experts = nn.ModuleList(module.experts[:target_num_experts])

                if hasattr(module, 'num_experts'):
                    module.num_experts = target_num_experts

                if hasattr(module, 'top_k'):
                    module.top_k = max(1, target_num_experts // 2)

                patched += 1

    if patched > 0:
        print(f"✓ Patched {patched} MoE layers to use {target_num_experts} experts")

    return model


def train(args):
    """Main training function"""

    print("=" * 70)
    print("CAMOXPERT TRAINING")
    print("=" * 70)

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Print config
    print(f"\n📋 Configuration:")
    print(f"  Backbone:       {args.backbone}")
    print(f"  Num Experts:    {args.num_experts}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Image Size:     {args.img_size}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning Rate:  {args.lr}")
    print(f"  Mixed Precision: {args.use_amp}")
    print(f"  Deep Supervision: {args.deep_supervision}")

    # Estimate memory
    estimated_memory = (args.batch_size * (args.img_size / 192) ** 2 *
                        (args.num_experts / 3) * 2.5)
    print(f"\n💾 Estimated VRAM: ~{estimated_memory:.1f} GB")
    if estimated_memory > 12:
        print("⚠️  WARNING: May exceed memory. Reduce img-size, batch-size, or num-experts")

    # Load datasets
    print(f"\n📂 Loading datasets from: {args.dataset_path}")

    train_dataset = COD10KDataset(
        root_dir=args.dataset_path,
        split='train',
        img_size=args.img_size,
        augment=True
    )

    val_dataset = COD10KDataset(
        root_dir=args.dataset_path,
        split='val',
        img_size=args.img_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"✓ Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    print(f"✓ Val:   {len(val_dataset):,} samples ({len(val_loader)} batches)")

    # Create model
    print(f"\n🔨 Creating model...")
    model = CamoXpert(
        in_channels=3,
        num_classes=1,
        pretrained=args.pretrained,
        backbone=args.backbone,
        num_experts=args.num_experts
    ).to(device)

    # Force num_experts (in case it's hardcoded in model)
    model = force_num_experts(model, args.num_experts)

    total_params, trainable_params = count_parameters(model)
    print(f"✓ Total params:     {total_params / 1e6:.2f}M")
    print(f"✓ Trainable params: {trainable_params / 1e6:.2f}M")

    # Setup training
    criterion = AdvancedCODLoss()
    metrics = CODMetrics()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # History
    history = []
    best_iou = 0.0
    best_dice = 0.0

    # ========================================
    # STAGE 1: Train decoder (frozen backbone)
    # ========================================
    print(f"\n{'=' * 70}")
    print("STAGE 1: DECODER TRAINING (Frozen Backbone)")
    print(f"{'=' * 70}")

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)

    for epoch in range(args.stage1_epochs):
        print(f"\nEpoch {epoch + 1}/{args.stage1_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                 device, args.use_amp, args.grad_clip, args.deep_supervision)
        val_metrics = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"  Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | "
              f"Dice: {val_metrics['Dice_Score']:.4f}")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            best_dice = val_metrics['Dice_Score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args)
            }, f"{args.checkpoint_dir}/best_model.pth")
            print(f"  🏆 NEW BEST! IoU: {best_iou:.4f}")

        history.append({'epoch': epoch, 'stage': 1, 'train_loss': train_loss, **val_metrics})

    # ========================================
    # STAGE 2: Full fine-tuning
    # ========================================
    print(f"\n{'=' * 70}")
    print("STAGE 2: FULL MODEL FINE-TUNING")
    print(f"{'=' * 70}")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-7)

    for epoch in range(args.stage1_epochs, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                 device, args.use_amp, args.grad_clip, args.deep_supervision)
        val_metrics = validate(model, val_loader, metrics, device)
        scheduler.step()

        print(f"  Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | "
              f"Dice: {val_metrics['Dice_Score']:.4f}")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            best_dice = val_metrics['Dice_Score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args)
            }, f"{args.checkpoint_dir}/best_model.pth")
            print(f"  🏆 NEW BEST! IoU: {best_iou:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args)
            }, f"{args.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth")

        history.append({'epoch': epoch, 'stage': 2, 'train_loss': train_loss, **val_metrics})

    # Save history
    with open(f"{args.checkpoint_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Final results
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Best IoU:  {best_iou:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved: {args.checkpoint_dir}/best_model.pth")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'train':
        train(args)