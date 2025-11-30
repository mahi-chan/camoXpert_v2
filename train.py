"""
CamoXpert Training Script with Anti-Under-Segmentation Improvements

Fixes under-segmentation by:
1. TverskyLoss (beta=0.7) - penalizes false negatives
2. Positive pixel weighting (pos_weight=3)
3. EMA for better generalization
4. Stronger augmentation with mixup
5. Multi-threshold validation

Target:
- Training Val: S-measure 0.93+, IoU 0.78+, F-measure 0.88+
- Test: S-measure 0.88+, IoU 0.72+, F-measure 0.82+
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from models.model_level_moe import ModelLevelMoE
from data.dataset import COD10KDataset
from losses import CombinedLoss
from utils.ema import EMA
from metrics.cod_metrics import CODMetrics


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MixupAugmentation:
    """
    Mixup augmentation: mixes pairs of images and labels.

    Args:
        alpha: Beta distribution parameter (default: 0.2)
               Higher alpha = more mixing
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, images, masks):
        """
        Args:
            images: [B, 3, H, W]
            masks: [B, 1, H, W]

        Returns:
            Mixed images and masks
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_masks = lam * masks + (1 - lam) * masks[index]

        return mixed_images, mixed_masks


class StrongerAugmentation:
    """
    Stronger augmentation pipeline for training.

    Includes:
    - ColorJitter
    - GaussianBlur
    - RandomGrayscale
    - CoarseDropout
    """

    def __init__(self, image_size=448):
        self.image_size = image_size

    def __call__(self, image, mask):
        """Apply random augmentations"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask)

        # ColorJitter
        if random.random() > 0.3:
            image = transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )(image)

        # Gaussian Blur
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            image = transforms.GaussianBlur(kernel_size, sigma)(image)

        # Random Grayscale
        if random.random() > 0.9:
            image = transforms.Grayscale(num_output_channels=3)(image)

        # Coarse Dropout (cutout)
        if random.random() > 0.7:
            image = self._coarse_dropout(image, num_holes=8, max_h_size=32, max_w_size=32)

        return image, mask

    def _coarse_dropout(self, image, num_holes=8, max_h_size=32, max_w_size=32):
        """Apply coarse dropout (random rectangular masks)"""
        h, w = image.shape[1], image.shape[2]

        for _ in range(num_holes):
            y = random.randint(0, h - max_h_size)
            x = random.randint(0, w - max_w_size)
            h_size = random.randint(1, max_h_size)
            w_size = random.randint(1, max_w_size)

            image[:, y:y+h_size, x:x+w_size] = 0

        return image


def validate_multi_threshold(model, dataloader, device, thresholds=[0.3, 0.4, 0.5]):
    """
    Validate at multiple thresholds and return best results.

    Also computes diagnostic metrics:
    - Mean prediction confidence
    - Percentage of images with IoU > 0.7

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device
        thresholds: List of thresholds to try

    Returns:
        best_metrics: Best metrics across all thresholds
        threshold_results: Results for each threshold
        diagnostics: Diagnostic information
    """
    model.eval()

    # Collect all predictions and ground truths
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Forward pass
            output = model(images)
            logits = output['pred'] if isinstance(output, dict) else (output[0] if isinstance(output, tuple) else output)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu())
            all_gts.append(masks.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Compute diagnostics
    mean_pred_confidence = all_preds.mean().item()

    # Compute IoU for each image at threshold=0.5 to get diagnostic
    diagnostic_ious = []
    for i in range(all_preds.size(0)):
        pred_bin = (all_preds[i] > 0.5).float()
        gt_bin = (all_gts[i] > 0.5).float()

        intersection = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum() - intersection
        iou = (intersection / (union + 1e-8)).item()
        diagnostic_ious.append(iou)

    pct_high_iou = (np.array(diagnostic_ious) > 0.7).mean() * 100

    diagnostics = {
        'mean_pred_confidence': mean_pred_confidence,
        'pct_iou_above_0.7': pct_high_iou,
        'warning': mean_pred_confidence < 0.2
    }

    # Evaluate at each threshold
    threshold_results = {}

    for thresh in thresholds:
        metrics = CODMetrics()

        for i in range(all_preds.size(0)):
            metrics.update(all_preds[i], all_gts[i], threshold=thresh)

        threshold_results[thresh] = metrics.compute()

    # Find best threshold based on IoU
    best_thresh = max(threshold_results.keys(), key=lambda t: threshold_results[t]['IoU'])
    best_metrics = threshold_results[best_thresh]
    best_metrics['best_threshold'] = best_thresh

    return best_metrics, threshold_results, diagnostics


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, ema, mixup, use_mixup=True):
    """
    Train for one epoch.

    Returns:
        avg_loss: Average total loss
        loss_components: Average of each loss component
    """
    model.train()

    total_loss = 0.0
    loss_components = {
        'focal': 0.0,
        'tversky': 0.0,
        'boundary': 0.0,
        'ssim': 0.0,
        'dice': 0.0
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Apply mixup augmentation
        if use_mixup and random.random() > 0.5:
            images, masks = mixup(images, masks)

        # Forward pass with mixed precision
        with autocast():
            output = model(images)
            logits = output['pred'] if isinstance(output, dict) else (output[0] if isinstance(output, tuple) else output)

            # Compute loss
            loss, loss_dict = criterion(logits, masks)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema.update()

        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'tversky': f"{loss_dict['tversky']:.4f}"
        })

    # Compute averages
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser(description='CamoXpert Training')

    # Data
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of COD10K dataset')
    parser.add_argument('--image-size', type=int, default=448,
                       help='Input image size (default: 448)')
    parser.add_argument('--batch-size', type=int, default=6,
                       help='Batch size (default: 6)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers (default: 8)')

    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                       help='Backbone architecture (default: pvt_v2_b2)')
    parser.add_argument('--num-experts', type=int, default=4,
                       help='Number of experts (default: 4)')
    parser.add_argument('--top-k', type=int, default=2,
                       help='Number of experts to select (default: 2)')

    # Training
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs (default: 5)')

    # EMA
    parser.add_argument('--ema-decay', type=float, default=0.999,
                       help='EMA decay rate (default: 0.999)')

    # Augmentation
    parser.add_argument('--use-mixup', action='store_true', default=True,
                       help='Use mixup augmentation (default: True)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='Mixup alpha parameter (default: 0.2)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')

    # Validation
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validate every N epochs (default: 5)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"CAMOXPERT TRAINING - FIXING UNDER-SEGMENTATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"EMA decay: {args.ema_decay}")
    print(f"Mixup: {args.use_mixup} (alpha={args.mixup_alpha})")
    print(f"{'='*70}\n")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("Loading datasets...")

    train_dataset = COD10KDataset(
        root=args.data_root,
        split='train',
        image_size=args.image_size,
        augmentation=True
    )

    val_dataset = COD10KDataset(
        root=args.data_root,
        split='val',
        image_size=args.image_size,
        augmentation=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}\n")

    # Create model
    print("Creating model...")
    model = ModelLevelMoE(
        backbone_name=args.backbone,
        num_experts=args.num_experts,
        top_k=args.top_k,
        pretrained=True
    ).to(device)

    # Create EMA
    ema = EMA(model, decay=args.ema_decay)
    print(f"✓ EMA created with decay={args.ema_decay}\n")

    # Create loss function
    criterion = CombinedLoss(
        focal_weight=1.0,
        tversky_weight=2.0,  # HIGH - fixes under-segmentation
        boundary_weight=1.0,
        ssim_weight=0.5,
        dice_weight=1.0,
        pos_weight=3.0  # Weight foreground pixels 3x
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Mixup augmentation
    mixup = MixupAugmentation(alpha=args.mixup_alpha)

    # Training loop
    best_iou = 0.0
    start_epoch = 0

    # Resume if needed
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"✓ Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}\n")

    # Training history
    history = {
        'train_loss': [],
        'train_loss_components': [],
        'val_metrics': [],
        'val_diagnostics': []
    }

    print("Starting training...\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        avg_loss, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ema, mixup, args.use_mixup
        )

        print(f"\nTraining Loss: {avg_loss:.4f}")
        print("Loss Components:")
        for key, value in loss_components.items():
            print(f"  {key}: {value:.4f}")

        history['train_loss'].append(avg_loss)
        history['train_loss_components'].append(loss_components)

        # Validate
        if (epoch + 1) % args.val_freq == 0:
            print("\nValidating with EMA weights...")

            # Apply EMA weights
            ema.apply_shadow()

            # Validate at multiple thresholds
            best_metrics, threshold_results, diagnostics = validate_multi_threshold(
                model, val_loader, device, thresholds=[0.3, 0.4, 0.5]
            )

            # Restore original weights
            ema.restore()

            print(f"\nValidation Results (best threshold={best_metrics['best_threshold']}):")
            print(f"  S-measure: {best_metrics['S-measure']:.4f} ⭐")
            print(f"  F-measure: {best_metrics['F-measure']:.4f}")
            print(f"  IoU:       {best_metrics['IoU']:.4f}")
            print(f"  MAE:       {best_metrics['MAE']:.4f}")

            print(f"\nDiagnostics:")
            print(f"  Mean prediction confidence: {diagnostics['mean_pred_confidence']:.4f}")
            print(f"  % images with IoU > 0.7:    {diagnostics['pct_iou_above_0.7']:.1f}%")

            if diagnostics['warning']:
                print(f"  ⚠️  WARNING: Mean prediction < 0.2 (under-confident model)")

            history['val_metrics'].append(best_metrics)
            history['val_diagnostics'].append(diagnostics)

            # Save best model
            if best_metrics['IoU'] > best_iou:
                best_iou = best_metrics['IoU']
                print(f"\n✓ New best IoU: {best_iou:.4f}")

                # Save EMA model as best
                ema.apply_shadow()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'best_iou': best_iou,
                    'metrics': best_metrics,
                    'args': vars(args)
                }, checkpoint_dir / 'best_model.pth')
                ema.restore()

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'best_iou': best_iou,
                'args': vars(args)
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

        # Step scheduler
        scheduler.step()

        print(f"\nLearning rate: {scheduler.get_last_lr()[0]:.2e}")

    # Save final model
    print("\nSaving final model...")
    ema.apply_shadow()
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'best_iou': best_iou,
        'args': vars(args)
    }, checkpoint_dir / 'final_model.pth')

    # Save training history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
