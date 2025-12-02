"""
Train Single SINet Expert - Diagnostic Script

Purpose: Verify the loss function works before using full MoE
Target: IoU > 0.65 on validation

This script trains a simple PVT-v2-b2 + SINet decoder to verify:
1. TverskyLoss (beta=0.7) correctly penalizes false negatives
2. Loss function achieves good IoU
3. Model predictions are confident (mean > 0.2)

If this works, we can confidently use the loss in full MoE training.
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from models.expert_architectures import SINetExpert
from data.dataset import COD10KDataset
from losses.tversky_loss import CombinedSingleExpertLoss
from metrics.cod_metrics import CODMetrics
import timm


class SimpleSINet(nn.Module):
    """
    Simple wrapper: PVT-v2-b2 backbone + SINet decoder

    This is equivalent to using just one expert from the MoE,
    but without the router overhead.
    """

    def __init__(self, backbone_name='pvt_v2_b2', pretrained=True):
        super().__init__()

        print(f"\n{'='*70}")
        print("SIMPLE SINET MODEL")
        print(f"{'='*70}")
        print(f"  Backbone: {backbone_name}")
        print(f"  Decoder: SINet-style")
        print(f"{'='*70}\n")

        # Create backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        # Get feature dimensions
        self.feature_dims = [64, 128, 320, 512]  # PVT-v2-b2

        # Create SINet decoder
        self.decoder = SINetExpert(self.feature_dims)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"✓ Total parameters: {total_params/1e6:.1f}M")
        print(f"✓ Trainable parameters: {trainable_params/1e6:.1f}M\n")

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            pred: [B, 1, H, W] - main prediction
            aux_preds: list of auxiliary predictions for deep supervision
        """
        # Extract features
        features = self.backbone(x)  # [f1, f2, f3, f4]

        # Decode
        pred, aux_preds = self.decoder(features)

        return pred, aux_preds


def validate_multi_threshold(model, dataloader, device, thresholds=[0.3, 0.4, 0.5]):
    """
    Validate at multiple thresholds and return best results.

    Returns:
        best_metrics: Best metrics across all thresholds
        diagnostics: Prediction statistics
    """
    model.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Forward
            pred, _ = model(images)
            pred = torch.sigmoid(pred)

            all_preds.append(pred.cpu())
            all_gts.append(masks.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Diagnostics
    mean_pred = all_preds.mean().item()
    min_pred = all_preds.min().item()
    max_pred = all_preds.max().item()

    diagnostics = {
        'mean_pred': mean_pred,
        'min_pred': min_pred,
        'max_pred': max_pred,
        'warning': mean_pred < 0.2
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


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    loss_components = {
        'tversky': 0.0,
        'bce': 0.0,
        'ssim': 0.0
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Forward with AMP
        with autocast():
            pred, aux_preds = model(images)
            loss, loss_dict = criterion(pred, masks, aux_preds)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Accumulate
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'tversky': f"{loss_dict.get('tversky', 0):.4f}"
        })

    # Compute averages
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser(description='Train Single SINet Expert')

    # Data
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of COD10K dataset')
    parser.add_argument('--image-size', type=int, default=448,
                       help='Input image size (default: 448)')
    parser.add_argument('--batch-size', type=int, default=6,
                       help='Batch size (default: 6)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of workers (default: 8)')

    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                       help='Backbone (default: pvt_v2_b2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone')

    # Training
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs (default: 150)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs (default: 5)')

    # Loss
    parser.add_argument('--tversky-weight', type=float, default=2.0,
                       help='Tversky loss weight (default: 2.0)')
    parser.add_argument('--bce-weight', type=float, default=1.0,
                       help='BCE loss weight (default: 1.0)')
    parser.add_argument('--ssim-weight', type=float, default=0.5,
                       help='SSIM loss weight (default: 0.5)')
    parser.add_argument('--pos-weight', type=float, default=3.0,
                       help='Positive pixel weight (default: 3.0)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_single',
                       help='Checkpoint directory')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save every N epochs (default: 10)')
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validate every N epochs (default: 5)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("SINGLE SINET EXPERT TRAINING - DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"\nLoss weights:")
    print(f"  Tversky: {args.tversky_weight} ⭐")
    print(f"  BCE:     {args.bce_weight} (pos_weight={args.pos_weight})")
    print(f"  SSIM:    {args.ssim_weight}")
    print(f"{'='*70}\n")

    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print("Loading datasets...")

    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.image_size,
        augment=True,
        cache_in_memory=True
    )

    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='val',
        img_size=args.image_size,
        augment=False,
        cache_in_memory=True
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
    model = SimpleSINet(
        backbone_name=args.backbone,
        pretrained=args.pretrained
    ).to(device)

    # Create loss
    criterion = CombinedSingleExpertLoss(
        tversky_weight=args.tversky_weight,
        bce_weight=args.bce_weight,
        ssim_weight=args.ssim_weight,
        pos_weight=args.pos_weight
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler with warmup
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=1,
        eta_min=args.lr * 0.01
    )

    # AMP scaler
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'val_metrics': [],
        'diagnostics': []
    }

    best_iou = 0.0

    print("Starting training...\n")

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        avg_loss, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        print(f"\nTraining Loss: {avg_loss:.4f}")
        print("Loss Components:")
        for key, value in loss_components.items():
            print(f"  {key}: {value:.4f}")

        history['train_loss'].append(avg_loss)

        # Step scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nLearning rate: {current_lr:.2e}")

        # Validate
        if (epoch + 1) % args.val_freq == 0:
            print("\nValidating...")

            best_metrics, threshold_results, diagnostics = validate_multi_threshold(
                model, val_loader, device, thresholds=[0.3, 0.4, 0.5]
            )

            print(f"\nValidation Results (best threshold={best_metrics['best_threshold']}):")
            print(f"  S-measure: {best_metrics['S-measure']:.4f}")
            print(f"  F-measure: {best_metrics['F-measure']:.4f}")
            print(f"  IoU:       {best_metrics['IoU']:.4f} {'⭐' if best_metrics['IoU'] > 0.65 else ''}")
            print(f"  MAE:       {best_metrics['MAE']:.4f}")

            print(f"\nPrediction Statistics:")
            print(f"  Mean: {diagnostics['mean_pred']:.4f}")
            print(f"  Min:  {diagnostics['min_pred']:.4f}")
            print(f"  Max:  {diagnostics['max_pred']:.4f}")

            if diagnostics['warning']:
                print(f"  ⚠️  WARNING: Mean prediction < 0.2 (under-confident model)")

            history['val_metrics'].append(best_metrics)
            history['diagnostics'].append(diagnostics)

            # Save best model
            if best_metrics['IoU'] > best_iou:
                best_iou = best_metrics['IoU']
                print(f"\n✓ New best IoU: {best_iou:.4f}")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'metrics': best_metrics,
                    'args': vars(args)
                }, checkpoint_dir / 'best_model.pth')

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args)
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Save final model
    print("\nSaving final model...")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'best_iou': best_iou,
        'args': vars(args)
    }, checkpoint_dir / 'final_model.pth')

    # Save history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Target:   0.6500 {'✓' if best_iou >= 0.65 else '✗'}")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"{'='*70}\n")

    if best_iou >= 0.65:
        print("✅ SUCCESS! Loss function works correctly.")
        print("   You can now use it in full MoE training.\n")
    else:
        print("⚠️  IoU below target. Consider:")
        print("   - Increasing tversky_weight to 3.0")
        print("   - Increasing pos_weight to 5.0")
        print("   - Training for more epochs\n")


if __name__ == '__main__':
    main()
