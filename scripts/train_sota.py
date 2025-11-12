"""
ULTIMATE SOTA Training - Maximum Model, Minimum Memory
Uses: Gradient Accumulation + Checkpointing + Mixed Precision
Target: IoU 0.72+ with edgenext_base + 7 experts + 416px resolution
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import json
from tqdm import tqdm
import math
import numpy as np
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, '/kaggle/working/camoXpert')

from models.camoxpert import CamoXpert, count_parameters
from data.dataset import COD10KDataset
from losses.advanced_loss import AdvancedCODLoss
from metrics.cod_metrics import CODMetrics
from models.utils import set_seed


class EMA:
    """Exponential Moving Average"""
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


def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for MoE layers
    Trades 30% speed for 40% memory savings
    """
    print("🔧 Enabling gradient checkpointing...")
    checkpointed = 0

    for name, module in model.named_modules():
        if 'moe' in name.lower() or 'sdta' in name.lower():
            # Wrap computationally expensive modules
            if hasattr(module, 'forward'):
                original_forward = module.forward

                def checkpointed_forward(self, *args, **kwargs):
                    def custom_forward(*inputs):
                        return original_forward(*inputs, **kwargs)
                    return checkpoint(custom_forward, *args, use_reentrant=False)

                module.forward = checkpointed_forward.__get__(module, type(module))
                checkpointed += 1

    print(f"✓ Checkpointed {checkpointed} modules")
    return model


def train_epoch_with_accumulation(model, loader, criterion, optimizer, scaler, ema,
                                  accumulation_steps, epoch, max_epochs):
    """
    Training with gradient accumulation

    Example: accumulation_steps=4, batch_size=2
    → Effective batch_size = 8, but only uses memory of 2
    """
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{max_epochs}")

    for batch_idx, (images, masks) in pbar:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            pred, aux_loss, deep = model(images, return_deep_supervision=True)
            loss, _ = criterion(pred, masks, aux_loss, deep)
            loss = loss / accumulation_steps  # Scale loss

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            ema.update()

        epoch_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    # Handle remaining gradients
    if len(loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, metrics):
    model.eval()
    all_metrics = []
    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images, masks = images.cuda(), masks.cuda()
        pred, _, _ = model(images)
        pred = torch.sigmoid(pred)
        all_metrics.append(metrics.compute_all(pred, masks))
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def main():
    # =====================================================================
    # CONFIG - MAXIMUM QUALITY WITH MEMORY OPTIMIZATION
    # =====================================================================
    config = {
        'backbone': 'edgenext_base',
        'num_experts': 7,               # All 7 experts!
        'batch_size': 2,                # Small batch per step
        'accumulation_steps': 4,        # Effective batch = 8
        'img_size': 384,                # High resolution
        'epochs': 120,
        'stage1_epochs': 30,
        'lr': 0.00025,
        'weight_decay': 0.0001,
        'gradient_checkpointing': True,  # Enable checkpointing
        'dataset_path': '/kaggle/input/cod10k-dataset/COD10K-v3',
        'checkpoint_dir': '/kaggle/working/checkpoints_ultimate',
        'seed': 42
    }

    set_seed(config['seed'])
    device = torch.device('cuda')

    print("\n" + "="*70)
    print("🚀 ULTIMATE CAMOXPERT TRAINING")
    print("="*70)
    print(f"Memory Optimization: ENABLED")
    print(f"  • Mixed Precision (AMP)")
    print(f"  • Gradient Accumulation (4x)")
    print(f"  • Gradient Checkpointing")
    print(f"\nConfiguration:")
    print(f"  Backbone:        {config['backbone']}")
    print(f"  Experts:         {config['num_experts']} (Top-3)")
    print(f"  Resolution:      {config['img_size']}px")
    print(f"  Batch Size:      {config['batch_size']} × {config['accumulation_steps']} = {config['batch_size'] * config['accumulation_steps']} effective")
    print(f"  Epochs:          {config['epochs']}")
    print(f"\n🎯 Target: IoU ≥ 0.72")
    print(f"📊 Estimated VRAM: ~14GB (safe for 16GB)")
    print("="*70 + "\n")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Datasets
    print("Loading datasets...")
    train_data = COD10KDataset(config['dataset_path'], 'train', config['img_size'], augment=True)
    val_data = COD10KDataset(config['dataset_path'], 'val', config['img_size'], augment=False)

    train_loader = DataLoader(train_data, config['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, config['batch_size'], shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"Effective batch: {config['batch_size']} × {config['accumulation_steps']} = {config['batch_size'] * config['accumulation_steps']}\n")

    # Model
    print("Creating model...")
    model = CamoXpert(3, 1, pretrained=True, backbone=config['backbone'],
                     num_experts=config['num_experts']).cuda()

    # Apply gradient checkpointing
    if config['gradient_checkpointing']:
        model = enable_gradient_checkpointing(model)

    total, trainable = count_parameters(model)
    print(f"✓ Model: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)\n")

    # Training components
    criterion = AdvancedCODLoss(bce_weight=5.0, iou_weight=3.0, edge_weight=2.0, aux_weight=0.1)
    metrics = CODMetrics()
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model)

    best_iou = 0.0
    history = []

    # ========== STAGE 1: Decoder Training ==========
    print("="*70)
    print("STAGE 1: DECODER TRAINING (30 epochs)")
    print("="*70)

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=config['lr'], weight_decay=config['weight_decay'])

    total_steps = len(train_loader) * config['stage1_epochs'] // config['accumulation_steps']
    scheduler = OneCycleLR(optimizer, max_lr=config['lr'], total_steps=total_steps, pct_start=0.1)

    for epoch in range(config['stage1_epochs']):
        train_loss = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, scaler, ema,
            config['accumulation_steps'], epoch, config['stage1_epochs']
        )

        # Step scheduler after each accumulated step
        for _ in range(len(train_loader) // config['accumulation_steps']):
            scheduler.step()

        ema.apply_shadow()
        val_metrics = validate(model, val_loader, metrics)
        ema.restore()

        print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow,
                'best_iou': best_iou,
                'config': config
            }, f"{config['checkpoint_dir']}/best_model.pth")
            print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

        history.append({'epoch': epoch, 'stage': 1, 'train_loss': train_loss, **val_metrics})

        # Memory stats
        if epoch % 10 == 0:
            print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB peak")
            torch.cuda.reset_peak_memory_stats()

    print(f"\n✓ Stage 1 Complete. Best IoU: {best_iou:.4f}\n")

    # ========== STAGE 2: Full Fine-tuning ==========
    print("="*70)
    print("STAGE 2: FULL FINE-TUNING (90 epochs)")
    print("="*70)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': config['lr'] * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])

    total_steps = len(train_loader) * (config['epochs'] - config['stage1_epochs']) // config['accumulation_steps']
    scheduler = OneCycleLR(optimizer, max_lr=config['lr'], total_steps=total_steps, pct_start=0.1)

    for epoch in range(config['stage1_epochs'], config['epochs']):
        train_loss = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, scaler, ema,
            config['accumulation_steps'], epoch, config['epochs']
        )

        for _ in range(len(train_loader) // config['accumulation_steps']):
            scheduler.step()

        ema.apply_shadow()
        val_metrics = validate(model, val_loader, metrics)
        ema.restore()

        print(f"Loss: {train_loss:.4f} | IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f}")

        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow,
                'best_iou': best_iou,
                'config': config
            }, f"{config['checkpoint_dir']}/best_model.pth")
            print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

        history.append({'epoch': epoch, 'stage': 2, 'train_loss': train_loss, **val_metrics})

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow,
                'best_iou': best_iou
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")

    # Save history
    with open(f"{config['checkpoint_dir']}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Best IoU:     {best_iou:.4f}")
    print(f"Target SOTA:  0.72")
    print(f"Status:       {'✅ SOTA ACHIEVED!' if best_iou >= 0.72 else f'Gap: {0.72-best_iou:.4f}'}")
    print(f"\nFinal metrics:")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()