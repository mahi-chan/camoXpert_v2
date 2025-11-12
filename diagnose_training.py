#!/usr/bin/env python3
"""
Diagnostic script to identify training issues
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.camoxpert import CamoXpert
from losses.advanced_loss import AdvancedCODLoss
from data.dataset import COD10KDataset
from torch.utils.data import DataLoader

def test_model_output(batch_size=2, img_size=384):
    """Test if model produces valid outputs"""
    print("\n" + "="*70)
    print("TEST 1: Model Output Validation")
    print("="*70)

    model = CamoXpert(3, 1, pretrained=False, backbone='edgenext_base', num_experts=7).cuda()
    model.eval()

    # Create dummy input
    x = torch.randn(batch_size, 3, img_size, img_size).cuda()

    with torch.no_grad():
        pred, aux_loss, deep = model(x, return_deep_supervision=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Output min: {pred.min().item():.4f}")
    print(f"Output max: {pred.max().item():.4f}")
    print(f"Output mean: {pred.mean().item():.4f}")
    print(f"Aux loss: {aux_loss.item() if aux_loss is not None else 'None'}")
    print(f"Deep supervision: {len(deep) if deep else 0} outputs")

    # Check if output is all zeros or all same value
    if torch.all(pred == pred[0,0,0,0]):
        print("❌ WARNING: Model outputs constant value!")
        return False

    # Check for NaN or Inf
    if torch.isnan(pred).any():
        print("❌ ERROR: Model outputs contain NaN!")
        return False

    if torch.isinf(pred).any():
        print("❌ ERROR: Model outputs contain Inf!")
        return False

    print("✅ Model outputs look valid")
    return True


def test_loss_function():
    """Test loss function"""
    print("\n" + "="*70)
    print("TEST 2: Loss Function Validation")
    print("="*70)

    criterion = AdvancedCODLoss(bce_weight=5.0, iou_weight=3.0, edge_weight=2.0, aux_weight=0.1)

    # Create dummy predictions and targets
    pred = torch.randn(2, 1, 320, 320).cuda()  # Logits
    target = torch.randint(0, 2, (2, 1, 320, 320)).float().cuda()
    aux_loss = torch.tensor(0.1).cuda()

    loss, loss_dict = criterion(pred, target, aux_loss, None)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")

    # Check for NaN
    if torch.isnan(loss):
        print("❌ ERROR: Loss is NaN!")
        return False

    # Check if loss is reasonable
    if loss.item() > 100 or loss.item() < 0:
        print(f"⚠️  WARNING: Loss value {loss.item():.4f} seems unusual")

    print("✅ Loss function works")
    return True


def test_gradient_flow(lr=0.00025, batch_size=2):
    """Test if gradients flow properly"""
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow Check")
    print("="*70)

    model = CamoXpert(3, 1, pretrained=False, backbone='edgenext_base', num_experts=7).cuda()
    criterion = AdvancedCODLoss(bce_weight=5.0, iou_weight=3.0, edge_weight=2.0, aux_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Freeze backbone (Stage 1)
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Create dummy data
    x = torch.randn(batch_size, 3, 384, 384).cuda()
    target = torch.randint(0, 2, (batch_size, 1, 384, 384)).float().cuda()

    # Forward
    pred, aux_loss, deep = model(x, return_deep_supervision=True)
    loss, _ = criterion(pred, target, aux_loss, deep)

    # Backward
    loss.backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 100:
                print(f"⚠️  Large gradient in {name}: {grad_norm:.2f}")

    if not grad_norms:
        print("❌ ERROR: No gradients computed!")
        return False

    avg_grad = sum(grad_norms) / len(grad_norms)
    max_grad = max(grad_norms)
    min_grad = min(grad_norms)

    print(f"Gradient norms - Min: {min_grad:.6f}, Avg: {avg_grad:.6f}, Max: {max_grad:.6f}")

    if max_grad > 1000:
        print("❌ ERROR: Gradient explosion detected!")
        return False

    if avg_grad < 1e-8:
        print("❌ ERROR: Vanishing gradients detected!")
        return False

    optimizer.step()
    print("✅ Gradients flow properly")
    return True


def test_hyperparameter_config(batch_size, lr, accum_steps):
    """Test if hyperparameters are configured correctly"""
    print("\n" + "="*70)
    print("TEST 4: Hyperparameter Configuration")
    print("="*70)

    effective_batch = batch_size * accum_steps
    effective_lr = lr / batch_size  # LR per sample

    print(f"Batch size: {batch_size}")
    print(f"Accumulation steps: {accum_steps}")
    print(f"Effective batch: {effective_batch}")
    print(f"Learning rate: {lr}")
    print(f"Effective LR per sample: {effective_lr:.8f}")

    # Compare with known good config
    good_batch = 8  # 2 * 4
    good_lr = 0.00025
    good_lr_per_sample = good_lr / good_batch

    print(f"\nKnown good config:")
    print(f"  Effective batch: {good_batch}")
    print(f"  Learning rate: {good_lr}")
    print(f"  LR per sample: {good_lr_per_sample:.8f}")

    # Calculate ratio
    lr_ratio = effective_lr / good_lr_per_sample
    batch_ratio = effective_batch / good_batch

    print(f"\nYour config vs good config:")
    print(f"  Batch size ratio: {batch_ratio:.2f}x")
    print(f"  LR per sample ratio: {lr_ratio:.2f}x")

    if lr_ratio < 0.5:
        print(f"⚠️  WARNING: Learning rate is {1/lr_ratio:.1f}x too small!")
        print(f"   Recommended LR: {lr * (1/lr_ratio):.6f}")
        return False

    if lr_ratio > 2.0:
        print(f"⚠️  WARNING: Learning rate is {lr_ratio:.1f}x too large!")
        return False

    print("✅ Hyperparameters look reasonable")
    return True


def test_ema_with_frozen_backbone():
    """Test EMA behavior with frozen backbone"""
    print("\n" + "="*70)
    print("TEST 5: EMA with Frozen Backbone")
    print("="*70)

    from train_ultimate import EMA

    model = CamoXpert(3, 1, pretrained=False, backbone='edgenext_base', num_experts=7).cuda()

    # Initialize EMA before freezing
    ema = EMA(model)
    initial_shadow_keys = set(ema.shadow.keys())
    print(f"EMA initialized with {len(initial_shadow_keys)} parameters")

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Count trainable params
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"After freezing: {trainable} trainable parameters")

    # Simulate training update
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    x = torch.randn(2, 3, 384, 384).cuda()
    target = torch.rand(2, 1, 384, 384).cuda()

    pred, _, _ = model(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Update EMA
    try:
        ema.update()
        print("✅ EMA update succeeded with frozen backbone")
    except KeyError as e:
        print(f"❌ ERROR: EMA update failed: {e}")
        return False

    # Unfreeze and test again
    for param in model.parameters():
        param.requires_grad = True

    try:
        ema.update()
        print("✅ EMA update succeeded with unfrozen backbone")
    except KeyError as e:
        print(f"❌ ERROR: EMA update failed after unfreezing: {e}")
        return False

    return True


def main():
    print("="*70)
    print("CAMOXPERT TRAINING DIAGNOSTICS")
    print("="*70)

    results = []

    # Test 1: Model outputs
    results.append(("Model Output", test_model_output()))

    # Test 2: Loss function
    results.append(("Loss Function", test_loss_function()))

    # Test 3: Gradient flow
    results.append(("Gradient Flow", test_gradient_flow()))

    # Test 4: Hyperparameters (use user's failed config)
    results.append(("Hyperparameters", test_hyperparameter_config(
        batch_size=16, lr=0.0001, accum_steps=8
    )))

    # Test 5: EMA
    results.append(("EMA", test_ema_with_frozen_backbone()))

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✅ All tests passed! Code is working correctly.")
        print("   The training issue is likely due to hyperparameter configuration.")
    else:
        print("\n❌ Some tests failed! There may be code issues to fix.")

    print("="*70)


if __name__ == '__main__':
    main()
