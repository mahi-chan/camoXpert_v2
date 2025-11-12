#!/usr/bin/env python3
"""
Quick integration test for Sparse MoE architecture
Tests forward pass compatibility without requiring training
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("=" * 70)
    print("TEST 1: Checking imports...")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
        print("✅ PyTorch imported")

        from models.camoxpert_sparse_moe import CamoXpertSparseMoE
        print("✅ CamoXpertSparseMoE imported")

        from models.sparse_moe_cod import EfficientSparseCODMoE, SparseRouter
        print("✅ Sparse MoE modules imported")

        from models.cod_modules import (
            SearchIdentificationModule,
            ReverseAttentionModule,
            BoundaryUncertaintyModule,
            IterativeBoundaryRefinement
        )
        print("✅ COD modules imported")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model instantiation"""
    print("\n" + "=" * 70)
    print("TEST 2: Creating Sparse MoE model...")
    print("=" * 70)

    try:
        from models.camoxpert_sparse_moe import CamoXpertSparseMoE

        # Create model with typical settings
        model = CamoXpertSparseMoE(
            in_channels=3,
            num_classes=1,
            pretrained=False,  # Don't download weights for test
            backbone='edgenext_base',
            num_experts=6,
            top_k=2
        )

        print(f"✅ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✅ Total parameters: {total_params / 1e6:.2f}M")
        print(f"✅ Trainable parameters: {trainable_params / 1e6:.2f}M")

        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test forward pass with different configurations"""
    print("\n" + "=" * 70)
    print("TEST 3: Testing forward pass...")
    print("=" * 70)

    try:
        import torch

        # Create dummy input
        batch_size = 2
        img_size = 352
        x = torch.randn(batch_size, 3, img_size, img_size)

        print(f"Input shape: {x.shape}")

        # Test 1: Basic forward (no warmup)
        print("\n[Test 3a] Basic forward pass (no warmup)...")
        with torch.no_grad():
            pred, aux, deep = model(x, return_deep_supervision=True)

        print(f"✅ Prediction shape: {pred.shape}")
        print(f"✅ Auxiliary outputs: {type(aux)}")
        if isinstance(aux, dict):
            print(f"   - Contains load_balance_loss: {'load_balance_loss' in aux}")
            if 'load_balance_loss' in aux:
                print(f"   - Load balance loss value: {aux['load_balance_loss']}")
        print(f"✅ Deep supervision outputs: {len(deep) if deep else 0}")

        # Test 2: Forward with warmup factor
        print("\n[Test 3b] Forward pass with warmup_factor=0.5...")
        with torch.no_grad():
            pred2, aux2, deep2 = model(x, return_deep_supervision=True, warmup_factor=0.5)

        print(f"✅ Prediction shape: {pred2.shape}")
        if isinstance(aux2, dict) and 'load_balance_loss' in aux2:
            print(f"✅ Load balance loss with warmup: {aux2['load_balance_loss']}")
            print(f"   (should be ~50% of previous: {aux['load_balance_loss']})")

        # Test 3: Forward without deep supervision (validation mode)
        print("\n[Test 3c] Validation mode (no deep supervision)...")
        with torch.no_grad():
            pred3, aux3, deep3 = model(x, return_deep_supervision=False)

        print(f"✅ Prediction shape: {pred3.shape}")
        print(f"✅ Deep outputs: {deep3}")

        # Test 4: Check for NaN/Inf
        print("\n[Test 3d] Checking for NaN/Inf...")
        has_nan = torch.isnan(pred).any() or torch.isinf(pred).any()
        if has_nan:
            print(f"❌ FOUND NaN/Inf in predictions!")
            return False
        else:
            print(f"✅ No NaN/Inf in predictions")

        return True

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_router_parameters(model):
    """Test router parameter identification for gradient clipping"""
    print("\n" + "=" * 70)
    print("TEST 4: Checking router parameters...")
    print("=" * 70)

    try:
        router_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'router' in name or 'gate' in name:
                router_params.append((name, param.numel()))
            else:
                other_params.append((name, param.numel()))

        print(f"✅ Found {len(router_params)} router parameter tensors")
        print(f"✅ Found {len(other_params)} other parameter tensors")

        router_total = sum(p[1] for p in router_params)
        other_total = sum(p[1] for p in other_params)

        print(f"\n Router parameters: {router_total / 1e3:.1f}K")
        print("   Router params (first 5):")
        for name, numel in router_params[:5]:
            print(f"     - {name}: {numel:,} params")

        print(f"\n Other parameters: {other_total / 1e6:.1f}M")

        return True

    except Exception as e:
        print(f"❌ Router parameter check failed: {e}")
        traceback.print_exc()
        return False


def test_training_compatibility():
    """Test compatibility with training loop"""
    print("\n" + "=" * 70)
    print("TEST 5: Training loop compatibility...")
    print("=" * 70)

    try:
        import torch
        from models.camoxpert_sparse_moe import CamoXpertSparseMoE

        model = CamoXpertSparseMoE(
            pretrained=False,
            backbone='edgenext_base',
            num_experts=6,
            top_k=2
        )

        # Simulate training mode
        model.train()

        # Create dummy input and target
        x = torch.randn(2, 3, 352, 352)
        target = torch.rand(2, 1, 352, 352)

        # Simulate training step with warmup
        print("\n[Test 5a] Training step with warmup_factor=0.5...")
        pred, aux, deep = model(x, return_deep_supervision=True, warmup_factor=0.5)

        # Check auxiliary outputs
        if not isinstance(aux, dict):
            print("❌ Auxiliary outputs should be dict!")
            return False

        if 'load_balance_loss' not in aux:
            print("❌ load_balance_loss missing from aux dict!")
            return False

        load_balance_loss = aux['load_balance_loss']
        print(f"✅ Load balance loss: {load_balance_loss}")

        # Simulate loss computation
        print("\n[Test 5b] Simulating loss computation...")
        task_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
        total_loss = task_loss + load_balance_loss

        print(f"✅ Task loss: {task_loss.item():.4f}")
        print(f"✅ Load balance loss: {load_balance_loss.item():.6f}")
        print(f"✅ Total loss: {total_loss.item():.4f}")

        # Check if loss is finite
        if not torch.isfinite(total_loss):
            print("❌ Loss is NaN/Inf!")
            return False

        print("✅ Loss is finite")

        return True

    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("SPARSE MOE INTEGRATION TEST SUITE")
    print("=" * 70)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    if not results[0][1]:
        print("\n❌ CRITICAL: Cannot proceed without imports")
        sys.exit(1)

    # Test 2: Model creation
    model = test_model_creation()
    results.append(("Model Creation", model is not None))

    if model is None:
        print("\n❌ CRITICAL: Cannot proceed without model")
        sys.exit(1)

    # Test 3: Forward pass
    results.append(("Forward Pass", test_forward_pass(model)))

    # Test 4: Router parameters
    results.append(("Router Parameters", test_router_parameters(model)))

    # Test 5: Training compatibility
    results.append(("Training Compatibility", test_training_compatibility()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Architecture is ready for training!")
    else:
        print("❌ SOME TESTS FAILED - Fix issues before training")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
