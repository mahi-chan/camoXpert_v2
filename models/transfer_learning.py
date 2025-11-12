import torch


def transfer_weights(small_model_path, base_model, device='cuda', verbose=True):
    """
    Transfer weights from trained small model to base model

    Strategy:
    1. Load small model weights
    2. Transfer compatible layers (decoder, SDTA, MoE)
    3. Keep pretrained backbone weights from base model
    4. Return initialized base model
    """
    print("\n" + "=" * 70)
    print("TRANSFERRING WEIGHTS: SMALL → BASE-USI")
    print("=" * 70)

    checkpoint = torch.load(small_model_path, map_location=device, weights_only=False)
    small_state_dict = checkpoint['model_state_dict']
    base_state_dict = base_model.state_dict()

    transferred = 0
    skipped = 0
    shape_mismatch = 0

    print("\nTransfer Strategy:")
    print("-" * 70)
    print("✓ Decoder layers:  Transfer (same architecture)")
    print("✓ SDTA blocks:     Transfer (feature enhancement)")
    print("✓ MoE layers:      Transfer (expert routing)")
    print("✓ Final conv:      Transfer (prediction head)")
    print("✗ Backbone:        Keep pretrained (size mismatch)")
    print("-" * 70)

    for name, param in small_state_dict.items():
        # Skip backbone layers - we want pretrained base-usi weights
        if 'backbone' in name:
            skipped += 1
            continue

        if name in base_state_dict:
            if param.shape == base_state_dict[name].shape:
                base_state_dict[name].copy_(param)
                transferred += 1
                if verbose and transferred <= 5:
                    print(f"  ✓ {name} {param.shape}")
            else:
                shape_mismatch += 1
                if verbose and shape_mismatch <= 3:
                    print(f"  ✗ Shape mismatch: {name}")
                    print(f"    Small: {param.shape} → Base: {base_state_dict[name].shape}")

    print("-" * 70)
    print(f"\n📊 Transfer Statistics:")
    print(f"  Transferred:     {transferred:4d} layers (decoder + heads)")
    print(f"  Shape mismatch:  {shape_mismatch:4d} layers")
    print(f"  Skipped:         {skipped:4d} layers (backbone)")
    print(f"  Transfer rate:   {transferred / (transferred + shape_mismatch + skipped) * 100:.1f}%")

    base_model.load_state_dict(base_state_dict, strict=False)

    print("\n✅ Weight transfer complete!")
    print("=" * 70 + "\n")

    return base_model