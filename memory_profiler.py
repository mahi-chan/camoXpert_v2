"""
Memory profiler for CamoXpert training
Helps identify memory bottlenecks and optimal batch sizes
"""
import torch
import torch.nn as nn
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.camoxpert import CamoXpert


def profile_memory(batch_size, img_size, backbone, num_experts, stage='stage1'):
    """Profile memory usage for different configurations"""
    print(f"\n{'='*70}")
    print(f"PROFILING: {stage.upper()} | Batch={batch_size} | Size={img_size}")
    print(f"{'='*70}")

    # Clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    model = CamoXpert(3, 1, pretrained=False, backbone=backbone, num_experts=num_experts).cuda()

    # Configure for stage
    if stage == 'stage1':
        print("Stage 1: Freezing backbone")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        print("Stage 2: All parameters trainable")
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M parameters")

    # Create optimizer (AdamW has 2 states per parameter)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # Create dummy data
    images = torch.randn(batch_size, 3, img_size, img_size).cuda()
    masks = torch.randn(batch_size, 1, img_size, img_size).cuda()

    print(f"\nMemory after model creation:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    try:
        # Forward pass
        with torch.amp.autocast('cuda'):
            pred, aux_loss, deep = model(images, return_deep_supervision=False)
            loss = pred.mean()

        print(f"\nMemory after forward pass:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")

        # Backward pass
        loss.backward()

        print(f"\nMemory after backward pass:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        print(f"  Peak:      {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        print(f"\nMemory after optimizer step:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")

        print(f"\n✅ SUCCESS: Batch size {batch_size} is feasible for {stage}")
        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ OOM: Batch size {batch_size} is too large for {stage}")
            print(f"   Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            return False
        else:
            raise e
    finally:
        del model, optimizer, images, masks, pred, loss
        torch.cuda.empty_cache()


def find_optimal_batch_size(img_size, backbone, num_experts, stage='stage1', max_batch=16):
    """Binary search for optimal batch size"""
    print(f"\n{'='*70}")
    print(f"FINDING OPTIMAL BATCH SIZE FOR {stage.upper()}")
    print(f"{'='*70}")

    left, right = 1, max_batch
    optimal = 1

    while left <= right:
        mid = (left + right) // 2
        success = profile_memory(mid, img_size, backbone, num_experts, stage)

        if success:
            optimal = mid
            left = mid + 1
        else:
            right = mid - 1

    print(f"\n{'='*70}")
    print(f"OPTIMAL BATCH SIZE FOR {stage.upper()}: {optimal}")
    print(f"{'='*70}")
    return optimal


def main():
    parser = argparse.ArgumentParser(description='Profile memory usage for CamoXpert')
    parser.add_argument('--img-size', type=int, default=384, help='Input image size')
    parser.add_argument('--backbone', type=str, default='edgenext_base', help='Backbone architecture')
    parser.add_argument('--num-experts', type=int, default=7, help='Number of MoE experts')
    parser.add_argument('--batch-size', type=int, default=None, help='Specific batch size to test')
    parser.add_argument('--stage', type=str, choices=['stage1', 'stage2', 'both'], default='both')
    parser.add_argument('--find-optimal', action='store_true', help='Find optimal batch size')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name}")
    print(f"Total Memory: {gpu_memory:.2f} GB")

    if args.find_optimal:
        if args.stage in ['stage1', 'both']:
            find_optimal_batch_size(args.img_size, args.backbone, args.num_experts, 'stage1')

        if args.stage in ['stage2', 'both']:
            find_optimal_batch_size(args.img_size, args.backbone, args.num_experts, 'stage2')
    else:
        batch_size = args.batch_size or 2

        if args.stage in ['stage1', 'both']:
            profile_memory(batch_size, args.img_size, args.backbone, args.num_experts, 'stage1')

        if args.stage in ['stage2', 'both']:
            profile_memory(batch_size, args.img_size, args.backbone, args.num_experts, 'stage2')


if __name__ == '__main__':
    main()
