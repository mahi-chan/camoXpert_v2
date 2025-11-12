#!/usr/bin/env python3
"""
GPU Usage Profiler for CamoXpert Training
Profiles one epoch to determine optimal batch sizes and settings
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time
import json
import argparse
import sys
import os
from datetime import datetime
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from losses.advanced_loss import AdvancedCODLoss


def get_gpu_memory():
    """Get current GPU memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        used, total, util = result.stdout.strip().split(',')
        return {
            'used_mb': float(used),
            'total_mb': float(total),
            'used_gb': float(used) / 1024,
            'total_gb': float(total) / 1024,
            'utilization': float(util)
        }
    except:
        return None


def profile_training_epoch(dataset_path, batch_size, img_size, backbone, num_experts,
                           stage='stage1', num_batches=50, accumulation_steps=4):
    """
    Profile one epoch of training

    Args:
        dataset_path: Path to dataset
        batch_size: Batch size to test
        img_size: Image size
        backbone: Backbone name
        num_experts: Number of experts
        stage: 'stage1' or 'stage2'
        num_batches: Number of batches to profile (50 = ~10% of full epoch)
        accumulation_steps: Gradient accumulation steps
    """
    print(f"\n{'='*70}")
    print(f"PROFILING: {stage.upper()} | Batch={batch_size} | Size={img_size}")
    print(f"{'='*70}")

    # Clear GPU
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create dataset and loader
    try:
        dataset = COD10KDataset(dataset_path, 'train', img_size, augment=True)
        loader = DataLoader(dataset, batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None

    # Create model
    try:
        model = CamoXpert(3, 1, pretrained=False, backbone=backbone,
                         num_experts=num_experts).cuda()
        print(f"Model created")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

    # Configure for stage
    if stage == 'stage1':
        print("Configuring Stage 1: Freezing backbone")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        print("Configuring Stage 2: All parameters trainable")
        for param in model.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params/1e6:.1f}M trainable / {total_params/1e6:.1f}M total")

    # Create optimizer and loss
    criterion = AdvancedCODLoss(bce_weight=5.0, iou_weight=3.0, edge_weight=2.0, aux_weight=0.1)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=0.00025, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler()

    # Memory before training
    mem_before = get_gpu_memory()
    if mem_before:
        print(f"\nMemory before training:")
        print(f"  Used: {mem_before['used_gb']:.2f} GB / {mem_before['total_gb']:.2f} GB")
        print(f"  GPU Util: {mem_before['utilization']:.1f}%")

    # Profile training
    model.train()
    batch_times = []
    memory_snapshots = []
    gpu_utils = []

    print(f"\nRunning {num_batches} batches...")
    print(f"Progress: ", end='', flush=True)

    optimizer.zero_grad(set_to_none=True)

    try:
        for batch_idx, (images, masks) in enumerate(loader):
            if batch_idx >= num_batches:
                break

            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"{batch_idx}...", end='', flush=True)

            start_time = time.time()

            # Move to GPU
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)

            # Forward pass
            with torch.amp.autocast('cuda'):
                pred, aux_loss, deep = model(images, return_deep_supervision=False)
                loss, _ = criterion(pred, masks, aux_loss, None)
                loss = loss / accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_time = time.time() - start_time
            batch_times.append(batch_time)

            # Memory snapshot
            mem = get_gpu_memory()
            if mem:
                memory_snapshots.append(mem['used_gb'])
                gpu_utils.append(mem['utilization'])

        print(" Done!")

        # Get peak memory
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        # Calculate statistics
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_memory = sum(memory_snapshots) / len(memory_snapshots) if memory_snapshots else 0
        max_memory = max(memory_snapshots) if memory_snapshots else 0
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0

        # Estimate full epoch time
        batches_per_epoch = len(loader)
        estimated_epoch_time = avg_batch_time * batches_per_epoch / 60  # minutes

        results = {
            'success': True,
            'stage': stage,
            'batch_size': batch_size,
            'img_size': img_size,
            'accumulation_steps': accumulation_steps,
            'effective_batch': batch_size * accumulation_steps,
            'trainable_params_M': trainable_params / 1e6,
            'total_params_M': total_params / 1e6,
            'batches_profiled': len(batch_times),
            'batches_per_epoch': batches_per_epoch,
            'avg_batch_time_s': avg_batch_time,
            'estimated_epoch_time_min': estimated_epoch_time,
            'peak_memory_gb': peak_memory_gb,
            'avg_memory_gb': avg_memory,
            'max_memory_gb': max_memory,
            'avg_gpu_utilization': avg_gpu_util,
            'gpu_total_gb': mem_before['total_gb'] if mem_before else 0,
            'memory_margin_gb': (mem_before['total_gb'] - max_memory) if mem_before else 0
        }

        # Print results
        print(f"\n{'='*70}")
        print("PROFILING RESULTS")
        print(f"{'='*70}")
        print(f"Performance:")
        print(f"  Avg batch time:     {avg_batch_time:.3f}s")
        print(f"  Batches per epoch:  {batches_per_epoch}")
        print(f"  Est. epoch time:    {estimated_epoch_time:.1f} min")
        print(f"  Throughput:         {batch_size / avg_batch_time:.1f} samples/sec")
        print(f"\nMemory:")
        print(f"  Peak memory:        {peak_memory_gb:.2f} GB")
        print(f"  Average memory:     {avg_memory:.2f} GB")
        print(f"  Max memory:         {max_memory:.2f} GB")
        print(f"  Total GPU:          {mem_before['total_gb']:.2f} GB" if mem_before else "  N/A")
        print(f"  Memory margin:      {results['memory_margin_gb']:.2f} GB")
        print(f"\nGPU Utilization:")
        print(f"  Average:            {avg_gpu_util:.1f}%")

        # Verdict
        print(f"\nVerdict:")
        if results['memory_margin_gb'] < 0.5:
            print(f"  ⚠️  TIGHT - Only {results['memory_margin_gb']:.2f} GB margin")
        elif results['memory_margin_gb'] < 1.0:
            print(f"  ⚠️  CLOSE - {results['memory_margin_gb']:.2f} GB margin")
        elif results['memory_margin_gb'] < 2.0:
            print(f"  ✅ GOOD - {results['memory_margin_gb']:.2f} GB margin")
        else:
            print(f"  ✅ SAFE - {results['memory_margin_gb']:.2f} GB margin, can increase batch")

        if avg_gpu_util < 50:
            print(f"  ⚠️  LOW GPU UTILIZATION ({avg_gpu_util:.0f}%) - increase batch size")
        elif avg_gpu_util < 70:
            print(f"  ✅ MODERATE GPU UTILIZATION ({avg_gpu_util:.0f}%)")
        else:
            print(f"  ✅ GOOD GPU UTILIZATION ({avg_gpu_util:.0f}%)")

        return results

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ OOM: Batch size {batch_size} is too large for {stage}")
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"   Attempted to use: {peak_memory_gb:.2f} GB")
            return {
                'success': False,
                'stage': stage,
                'batch_size': batch_size,
                'error': 'OOM',
                'peak_memory_gb': peak_memory_gb
            }
        else:
            raise e
    finally:
        # Cleanup
        del model, optimizer, criterion, scaler
        torch.cuda.empty_cache()


def test_multiple_batch_sizes(dataset_path, img_size, backbone, num_experts,
                               batch_sizes=None, stages=None):
    """Test multiple batch sizes for different stages"""

    if batch_sizes is None:
        batch_sizes = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]

    if stages is None:
        stages = ['stage1', 'stage2']

    results = {}

    for stage in stages:
        print(f"\n{'='*70}")
        print(f"TESTING {stage.upper()}")
        print(f"{'='*70}")

        stage_results = []

        for batch_size in batch_sizes:
            result = profile_training_epoch(
                dataset_path, batch_size, img_size, backbone, num_experts,
                stage=stage, num_batches=50
            )

            if result:
                stage_results.append(result)

                # Stop if OOM or memory is too tight
                if not result['success']:
                    print(f"\n⚠️  Stopping tests for {stage} (OOM at batch={batch_size})")
                    break
                elif result.get('memory_margin_gb', 0) < 0.5:
                    print(f"\n⚠️  Stopping tests for {stage} (memory too tight)")
                    break

        results[stage] = stage_results

    return results


def generate_recommendations(results):
    """Generate training recommendations based on profiling results"""

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    recommendations = {}

    for stage, stage_results in results.items():
        print(f"\n{stage.upper()}:")

        # Find valid results
        valid_results = [r for r in stage_results if r['success']]

        if not valid_results:
            print("  ❌ No valid batch sizes found!")
            continue

        # Find optimal batch size (highest that's safe with good GPU util)
        optimal = None
        for r in sorted(valid_results, key=lambda x: x['batch_size'], reverse=True):
            if r['memory_margin_gb'] >= 1.0 and r['avg_gpu_utilization'] >= 50:
                optimal = r
                break

        # If no optimal, pick safest
        if optimal is None:
            optimal = max(valid_results, key=lambda x: x['memory_margin_gb'])

        # Find fastest (highest throughput with safety)
        fastest = max(
            [r for r in valid_results if r['memory_margin_gb'] >= 1.0],
            key=lambda x: x['batch_size'],
            default=optimal
        )

        recommendations[stage] = {
            'optimal_batch': optimal['batch_size'],
            'fastest_batch': fastest['batch_size'],
            'conservative_batch': valid_results[0]['batch_size'] if valid_results else 2
        }

        print(f"  Conservative (safest):   batch={recommendations[stage]['conservative_batch']}")
        print(f"  Optimal (recommended):   batch={recommendations[stage]['optimal_batch']} ⭐")
        print(f"  Fastest (aggressive):    batch={recommendations[stage]['fastest_batch']}")

        print(f"\n  With batch={optimal['batch_size']}:")
        print(f"    Epoch time:      {optimal['estimated_epoch_time_min']:.1f} min")
        print(f"    Memory usage:    {optimal['max_memory_gb']:.2f} GB / {optimal['gpu_total_gb']:.2f} GB")
        print(f"    GPU utilization: {optimal['avg_gpu_utilization']:.0f}%")
        print(f"    Effective batch: {optimal['effective_batch']}")

    # Generate training command
    if 'stage1' in recommendations and 'stage2' in recommendations:
        print(f"\n{'='*70}")
        print("RECOMMENDED TRAINING COMMAND")
        print(f"{'='*70}")

        s1_batch = recommendations['stage1']['optimal_batch']
        s2_batch = recommendations['stage2']['optimal_batch']

        # Scale learning rate based on batch size
        base_lr = 0.00025
        base_batch = 8
        scaled_lr = base_lr * (s1_batch * 4) / base_batch

        print(f"""
python train_ultimate.py train \\
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \\
    --checkpoint-dir /kaggle/working/checkpoints_sota \\
    --backbone edgenext_base \\
    --num-experts 7 \\
    --batch-size {s1_batch} \\
    --stage2-batch-size {s2_batch} \\
    --accumulation-steps 4 \\
    --img-size {valid_results[0]['img_size']} \\
    --epochs 120 \\
    --stage1-epochs 30 \\
    --lr {scaled_lr:.6f} \\
    --num-workers 4 \\
    --progressive-unfreeze

Effective batch sizes:
  Stage 1: {s1_batch} × 4 = {s1_batch * 4}
  Stage 2: {s2_batch} × 4 = {s2_batch * 4}

Learning rate scaled from base {base_lr} to {scaled_lr:.6f}
(Ratio: {scaled_lr/base_lr:.2f}x for {(s1_batch * 4) / base_batch:.2f}x batch increase)
""")

    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Profile GPU usage for CamoXpert training')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--img-size', type=int, default=384,
                       help='Image size (default: 384)')
    parser.add_argument('--backbone', type=str, default='edgenext_base',
                       help='Backbone architecture')
    parser.add_argument('--num-experts', type=int, default=7,
                       help='Number of MoE experts')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[2, 4, 6, 8, 12, 16, 24, 32, 48],
                       help='Batch sizes to test')
    parser.add_argument('--stages', type=str, nargs='+',
                       choices=['stage1', 'stage2'], default=['stage1', 'stage2'],
                       help='Stages to profile')
    parser.add_argument('--num-batches', type=int, default=50,
                       help='Number of batches to profile per test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer batch sizes')
    parser.add_argument('--output', type=str, default='gpu_profile.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.batch_sizes = [4, 8, 16, 32]
        args.num_batches = 25

    print(f"\n{'='*70}")
    print("CAMOXPERT GPU PROFILER")
    print(f"{'='*70}")
    print(f"Dataset:      {args.dataset_path}")
    print(f"Image size:   {args.img_size}")
    print(f"Backbone:     {args.backbone}")
    print(f"Experts:      {args.num_experts}")
    print(f"Batch sizes:  {args.batch_sizes}")
    print(f"Stages:       {args.stages}")
    print(f"Batches/test: {args.num_batches}")
    print(f"{'='*70}\n")

    # Check GPU
    gpu_info = get_gpu_memory()
    if gpu_info:
        print(f"GPU: {gpu_info['total_gb']:.2f} GB total")
        print(f"Currently used: {gpu_info['used_gb']:.2f} GB\n")

    # Run profiling
    results = test_multiple_batch_sizes(
        args.dataset_path, args.img_size, args.backbone, args.num_experts,
        batch_sizes=args.batch_sizes, stages=args.stages
    )

    # Generate recommendations
    recommendations = generate_recommendations(results)

    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'results': results,
        'recommendations': recommendations
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
