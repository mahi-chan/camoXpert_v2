"""
GPU Performance Profiler - Identify Training Bottlenecks
Run this to see where time is being spent
"""

import torch
import time
import sys
sys.path.insert(0, '/kaggle/working/camoXpert')

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

print("="*70)
print("GPU PERFORMANCE PROFILER")
print("="*70)

# Check GPU
if not torch.cuda.is_available():
    print("❌ No GPU available!")
    exit(1)

print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Create model
print("\n[1/5] Creating model...")
model = CamoXpert(3, 1, pretrained=False, backbone='edgenext_base', num_experts=7).cuda()
model.train()
print(f"✓ Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# Create dummy dataloader
print("\n[2/5] Setting up data loader...")
try:
    dataset = COD10KDataset(
        root='/kaggle/input/cod10k-dataset/COD10K-v3',
        split='train',
        img_size=320
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"✓ DataLoader created: {len(dataset)} samples, batch_size=16, workers=4")
    use_real_data = True
except Exception as e:
    print(f"⚠️ Could not load real data: {e}")
    print("Using dummy data for profiling...")
    use_real_data = False

# Profile model forward pass
print("\n[3/5] Profiling model forward pass...")
dummy_input = torch.randn(16, 3, 320, 320).cuda()
dummy_target = torch.randn(16, 1, 320, 320).cuda()

# Warmup
for _ in range(3):
    with torch.cuda.amp.autocast():
        pred, aux_loss, deep = model(dummy_input, return_deep_supervision=True)

torch.cuda.synchronize()

# Time forward pass
num_iters = 10
start = time.time()
for _ in range(num_iters):
    with torch.cuda.amp.autocast():
        pred, aux_loss, deep = model(dummy_input, return_deep_supervision=True)
torch.cuda.synchronize()
forward_time = (time.time() - start) / num_iters

print(f"✓ Forward pass: {forward_time*1000:.1f} ms/iter")
print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")

# Profile backward pass
print("\n[4/5] Profiling backward pass...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start = time.time()
for _ in range(num_iters):
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():
        pred, aux_loss, deep = model(dummy_input, return_deep_supervision=True)
        loss = F.binary_cross_entropy_with_logits(pred, dummy_target) + aux_loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

torch.cuda.synchronize()
backward_time = (time.time() - start) / num_iters

print(f"✓ Forward + Backward: {backward_time*1000:.1f} ms/iter")
print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")

# Profile data loading
if use_real_data:
    print("\n[5/5] Profiling data loading...")

    # Time data loading
    start = time.time()
    num_batches = 10
    for i, (images, masks) in enumerate(loader):
        if i >= num_batches:
            break
    data_load_time = (time.time() - start) / num_batches

    print(f"✓ Data loading: {data_load_time*1000:.1f} ms/batch")

    # Time data transfer to GPU
    images, masks = next(iter(loader))
    start = time.time()
    for _ in range(num_iters):
        images_gpu = images.cuda(non_blocking=True)
        masks_gpu = masks.cuda(non_blocking=True)
        torch.cuda.synchronize()
    transfer_time = (time.time() - start) / num_iters

    print(f"✓ CPU→GPU transfer: {transfer_time*1000:.1f} ms/batch")

    # Full iteration (load + transfer + forward + backward)
    print("\n[FULL ITERATION] Profiling complete training iteration...")
    start = time.time()
    num_full_iters = 5

    data_iter = iter(loader)
    for _ in range(num_full_iters):
        try:
            images, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, masks = next(data_iter)

        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            pred, aux_loss, deep = model(images, return_deep_supervision=True)
            loss = F.binary_cross_entropy_with_logits(pred, masks) + aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()

    full_iter_time = (time.time() - start) / num_full_iters
    print(f"✓ Full iteration: {full_iter_time*1000:.1f} ms")
else:
    full_iter_time = backward_time

# Analysis
print("\n" + "="*70)
print("BOTTLENECK ANALYSIS")
print("="*70)

print(f"\nTime breakdown per iteration:")
print(f"  Forward pass:       {forward_time*1000:>6.1f} ms ({forward_time/full_iter_time*100:>5.1f}%)")
print(f"  Forward+Backward:   {backward_time*1000:>6.1f} ms ({backward_time/full_iter_time*100:>5.1f}%)")
if use_real_data:
    print(f"  Data loading:       {data_load_time*1000:>6.1f} ms ({data_load_time/full_iter_time*100:>5.1f}%)")
    print(f"  CPU→GPU transfer:   {transfer_time*1000:>6.1f} ms ({transfer_time/full_iter_time*100:>5.1f}%)")
print(f"  Full iteration:     {full_iter_time*1000:>6.1f} ms (100.0%)")

# Estimate training time
print(f"\nTraining time estimates:")
batches_per_epoch = 375  # From your output
effective_batch = 16 * 8  # batch_size * accumulation_steps
actual_iters_per_epoch = batches_per_epoch / 8  # Due to gradient accumulation

estimated_time_per_epoch = (full_iter_time * actual_iters_per_epoch) / 60
print(f"  Estimated time per epoch: {estimated_time_per_epoch:.1f} minutes")
print(f"  Current actual time: 45 minutes")

if estimated_time_per_epoch < 45:
    slowdown_factor = 45 / estimated_time_per_epoch
    print(f"\n⚠️ WARNING: {slowdown_factor:.1f}x slower than expected!")
    print("  Possible causes:")
    print("  - Data loading bottleneck (increase num_workers)")
    print("  - Gradient checkpointing overhead")
    print("  - Gradient accumulation overhead")
    print("  - I/O bottleneck reading from disk")

# GPU utilization
print(f"\nGPU Memory Usage:")
print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  Peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"  Available: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
utilization = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
print(f"  Utilization: {utilization:.1f}%")

if utilization < 50:
    print("\n⚠️ WARNING: Low GPU memory utilization!")
    print("  You can increase batch_size for better performance")

# Recommendations
print("\n" + "="*70)
print("OPTIMIZATION RECOMMENDATIONS")
print("="*70)

if use_real_data and data_load_time > forward_time:
    print("\n❌ DATA LOADING IS THE BOTTLENECK!")
    print("  Current: num_workers=4")
    print("  Recommendation: Increase to 8-12 workers")
    print("  Try: --num-workers 8 or --num-workers 12")

if utilization < 50:
    print("\n⚠️ GPU UNDERUTILIZED")
    print(f"  Current batch_size: 16 (only using {utilization:.1f}% GPU memory)")
    print("  Recommendation: Increase batch size")
    print("  Try: --batch-size 24 or --batch-size 32")

if estimated_time_per_epoch > 15:
    print("\n⚠️ TRAINING IS SLOW")
    print("  Possible optimizations:")
    print("  1. Remove --gradient-checkpointing (trades memory for speed)")
    print("  2. Reduce --accumulation-steps from 8 to 4")
    print("  3. Increase --num-workers from 4 to 8")
    print("  4. Use smaller --img-size (288 instead of 320)")

print("\n" + "="*70)
print("Profiling complete! Use recommendations above to optimize.")
print("="*70)
