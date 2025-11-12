# Maximum GPU Utilization Guide - Batch Size 8+

## 🎯 Problem: Only 70% GPU Usage

**Symptoms:**
- Batch size 8, but GPU only at 70%
- Training slower than it could be
- GPU has unused capacity

**Root causes:**
1. **CPU bottleneck** - Data loading too slow (most likely!)
2. **Batch size too small** - GPU can handle more
3. **I/O bottleneck** - Disk reads limiting speed

---

## ✅ SOLUTION 1: Increase Workers (Try This First!)

### Command

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_maxgpu \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 8 \
    --accumulation-steps 1 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.0004 \
    --stage2-lr 0.00024 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 12 \
    --num-workers 12
```

**Key changes:**
- ✅ `--num-workers 12` (increased from 4) ← **THIS IS THE FIX!**
- ✅ `--stage2-batch-size 8` (full batch 8)
- ✅ `--accumulation-steps 1` (less waiting between batches)

**Expected GPU usage:** 85-95%

---

## 🚀 SOLUTION 2: Increase Batch Size (If You Have Memory)

### Command

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_maxgpu \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 12 \
    --stage2-batch-size 12 \
    --accumulation-steps 1 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.00048 \
    --stage2-lr 0.00027 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 12 \
    --num-workers 12
```

**Key changes:**
- ✅ `--batch-size 12` (50% increase!)
- ✅ `--stage2-batch-size 12`
- ✅ `--lr 0.00048` (scaled for larger batch)
- ✅ `--stage2-lr 0.00027` (scaled for larger batch)
- ✅ `--num-workers 12`

**Expected GPU usage:** 90-100%

**Memory:** ~16-17GB (might be tight on Kaggle P100!)

---

## 💪 SOLUTION 3: Maximum Performance (16GB+ GPU)

### Command

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_maxgpu \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 16 \
    --stage2-batch-size 16 \
    --accumulation-steps 1 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.0006 \
    --stage2-lr 0.00030 \
    --scheduler cosine \
    --min-lr 0.00002 \
    --warmup-epochs 3 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 16 \
    --num-workers 16
```

**Key changes:**
- ✅ `--batch-size 16` (2x your current!)
- ✅ `--stage2-batch-size 16`
- ✅ `--lr 0.0006` (scaled for 2x larger batch)
- ✅ `--stage2-lr 0.00030` (scaled but safe with deep supervision)
- ✅ `--num-workers 16`
- ✅ `--warmup-epochs 3` (safety with higher LR)

**Expected GPU usage:** 95-100%

**Memory:** ~18-19GB ⚠️ **Might OOM on 16GB GPU!**

---

## 📊 Comparison Table

| Solution | Batch | Workers | Stage2 LR | GPU Usage | Memory | Speed | Risk |
|----------|-------|---------|-----------|-----------|--------|-------|------|
| **Current** | 8 | 4 | 0.00020 | 70% | 14GB | Baseline | Low |
| **Solution 1** | 8 | 12 | 0.00024 | 85-95% | 15GB | +20% | **Low** ✅ |
| **Solution 2** | 12 | 12 | 0.00027 | 90-100% | 16-17GB | +35% | Medium |
| **Solution 3** | 16 | 16 | 0.00030 | 95-100% | 18-19GB | +50% | Higher |

---

## 🔍 Why 70% GPU Usage?

### Root Cause: CPU Bottleneck

```
Timeline of one training step:

CPU: Load batch from disk     [========] 150ms
     Preprocess/augment        [====]     60ms
     Transfer to GPU           [=]        10ms
                                          ----
                                          220ms total

GPU: Forward pass              [====]     50ms
     Backward pass             [====]     50ms
     Optimizer step            [=]        10ms
                                          ----
                                          110ms total

GPU idle time: 220ms - 110ms = 110ms (50% idle!)
```

**Problem:** GPU finishes work (110ms) but waits 110ms for next batch!

**Solution:** More workers = faster data loading = less GPU idle time

---

## 💡 Why More Workers Helps

### With 4 Workers:

```
Worker 1: Loading batch 1 [==========]
Worker 2: Loading batch 2 [==========]
Worker 3: Loading batch 3 [==========]
Worker 4: Loading batch 4 [==========]

GPU processes batches slower than workers can load
→ GPU waits for data
→ 70% utilization
```

### With 12 Workers:

```
Worker 1: Loading batch 1  [====]
Worker 2: Loading batch 2  [====]
Worker 3: Loading batch 3  [====]
...
Worker 12: Loading batch 12 [====]

12 workers → 3x faster data loading
→ GPU always has data ready
→ 90%+ utilization
```

---

## 🎯 Recommended Strategy

### Step 1: Try Solution 1 (Safest)

**Command:** Solution 1 from above (batch=8, workers=12)

**Run for 1 epoch and monitor:**

```python
# Watch GPU utilization in real-time
nvidia-smi -l 1
```

**Expected:**
- GPU utilization: 85-95%
- GPU memory: ~15GB
- Speed: ~20-25% faster than current

**If GPU usage is now 90%+**: ✅ Perfect! Keep these settings.

**If GPU usage is still only 75-80%**: Try Step 2.

---

### Step 2: Try Solution 2 (If Step 1 Still Underutilizes)

**Command:** Solution 2 from above (batch=12, workers=12)

**Monitor:**
- GPU utilization should hit 90-100%
- Watch for OOM errors

**If OOM occurs:**
```bash
# Reduce to batch=10
--batch-size 10 \
--stage2-batch-size 10 \
--lr 0.00044 \
--stage2-lr 0.00026
```

---

### Step 3: Try Solution 3 (Maximum Performance)

**Only if:**
- ✅ You have 24GB GPU (not 16GB Kaggle)
- ✅ You want absolute maximum speed
- ✅ You're willing to risk OOM

---

## 🔧 Additional Optimizations

### 1. Enable Persistent Workers (PyTorch 1.13+)

Add to your DataLoader creation in train_ultimate.py:

```python
train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True  # ← ADD THIS
)
```

**Benefit:** Workers don't restart between epochs = faster!

### 2. Increase Prefetch Factor

```python
train_loader = DataLoader(
    ...,
    prefetch_factor=4  # ← ADD THIS (default is 2)
)
```

**Benefit:** Each worker loads 4 batches ahead = less waiting

### 3. Verify Data Location

**If data is on slow storage:**
```bash
# Copy dataset to faster storage if available
# On Kaggle, data is already on SSD, so this is fine
```

---

## 📈 Expected Performance Improvements

### Solution 1 (workers=12, batch=8)

```
Before:
- GPU usage: 70%
- Time per epoch: 8 minutes
- Samples/sec: ~30

After:
- GPU usage: 90%
- Time per epoch: 6.5 minutes (19% faster!)
- Samples/sec: ~37
```

### Solution 2 (workers=12, batch=12)

```
Before:
- GPU usage: 70%
- Time per epoch: 8 minutes
- Samples/sec: ~30

After:
- GPU usage: 95%
- Time per epoch: 6 minutes (25% faster!)
- Samples/sec: ~40
```

### Solution 3 (workers=16, batch=16)

```
Before:
- GPU usage: 70%
- Time per epoch: 8 minutes
- Samples/sec: ~30

After:
- GPU usage: 98%
- Time per epoch: 5 minutes (38% faster!)
- Samples/sec: ~48
```

---

## ⚠️ Watch Out For

### 1. CPU Usage

With more workers, CPU usage will increase:

```
4 workers:  ~40% CPU
12 workers: ~90% CPU (this is fine!)
16 workers: ~100% CPU (might cause slowdown if CPU is weak)
```

**On Kaggle:** CPU is usually strong enough for 12 workers

### 2. Memory (RAM) Usage

More workers = more RAM:

```
4 workers:  ~2GB RAM
12 workers: ~6GB RAM
16 workers: ~8GB RAM
```

**On Kaggle:** Should have enough RAM

### 3. OOM (GPU Memory)

**If you get OOM with larger batches:**

```bash
# Reduce batch size by 2
--batch-size 10
--stage2-batch-size 10

# Or reduce image size
--img-size 320  # instead of 384
```

---

## 🎯 LR Adjustments for Each Solution

### Solution 1 (batch=8)

```python
# Effective batch: 8
# Stage 1 LR: 0.0004
# Stage 2 LR: 0.00024 (reduced for deep supervision)
```

### Solution 2 (batch=12)

```python
# Effective batch: 12 (1.5x larger than 8)
# Stage 1 LR: 0.0004 × sqrt(1.5) = 0.00048
# Stage 2 LR: 0.00048 × 0.56 = 0.00027 (deep sup reduction)
```

### Solution 3 (batch=16)

```python
# Effective batch: 16 (2x larger than 8)
# Stage 1 LR: 0.0004 × sqrt(2) = 0.0006
# Stage 2 LR: 0.0006 × 0.5 = 0.00030 (deep sup reduction)
```

**Important:** These LRs are calibrated to:
1. Scale with batch size (sqrt rule)
2. Account for deep supervision (reduce by 25-50%)
3. Stay safe after your previous explosion

---

## 🚀 Quick Start

### Immediate Action

**Run Solution 1 right now:**

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_maxgpu \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 8 \
    --accumulation-steps 1 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.0004 \
    --stage2-lr 0.00024 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 12
```

**Monitor GPU usage:**

```bash
# In another terminal/notebook cell
watch -n 1 nvidia-smi
```

**If GPU usage is 85%+:** ✅ Success! Keep these settings.

**If still < 80%:** Try Solution 2 (batch=12)

---

## 📊 Diagnostic Commands

### Check Current Bottleneck

```python
import time
import torch

# Time data loading
start = time.time()
for batch in train_loader:
    break
data_time = time.time() - start
print(f"Data loading time: {data_time:.3f}s")

# Time GPU forward pass
batch = next(iter(train_loader))
images, masks = batch
images, masks = images.cuda(), masks.cuda()

start = time.time()
with torch.cuda.amp.autocast():
    pred, aux, deep = model(images, return_deep_supervision=True)
gpu_time = time.time() - start
print(f"GPU forward time: {gpu_time:.3f}s")

print(f"GPU utilization: {gpu_time / data_time * 100:.1f}%")
```

**Interpretation:**
- Data time > GPU time → CPU bottleneck (increase workers!)
- GPU time > Data time → GPU is the bottleneck (increase batch!)

---

## ✅ Bottom Line

**Your issue:** CPU can't load data fast enough → GPU waits → 70% usage

**Quick fix:** Increase `--num-workers 12` (Solution 1)

**Expected:** GPU usage 85-95%, training 20% faster

**If you want even more:** Try batch=12 or 16 (Solutions 2-3)

**Safety:** All solutions include safe LR adjustments to avoid another explosion

---

**Try Solution 1 first and let me know your GPU usage!** 🚀
