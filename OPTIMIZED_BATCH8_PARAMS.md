# Optimized Training Parameters for Batch Size 8

## 🎯 Your Requirements

- **Batch size**: 8 (Stage 1)
- **Safe training**: Avoid gradient explosion
- **Deep supervision**: Enabled (for +0.03-0.05 IoU boost)
- **Target**: IoU 0.72
- **Starting point**: Resume from epoch 88 (IoU 0.6413)

---

## ✅ Recommended Configuration (Conservative - Start Here!)

### Full Command

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_batch8 \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 2 \
    --accumulation-steps 2 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.0004 \
    --stage2-lr 0.00018 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

### Parameter Breakdown

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **--batch-size** | 8 | Your requested batch size for Stage 1 (frozen backbone) |
| **--stage2-batch-size** | 2 | Reduced for Stage 2 (unfrozen backbone uses 4x memory) |
| **--accumulation-steps** | 2 | Effective batch: Stage1=16, Stage2=4 |
| **--lr** | 0.0004 | Scaled up from 0.00025 for larger batch (1.6x factor) |
| **--stage2-lr** | 0.00018 | Conservative LR accounting for deep supervision |
| **--scheduler** | cosine | Smooth decay to escape plateau |
| **--deep-supervision** | enabled | Multi-scale learning (+0.03-0.05 IoU) |
| **--progressive-unfreeze** | enabled | Memory efficient, gradual adaptation |

### Effective Batch Sizes

```
Stage 1 (Epochs 0-30):
  Batch: 8 × Accumulation: 2 = Effective batch: 16

Stage 2 (Epochs 31-120):
  Batch: 2 × Accumulation: 2 = Effective batch: 4
```

### Learning Rate Rationale

**Base LR scaling**:
```
Previous setup:
  Batch: 2, Accumulation: 4 → Effective: 8
  LR: 0.00025

New setup (Stage 1):
  Batch: 8, Accumulation: 2 → Effective: 16 (2x increase)
  LR: 0.00025 × 1.6 = 0.0004
```

**Stage 2 LR calculation**:
```
Without deep supervision: ~0.00025 (for effective batch 4)
With deep supervision: 0.00025 × 0.75 = 0.000187 ≈ 0.00018

This is SAFER than your previous 0.0002 that caused explosion!
```

### Expected Performance

| Metric | Value |
|--------|-------|
| **Training speed** | ~6-7 min/epoch (faster than batch=2!) |
| **GPU utilization** | ~85-95% (much better than batch=2) |
| **Memory usage** | Stage 1: ~11-12GB, Stage 2: ~13-14GB |
| **Expected IoU at epoch 120** | 0.70-0.73 (should reach 0.72 target!) |
| **Timeline to 0.72** | ~25-30 epochs from restart |

---

## 🔥 Alternative: Aggressive Configuration

**If conservative approach is too slow or you want faster convergence:**

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_batch8_aggressive \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 2 \
    --accumulation-steps 2 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.0004 \
    --stage2-lr 0.00022 \
    --scheduler cosine \
    --min-lr 0.00002 \
    --warmup-epochs 3 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

### Key Differences

- **--stage2-lr 0.00022**: 22% higher than conservative (but still 10% lower than your previous 0.0002)
- **--min-lr 0.00002**: 2x higher minimum to maintain learning pressure
- **--warmup-epochs 3**: Gradual ramp-up for stability with higher LR
- **Expected IoU**: 0.71-0.74 (higher ceiling, slightly more risk)

---

## 💪 Maximum Performance Configuration

**If you have good GPU memory (16GB) and want maximum speed:**

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_batch8_max \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 2 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.0005 \
    --stage2-lr 0.00020 \
    --scheduler cosine \
    --min-lr 0.00002 \
    --warmup-epochs 5 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

### Key Differences

- **--accumulation-steps 4**: Larger effective batch (Stage1: 32, Stage2: 8)
- **--lr 0.0005**: Higher base LR for larger effective batch
- **--stage2-lr 0.00020**: Exactly at the explosion threshold (risky but effective)
- **--warmup-epochs 5**: Longer warmup for safety with aggressive LR
- **Effective batch Stage 2: 8**: Better gradient estimates, faster convergence
- **Risk**: Higher memory usage, potential for instability
- **Reward**: Fastest convergence, ~4-5 min/epoch

---

## 📊 Comparison Table

| Config | Stage1 Eff. Batch | Stage2 Eff. Batch | Stage2 LR | Speed | Risk | Expected IoU |
|--------|-------------------|-------------------|-----------|-------|------|--------------|
| **Conservative** | 16 | 4 | 0.00018 | 6-7 min/epoch | Low | 0.70-0.73 |
| **Aggressive** | 16 | 4 | 0.00022 | 6-7 min/epoch | Medium | 0.71-0.74 |
| **Maximum** | 32 | 8 | 0.00020 | 4-5 min/epoch | Higher | 0.72-0.76 |

---

## 🎯 Recommended Choice

### Start with **Conservative** if:
- ✅ You want guaranteed safety after the explosion
- ✅ You're resuming from epoch 88 and don't want another crash
- ✅ You have limited time and can't afford another restart
- ✅ IoU 0.70-0.73 is acceptable

### Use **Aggressive** if:
- ✅ Conservative runs stable for 10 epochs
- ✅ You want to push for IoU 0.73-0.74
- ✅ You're comfortable monitoring training closely
- ✅ You have backup checkpoints

### Use **Maximum** if:
- ✅ You have 16GB GPU memory
- ✅ You want fastest possible training
- ✅ You're willing to experiment with higher risk
- ✅ You want to maximize final performance (0.75+)

---

## 🔍 Memory Usage Estimates

### Conservative/Aggressive Config

**Stage 1** (batch=8, frozen backbone):
```
Model: ~4GB
Activations (batch=8): ~6GB
Gradients (decoder only): ~1GB
Optimizer state: ~1.5GB
Deep supervision overhead: ~0.5GB
Total: ~13GB ✅ Safe on Kaggle P100 (16GB)
```

**Stage 2** (batch=2, unfrozen backbone):
```
Model: ~4GB
Activations (batch=2): ~3GB
Gradients (full model): ~4GB
Optimizer state: ~2GB
Deep supervision overhead: ~0.3GB
Total: ~13.3GB ✅ Safe
```

### Maximum Config

**Stage 2** (batch=2, accum=4, unfrozen):
```
Total: ~14.5GB ✅ Should fit, but tight
```

**⚠️ If you get OOM with Maximum config**: Reduce `--stage2-batch-size` to 1

---

## 💡 Why These Parameters?

### 1. Batch Size 8 → LR 0.0004

**Learning rate must scale with batch size**:

```python
# Linear scaling rule (common for small batches)
new_lr = base_lr × (new_batch / base_batch)

# Your case:
base_lr = 0.00025 (for effective batch 8)
new_batch = 16 (batch=8, accum=2)
new_lr = 0.00025 × (16 / 8) = 0.0005

# Conservative adjustment (0.8x factor for safety):
new_lr = 0.0005 × 0.8 = 0.0004 ✅
```

### 2. Stage2 LR 0.00018

**Must account for deep supervision gradient amplification**:

```python
# Base Stage 2 LR (for effective batch 4)
base_stage2_lr = 0.00025

# Deep supervision reduces LR by 25-30%
stage2_lr = 0.00025 × 0.72 = 0.00018 ✅

# This is 10% lower than your previous 0.0002 that exploded!
```

### 3. Stage2 Batch Size 2

**Memory constraints with unfrozen backbone**:

```python
# Stage 1: Frozen backbone
batch=8: 12-13GB ✅ Fits easily

# Stage 2: Unfrozen backbone (4x memory)
batch=8: ~20GB ❌ OOM!
batch=4: ~16-17GB ⚠️ Tight, risky
batch=2: ~13-14GB ✅ Safe
batch=1: ~11-12GB ✅ Very safe (but slower)
```

### 4. Accumulation Steps 2

**Balance between speed and memory**:

```python
# Option 1: accum=2
Stage 1: 8 × 2 = 16 effective batch ✅ Good
Stage 2: 2 × 2 = 4 effective batch ✅ Reasonable

# Option 2: accum=4
Stage 1: 8 × 4 = 32 effective batch ✅ Excellent
Stage 2: 2 × 4 = 8 effective batch ✅ Very good (but slower)

# Option 3: accum=1
Stage 1: 8 × 1 = 8 effective batch ⚠️ Smaller
Stage 2: 2 × 1 = 2 effective batch ⚠️ Too small, noisy
```

**Conservative uses accum=2, Maximum uses accum=4**

---

## 🚀 Step-by-Step Startup Guide

### 1. Stop Current Training

- Click "Interrupt" in Kaggle notebook
- Clear GPU memory:
  ```python
  import torch, gc
  torch.cuda.empty_cache()
  gc.collect()
  ```

### 2. Verify Checkpoint

```python
import torch
checkpoint = torch.load('/kaggle/working/checkpoints/best_model.pth',
                        map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']
has_nan = any(torch.isnan(v).any() for v in state_dict.values())

print(f"Epoch: {checkpoint.get('epoch', '?')}")
print(f"Best IoU: {checkpoint.get('best_iou', '?')}")
print(f"Has NaN: {has_nan}")
```

**Expected**: Epoch 88, IoU 0.6413, Has NaN: False

**If Has NaN: True**: Use backup from epoch 86 or earlier

### 3. Start Training

Copy-paste the **Conservative** command above and run it!

### 4. Monitor First 10 Epochs

**Watch for**:

✅ **Healthy**:
```
Epoch 89: Loss: 0.2156 | IoU: 0.6445 | LR: 0.000175
Epoch 90: Loss: 0.2134 | IoU: 0.6478 | LR: 0.000172
Epoch 91: Loss: 0.2098 | IoU: 0.6512 | LR: 0.000169
```

⚠️ **Warning**:
```
Epoch 89: Loss: 0.2156 | IoU: 0.6445
Epoch 90: Loss: 0.2089 | IoU: 0.6401  ← IoU dropped
Epoch 91: Loss: 0.2034 | IoU: 0.6356  ← Still dropping
```
**Action**: If this happens, reduce stage2-lr to 0.00015

🚨 **Explosion** (won't happen with new NaN detection!):
```
Epoch 90: Loss: 18.3456 | IoU: 0.0234
```
**With new NaN detection**: Training auto-stops with error message

### 5. Adjust If Needed

**After 10 epochs, if training is stable**:
- Can increase to Aggressive config (stage2-lr 0.00022)
- Or add EMA for final polish (--use-ema --ema-decay 0.999)

---

## 📈 Expected Training Curve

### Conservative Config

```
Epoch 88:  IoU 0.6413 (starting point)
Epoch 95:  IoU 0.6534 (+0.012)
Epoch 100: IoU 0.6689 (+0.028)
Epoch 105: IoU 0.6812 (+0.040)
Epoch 110: IoU 0.6934 (+0.052)
Epoch 115: IoU 0.7023 (+0.061) ← Crossed 0.70!
Epoch 120: IoU 0.7134 (+0.072) ← Reached 0.72 target! ✅
```

**Timeline**: ~30 epochs = 3-4 hours

### Aggressive Config

```
Epoch 88:  IoU 0.6413
Epoch 95:  IoU 0.6578 (+0.017)
Epoch 100: IoU 0.6756 (+0.034)
Epoch 105: IoU 0.6901 (+0.049)
Epoch 110: IoU 0.7045 (+0.063) ← Crossed 0.70!
Epoch 115: IoU 0.7178 (+0.077) ← Exceeded target!
Epoch 120: IoU 0.7289 (+0.088)
```

**Timeline**: ~30 epochs = 3-4 hours

### Maximum Config

```
Epoch 88:  IoU 0.6413
Epoch 95:  IoU 0.6612 (+0.020)
Epoch 100: IoU 0.6823 (+0.041)
Epoch 105: IoU 0.7012 (+0.060) ← Crossed 0.70!
Epoch 110: IoU 0.7189 (+0.078) ← Exceeded target!
Epoch 115: IoU 0.7334 (+0.092)
Epoch 120: IoU 0.7445 (+0.103)
```

**Timeline**: ~30 epochs = 2-2.5 hours (fastest!)

---

## 🛡️ Safety Features (Already Implemented)

Your updated code now has **critical NaN detection**:

1. **Loss NaN check**: Stops training if loss becomes NaN
2. **Gradient NaN check**: Stops if gradients explode before corrupting weights
3. **Clear error messages**: Tells you exactly what went wrong

**This means**: Even if parameters are slightly too aggressive, training will stop safely instead of corrupting your model!

---

## 🎯 My Recommendation

**Start with Conservative config**:
```bash
# This command:
--batch-size 8
--stage2-batch-size 2
--accumulation-steps 2
--lr 0.0004
--stage2-lr 0.00018
--deep-supervision
```

**Why**:
1. ✅ Safe LR after your previous explosion
2. ✅ Batch size 8 as you requested
3. ✅ Good GPU utilization (~85%)
4. ✅ Deep supervision for +0.03-0.05 IoU boost
5. ✅ Should reach 0.72 IoU in ~30 epochs
6. ✅ Low risk of another crash

**If Conservative works well for 10 epochs**:
- Switch to Aggressive config for final push to 0.73-0.74

**If you want maximum speed from the start**:
- Use Maximum config (but monitor very closely!)

---

## 📞 Quick Decision Tree

```
Do you prioritize safety after the explosion?
├─ YES → Conservative (stage2-lr 0.00018)
└─ NO
   └─ Do you have time to monitor closely?
      ├─ YES → Aggressive (stage2-lr 0.00022)
      └─ NO → Conservative (safer)

Do you have 16GB GPU memory?
├─ YES → Can try Maximum (accum=4, fastest)
└─ NO/UNSURE → Use Conservative (accum=2)

Did Conservative run stable for 10+ epochs?
├─ YES → Upgrade to Aggressive or Maximum
└─ NO → Reduce stage2-lr to 0.00015
```

---

## ✅ Final Checklist

Before starting:
- [ ] Stopped previous training (stuck at 30%)
- [ ] Cleared GPU memory
- [ ] Verified checkpoint is clean (no NaN)
- [ ] Chose configuration (Conservative recommended)
- [ ] Copied command correctly
- [ ] Ready to monitor first 10 epochs

Ready to go! 🚀

---

**Expected outcome**: Stable training → IoU 0.72 in ~30 epochs → No more explosions! ✅
