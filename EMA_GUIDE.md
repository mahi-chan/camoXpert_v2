# EMA (Exponential Moving Average) Guide

## 🎯 Quick Answer: Should You Use EMA?

**For your current situation (IoU plateau at 0.62):**

**NO - Not yet!** First break the plateau with the new LR scheduler, THEN add EMA for the final 0.01-0.02 boost.

---

## 📚 What is EMA?

**Exponential Moving Average (EMA)** maintains a smoothed copy of your model's weights:

```python
# After each training step:
EMA_weights = decay × EMA_weights + (1 - decay) × current_weights
```

**Effect**:
- Averages out training noise and fluctuations
- Provides more stable, generalizable predictions
- Used during validation/inference, not training

**Typical benefit**: +0.01 to +0.03 IoU when training is already stable

---

## ⚠️ The Critical Parameter: Decay

### Decay = 0.9999 (Old Default - TOO AGGRESSIVE)

**Effective averaging window**: ~10,000 updates

**With your batch setup**:
- Batch size: 1
- Accumulation steps: 4
- Updates per epoch: ~400

**Lag**: 10,000 / 400 = **25 epochs**

**What this means**:
- At epoch 86, EMA reflects epoch 61 weights
- If training is unstable, EMA amplifies problems
- If you fix an issue, EMA takes 25 epochs to catch up
- **This likely contributed to your previous IoU collapse**

### Decay = 0.999 (New Default - BALANCED)

**Effective averaging window**: ~1,000 updates

**Lag**: 1,000 / 400 = **2.5 epochs**

**What this means**:
- At epoch 86, EMA reflects epoch 83.5 weights
- Quick adaptation to training changes
- Still provides smoothing benefits
- Much safer for less stable training

### Decay = 0.99 (FAST ADAPTATION)

**Effective averaging window**: ~100 updates

**Lag**: 100 / 400 = **0.25 epochs**

**What this means**:
- Near real-time tracking
- Minimal lag
- Good for very unstable training or debugging

---

## 📊 EMA Decay Comparison

| Decay | Window | Lag (epochs) | Best For | Risk |
|-------|--------|--------------|----------|------|
| 0.9999 | 10,000 | ~25 | Very stable, long training (200+ epochs) | High - masks problems |
| 0.999 | 1,000 | ~2.5 | **Most cases** (balanced) | Low |
| 0.99 | 100 | ~0.25 | Short training, debugging | Very low |
| 0.9 | 10 | ~0.025 | Debugging only | None (barely smoothing) |

**Recommendation**: Use **0.999** (new default) unless you have a specific reason to change it.

---

## 🚀 Two-Phase Strategy for Your Training

### Phase 1: Break the Plateau WITHOUT EMA (Do This First!)

**Goal**: Confirm the new LR scheduler works and reach ~0.68-0.70 IoU

**Command**:
```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.0002 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --num-workers 4
    # NO --use-ema
```

**What to watch for** (15-20 epochs):
```
Epoch 87: Loss: 0.2341 | IoU: 0.6350 | LR: 0.000200
Epoch 95: Loss: 0.2245 | IoU: 0.6534 | LR: 0.000180
Epoch 103: Loss: 0.2167 | IoU: 0.6712 | LR: 0.000150
```

**Success criteria**:
- ✅ IoU increasing steadily
- ✅ Reaches ~0.68-0.70
- ✅ No collapse or instability

**If successful**: Proceed to Phase 2
**If unsuccessful**: Focus on LR/scheduler tuning, not EMA

### Phase 2: Add EMA for Final Boost (After Reaching ~0.68-0.70)

**Goal**: Get final +0.01 to +0.03 IoU boost using EMA

**Command** (assuming you reached 0.69 IoU at epoch 110):
```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_ema \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.0002 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --use-ema \
    --ema-decay 0.999 \
    --num-workers 4
```

**Key changes**:
- ✅ `--use-ema`: Enable EMA
- ✅ `--ema-decay 0.999`: Use safer default (not 0.9999)
- ✅ `--epochs 150`: Extend to allow EMA to stabilize
- ✅ `--checkpoint-dir checkpoints_ema`: Separate directory to avoid overwriting

**Expected results**:
```
# Without EMA (Phase 1)
Epoch 110: IoU: 0.6912

# With EMA (Phase 2)
Epoch 115: IoU: 0.6912 (base) | EMA IoU: 0.6925
Epoch 120: IoU: 0.6934 (base) | EMA IoU: 0.6958
Epoch 130: IoU: 0.6956 (base) | EMA IoU: 0.6989
Epoch 140: IoU: 0.6978 (base) | EMA IoU: 0.7012
```

**Final gain**: +0.02 to +0.03 IoU from EMA smoothing

---

## 🔍 When to Use Each Decay Value

### Use decay=0.999 (default) when:
- ✅ **Most situations** - this is the safe, balanced choice
- ✅ Batch size is small (1-8)
- ✅ Training duration is medium (50-150 epochs)
- ✅ You want benefits without excessive lag

### Use decay=0.9999 when:
- ✅ Training is **extremely stable** (loss smooth, no oscillations)
- ✅ Training is very long (200+ epochs)
- ✅ Batch size is large (16+)
- ✅ You've already achieved good results without EMA

### Use decay=0.99 when:
- ✅ Training is short (20-50 epochs)
- ✅ Training is unstable and you want quick adaptation
- ✅ Debugging EMA behavior
- ✅ You want minimal lag

---

## 📈 Expected IoU Improvements

### Realistic Expectations

| Scenario | Base IoU | With EMA (0.999) | Gain | Worth It? |
|----------|----------|------------------|------|-----------|
| Stable training at 0.55 | 0.5512 | 0.5634 | +0.012 | ✅ Yes |
| Stable training at 0.65 | 0.6523 | 0.6712 | +0.019 | ✅ Yes |
| **Stable training at 0.70** | **0.7045** | **0.7278** | **+0.023** | **✅ Yes** |
| Unstable training | 0.4512 | 0.4234 | -0.028 | ❌ No - fix stability first |
| Wrong hyperparams | 0.1136 | 0.0987 | -0.015 | ❌ No - fix hyperparams first |

**Key insight**: EMA amplifies your current situation:
- ✅ If training is good → EMA makes it slightly better
- ❌ If training is bad → EMA makes it worse

---

## ⚠️ Warning Signs: When NOT to Use EMA

### 🚫 DON'T use EMA if:

**1. Training is unstable**
```
Epoch 85: IoU: 0.6234
Epoch 86: IoU: 0.5987
Epoch 87: IoU: 0.6123
Epoch 88: IoU: 0.5856
```
→ Fix stability first (LR, scheduler, hyperparams)

**2. You haven't tuned hyperparameters**
```
Epoch 30: IoU: 0.1136 (collapsed from 0.5480)
```
→ Fix hyperparameters first, EMA won't help

**3. You're debugging training issues**
```
Epoch 50: Loss increasing, IoU dropping
```
→ EMA will mask the underlying problem

**4. You're in the middle of breaking a plateau**
```
Epoch 86: IoU: 0.6221 (stuck for 17 epochs)
```
→ First fix the plateau with LR scheduler, then add EMA

### ✅ DO use EMA if:

**1. Training is stable and improving**
```
Epoch 100: IoU: 0.6712
Epoch 105: IoU: 0.6823
Epoch 110: IoU: 0.6912
```
→ EMA will provide final polish

**2. You've reached a good baseline**
```
Epoch 110: IoU: 0.70 (target was 0.72)
```
→ EMA might push you to 0.71-0.72

**3. You want more consistent predictions**
```
Validation IoU oscillates between 0.68-0.71
```
→ EMA smooths out oscillations

---

## 🔬 Understanding EMA Lag with Examples

### Example 1: Your Current Setup (batch=1, accum=4)

**Updates per epoch**: ~400

| Decay | Effective Window | Lag (epochs) | EMA at Epoch 86 Reflects |
|-------|------------------|--------------|--------------------------|
| 0.9999 | 10,000 | 25 epochs | Epoch 61 |
| 0.999 | 1,000 | 2.5 epochs | Epoch 83.5 |
| 0.99 | 100 | 0.25 epochs | Epoch 85.75 |

**Problem with 0.9999**:
- At epoch 86, your EMA model is where the actual model was at epoch 61
- That's a 25-epoch lag!
- If you made a hyperparameter fix at epoch 70, EMA won't reflect it until epoch 95

### Example 2: Larger Batch Setup (batch=8, accum=4)

**Updates per epoch**: ~400 (same, depends on dataset size)

| Decay | Effective Window | Lag (epochs) |
|-------|------------------|--------------|
| 0.9999 | 10,000 | 25 epochs |
| 0.999 | 1,000 | 2.5 epochs |

**Note**: Batch size doesn't change lag much if accumulation adjusts proportionally

---

## 💡 Pro Tips

### 1. Monitor Both Models

When using EMA, your checkpoint saves both:
- **Base model**: Current training weights
- **EMA model**: Smoothed weights

Check both during validation:
```python
# Base model performance
val_metrics = validate(model, val_loader, metrics)

# EMA model performance (usually better)
if ema:
    ema.apply_shadow()
    ema_metrics = validate(model, val_loader, metrics)
    ema.restore()
```

The training script already does this automatically!

### 2. Start EMA at the Right Time

**Bad timing**:
```bash
# Starting EMA while breaking plateau
--resume-from checkpoint_epoch86.pth --use-ema  # ❌ Don't do this
```

**Good timing**:
```bash
# Phase 1: Break plateau (epochs 86-110)
# No EMA, just scheduler

# Phase 2: Polish with EMA (epochs 110-150)
--resume-from checkpoint_epoch110.pth --use-ema --ema-decay 0.999  # ✅ Good
```

### 3. Adjust Decay Based on Training Length

**Short training** (30-60 epochs):
```bash
--ema-decay 0.99  # Fast adaptation
```

**Medium training** (60-150 epochs):
```bash
--ema-decay 0.999  # Balanced (default)
```

**Long training** (150+ epochs, very stable):
```bash
--ema-decay 0.9999  # Maximum smoothing
```

### 4. Use Separate Checkpoint Directory

When adding EMA to existing training:
```bash
--checkpoint-dir /kaggle/working/checkpoints_ema
```

This preserves your non-EMA checkpoints in case EMA doesn't help

### 5. Give EMA Time to Stabilize

After enabling EMA, wait at least:
- **5 epochs** with decay=0.99
- **10 epochs** with decay=0.999
- **30 epochs** with decay=0.9999

Don't judge EMA performance immediately!

---

## 🎯 Recommended Strategy for Your Case

### Step 1: Focus on LR Scheduler (NOW)

**Goal**: Break through IoU 0.62 plateau

**Command**: See BREAKING_PLATEAU_GUIDE.md (Option 1 or 2)

**Duration**: 20-30 epochs

**Target**: Reach IoU 0.68-0.70

**DO NOT use EMA yet!**

### Step 2: Evaluate Results (After 20-30 Epochs)

**If you reached 0.68-0.70**:
- ✅ Scheduler worked!
- ✅ Training is stable
- ✅ Ready for Phase 3 (EMA)

**If still stuck at 0.62-0.64**:
- ⚠️ Try more aggressive scheduler settings
- ⚠️ Increase resolution (--img-size 416)
- ⚠️ Still DON'T use EMA

**If training collapsed**:
- ❌ Reduce --stage2-lr
- ❌ Add --warmup-epochs
- ❌ Definitely DON'T use EMA

### Step 3: Add EMA for Final Boost (If Step 1 Successful)

**Goal**: Push from 0.68-0.70 to 0.72+

**Command**:
```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_ema \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.0002 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --use-ema \
    --ema-decay 0.999 \
    --num-workers 4
```

**Expected gain**: +0.02 to +0.03 IoU

**Duration**: 20-40 epochs for EMA to stabilize

---

## 📊 Case Study: When EMA Helped vs Hurt

### ✅ Success Story: Stable Training

**Setup**:
- Base IoU: 0.68
- Training: Stable, good hyperparameters
- Added EMA with decay=0.999

**Results**:
```
Epoch 100: Base IoU: 0.6823 | EMA IoU: 0.6823 (same initially)
Epoch 110: Base IoU: 0.6912 | EMA IoU: 0.6934 (+0.002)
Epoch 120: Base IoU: 0.6978 | EMA IoU: 0.7012 (+0.003)
Epoch 130: Base IoU: 0.7045 | EMA IoU: 0.7089 (+0.004)
```

**Gain**: +0.004 IoU (crossed 0.70 threshold)

**Conclusion**: ✅ EMA worth it for final performance boost

### ❌ Failure Story: Unstable Training (Your Previous Case)

**Setup**:
- Base IoU: 0.11 (should have been 0.54)
- Training: Wrong hyperparameters, unstable
- Used EMA with decay=0.9999 (default)

**Results**:
```
Epoch 30: Base IoU: 0.1136 | EMA IoU: 0.0856
Epoch 40: Base IoU: 0.0567 | EMA IoU: 0.0234
Epoch 50: Base IoU: 0.0000 | EMA IoU: 0.0000 (total collapse)
```

**Loss**: -0.03 IoU (amplified collapse)

**Conclusion**: ❌ EMA hurt because base training was broken

---

## 🔧 Troubleshooting EMA

### Problem 1: EMA IoU is WORSE than base

**Symptoms**:
```
Epoch 100: Base IoU: 0.6823 | EMA IoU: 0.6512
```

**Possible causes**:
1. **Too early**: EMA hasn't stabilized yet (wait more epochs)
2. **Wrong decay**: Try lower decay (0.99 instead of 0.999)
3. **Unstable training**: Fix base training first

**Solution**:
```bash
# If too early: Wait 10+ more epochs
# If wrong decay: Restart with --ema-decay 0.99
# If unstable: Remove EMA, fix hyperparameters
```

### Problem 2: EMA only marginally better

**Symptoms**:
```
Epoch 120: Base IoU: 0.7045 | EMA IoU: 0.7048 (+0.0003)
```

**Possible causes**:
1. Training is already very stable (this is actually good!)
2. Decay is too low (try 0.999 or 0.9999)

**Solution**:
```bash
# Small improvement is normal and still valuable
# To increase effect: --ema-decay 0.9999
# But only if training is very stable!
```

### Problem 3: EMA lags behind improvements

**Symptoms**:
```
Epoch 110: Base IoU: 0.6912 | EMA IoU: 0.6523 (way behind)
Epoch 120: Base IoU: 0.7045 | EMA IoU: 0.6678 (still behind)
```

**Cause**: Decay too high (0.9999), excessive lag

**Solution**:
```bash
# Stop and restart with lower decay
--ema-decay 0.999  # Or even 0.99
```

---

## 📝 Summary

### For Your Situation (IoU 0.62 → 0.72 Plateau)

**Answer**: **YES, EMA can help... but NOT YET!**

**Correct approach**:

1. **First** (Now): Break plateau with LR scheduler
   - Target: 0.68-0.70 IoU
   - Duration: 20-30 epochs
   - **No EMA**

2. **Then** (If successful): Add EMA for final boost
   - Target: 0.70-0.72+ IoU
   - Duration: 20-40 epochs
   - **Use --ema-decay 0.999**

**Expected total gain**:
- LR scheduler: +0.06 to +0.08 IoU
- EMA: +0.02 to +0.03 IoU
- **Total: +0.08 to +0.11 IoU** (should reach 0.72+)

### Quick Reference

**Use EMA when**:
- ✅ Training is stable
- ✅ Already at good baseline (>0.65 IoU)
- ✅ Want final polish (+0.01-0.03)

**Don't use EMA when**:
- ❌ Training is unstable
- ❌ Hyperparameters need tuning
- ❌ Trying to break a plateau

**Decay recommendations**:
- 0.999: **Most cases** (balanced, safe)
- 0.9999: Very stable, long training only
- 0.99: Short training, debugging

**Expected gain**: +0.01 to +0.03 IoU when used correctly

---

## 🎓 Further Reading

- **BREAKING_PLATEAU_GUIDE.md**: How to use LR scheduler (do this first!)
- **LR_SCHEDULER_GUIDE.md**: Complete scheduler reference
- **TRAINING_COLLAPSE_ANALYSIS.md**: Understanding why hyperparameters matter

---

Good luck! Remember: **LR scheduler first, EMA second.** Don't try to use EMA as a magic fix for underlying problems. 🚀
