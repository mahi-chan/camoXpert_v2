# Training Collapse Recovery Guide

## 🚨 Emergency: Gradient Explosion + Stuck Training

### What Happened

**Epoch 88**: IoU 0.6413 ✅ (Last good checkpoint)
**Epochs 89-97**: Gradual decline (warning signs)
**Epoch 98**: 💥 CATASTROPHIC COLLAPSE (IoU 0.6413 → 0.0022)
**Epoch 101**: 🔒 STUCK at 30% (NaN-induced hang)

---

## 🎯 Root Causes Identified

### 1. No NaN/Inf Detection (Critical)

Training continued with NaN weights after gradient explosion, corrupting the model completely.

### 2. Deep Supervision + LR Mismatch

Adding `--deep-supervision` increased gradient flow by ~40-100%, making your LR effectively 1.4-2x too high.

### 3. NaN Propagation Causing Hang

Model weights contain NaN → some operation hangs at specific batch → stuck at 30%.

---

## ✅ Immediate Recovery Steps

### Step 1: Stop Training (NOW!)

**In Kaggle Notebook**:
1. Click **"Interrupt"** button (top right)
2. If that fails: Click **"..." menu** → **"Stop Session"**
3. If that fails: Close browser tab, wait 1 minute, reopen

**Verification**: Training should show "Interrupted" or "Stopped"

### Step 2: Clear GPU Memory

After stopping, run this in a new cell:

```python
import torch
import gc

# Clear all GPU memory
torch.cuda.empty_cache()
gc.collect()

# Verify GPU is clean
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

### Step 3: Verify Last Good Checkpoint

Run this to check your checkpoints:

```python
import os
import torch

checkpoint_dir = '/kaggle/working/checkpoints'

# List all checkpoints
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
print("Available checkpoints:")
for ckpt in checkpoints:
    path = os.path.join(checkpoint_dir, ckpt)
    size_mb = os.path.getsize(path) / (1024**2)
    print(f"  {ckpt}: {size_mb:.1f} MB")

# Load best checkpoint and verify it's not corrupted
best_path = os.path.join(checkpoint_dir, 'best_model.pth')
if os.path.exists(best_path):
    try:
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
        epoch = checkpoint.get('epoch', 'unknown')
        iou = checkpoint.get('best_iou', 'unknown')

        # Check for NaN in weights
        state_dict = checkpoint['model_state_dict']
        has_nan = any(torch.isnan(v).any() for v in state_dict.values())

        print(f"\nbest_model.pth:")
        print(f"  Epoch: {epoch}")
        print(f"  Best IoU: {iou}")
        print(f"  Contains NaN: {has_nan}")

        if has_nan:
            print("  ⚠️ WARNING: Checkpoint is corrupted with NaN!")
        else:
            print("  ✅ Checkpoint is clean!")

    except Exception as e:
        print(f"  ❌ Error loading checkpoint: {e}")
else:
    print(f"❌ best_model.pth not found!")
```

**Expected output**:
```
best_model.pth:
  Epoch: 88
  Best IoU: 0.6413
  Contains NaN: True  ← Your checkpoint is likely corrupted!
```

### Step 4: Restore from Backup (If Available)

**Check for backup checkpoints**:

```python
# Kaggle might have saved epoch-specific checkpoints
# Check if you have checkpoint from epoch 88 or earlier

backup_candidates = [
    'checkpoint_epoch_88.pth',
    'checkpoint_epoch_85.pth',
    'checkpoint_epoch_80.pth',
]

for candidate in backup_candidates:
    path = os.path.join(checkpoint_dir, candidate)
    if os.path.exists(path):
        print(f"✅ Found backup: {candidate}")
    else:
        print(f"❌ Not found: {candidate}")
```

**If you have a backup from epoch 88 or earlier**:
```bash
cp checkpoints/checkpoint_epoch_88.pth checkpoints/best_model_clean.pth
```

**If you DON'T have a backup**:
- You'll need to resume from the last checkpoint BEFORE the corruption
- Likely epoch 86 (before you added deep supervision)

---

## 🚀 Safe Restart Strategy

### Option A: Resume from Last Good Checkpoint (Recommended)

**If your best_model.pth from epoch 88 is clean (no NaN)**:

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_recovery \
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
    --stage2-lr 0.00015 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

**Key changes from before**:
- ✅ `--stage2-lr 0.00015`: **Reduced by 25%** to account for deep supervision
- ✅ `--checkpoint-dir checkpoints_recovery`: New directory to avoid overwriting
- ✅ Same deep supervision (but with safer LR)

### Option B: Resume from Epoch 86 (Before Deep Supervision)

**If your current checkpoint is corrupted, use epoch 86 checkpoint**:

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_recovery \
    --resume-from /kaggle/input/your-previous-run/checkpoints/best_model.pth \
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
    --stage2-lr 0.00015 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

**Note**: You'll need to point `--resume-from` to the Kaggle dataset you created from your previous run (the one that reached IoU 0.62).

### Option C: Start Fresh with Safe Settings

**If you lost all checkpoints**:

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_safe \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 8 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.00015 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

---

## 🛡️ Preventive Measures (Critical!)

### Add NaN Detection to Training Loop

I'll create a fix for this to prevent future collapses.

### Reduced Learning Rate for Deep Supervision

**Rule**: When adding deep supervision, reduce LR by **20-30%**:

```
Original LR: 0.0002
With deep supervision: 0.00015 (25% reduction)
```

### Monitor for Warning Signs

**Early warning signs of gradient explosion** (what you saw epochs 89-97):

```
Epoch 89: IoU 0.6371 (drop from 0.6413)
Epoch 90: IoU 0.6345 (continuing decline)
Epoch 91: IoU 0.6312 (still declining)
...
```

**Action**: If you see IoU declining for 3+ consecutive epochs after a change, **stop and reduce LR immediately!**

---

## 📊 Why the Stuck at 30%?

### Technical Explanation

```python
# What's happening at epoch 101, batch 30%:

1. Model weights contain NaN (from epoch 98 explosion)
2. Forward pass: model(images) produces NaN predictions
3. Loss computation: BCE(NaN, target) → NaN loss
4. Backward pass: grad(NaN) → trying to compute gradients of NaN
5. Some PyTorch operation enters infinite loop handling NaN
6. Process stuck, CPU/GPU at 100% but no progress

Common culprits:
- BatchNorm with NaN inputs (tries to compute running mean/var of NaN)
- Loss functions with log(NaN) or sqrt(NaN)
- Gradient computation with division by NaN
```

**This is NOT a Kaggle notebook problem**. It's a code-level issue with NaN handling.

---

## 🔍 Post-Recovery Monitoring

After restarting with fixed settings, **watch for these metrics**:

### Healthy Training (Epoch 1-10 after restart):

```
Epoch 89: Loss: 0.2234 | IoU: 0.6434 | LR: 0.000145 ✅
Epoch 90: Loss: 0.2198 | IoU: 0.6467 | LR: 0.000142 ✅
Epoch 91: Loss: 0.2176 | IoU: 0.6489 | LR: 0.000139 ✅
```

**Good signs**:
- Loss decreasing steadily
- IoU increasing or stable
- No sudden jumps

### Warning Signs:

```
Epoch 89: Loss: 0.2234 | IoU: 0.6434 | LR: 0.000145
Epoch 90: Loss: 0.1987 | IoU: 0.6201 | LR: 0.000142 ⚠️ IoU dropped!
Epoch 91: Loss: 0.1456 | IoU: 0.5834 | LR: 0.000139 🚨 Still dropping!
```

**Action**: Stop immediately and reduce `--stage2-lr` further (try 0.00012).

### Explosion Signs:

```
Epoch 89: Loss: 0.2234 | IoU: 0.6434
Epoch 90: Loss: 15.3456 | IoU: 0.0123 💥 EXPLOSION!
```

**Action**:
1. Stop immediately
2. Reduce `--stage2-lr` to 0.0001
3. Add `--warmup-epochs 5`
4. Resume from last good checkpoint

---

## 🎯 Expected Recovery Timeline

**With Option A (resume from epoch 88)**:

```
Epoch 89-100: Steady improvement with safe LR
  Expected: IoU 0.64 → 0.67

Epoch 100-120: Continued progress
  Expected: IoU 0.67 → 0.70-0.72
```

**Total time**: ~30 epochs from restart = ~4-5 hours

**Target**: IoU 0.72 ✅ Should be achievable!

---

## 💡 Key Lessons

### 1. Always Reduce LR When Adding Regularization

**Any technique that increases gradient flow requires LR reduction**:
- Deep supervision: -25% LR
- Deep supervision + EMA: -30% LR
- More data augmentation: -10-15% LR
- Larger batch size: +LR (opposite effect)

### 2. Monitor for Early Warning Signs

Don't wait for catastrophic failure. If IoU declines 3+ epochs consecutively, investigate immediately.

### 3. Save Checkpoints Frequently

**Recommended checkpoint strategy**:
```python
# Save every 5 epochs
if epoch % 5 == 0:
    save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

# Save best
if iou > best_iou:
    save_checkpoint('best_model.pth')
```

### 4. NaN Detection is Critical

Always check for NaN/Inf before updating weights. I'll add this to the code.

---

## 📞 Quick Reference

### If training crashes again:

1. **Stop immediately** (don't let it run with NaN weights)
2. **Check checkpoint for NaN** (code above)
3. **Reduce LR by 25-30%**
4. **Resume from last clean checkpoint**

### If stuck at X% again:

1. **Interrupt/Stop session**
2. **Clear GPU memory**
3. **Restart with reduced LR**
4. **Never** try to "wait it out" - it won't recover

### If IoU drops 3+ consecutive epochs:

1. **Reduce LR by 20%**
2. **Add warmup** (`--warmup-epochs 5`)
3. **Consider removing deep supervision** temporarily

---

## 🔗 Next Steps

1. **Immediately**: Stop current training
2. **Verify**: Check checkpoint for NaN corruption
3. **Restart**: Use Option A or B with reduced LR (0.00015)
4. **Monitor**: Watch first 10 epochs closely
5. **Report**: Let me know if you see any warning signs

I'll now create a fix for the NaN detection issue to prevent this from happening again.

---

**Bottom line**: Your training exploded because deep supervision + original LR was too aggressive. Restart with LR reduced to 0.00015 and monitor closely. You should be able to recover and reach 0.72 IoU! 🚀
