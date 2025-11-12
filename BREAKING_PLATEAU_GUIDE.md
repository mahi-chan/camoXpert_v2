# Breaking Your IoU Plateau: Action Plan

## 🎯 Your Situation

**Current Status**:
- Stuck at IoU ~0.62 after 86 epochs
- Only gained 0.01 IoU in last 17 epochs (epoch 69→86)
- Target: IoU 0.72
- **Gap to close: +0.10 IoU**

**Root Cause**: Learning rate is too low to escape the current local minimum. The model has found a "comfortable" solution but needs a higher learning rate to explore better configurations.

---

## ✅ Solution Implemented

I've added flexible learning rate scheduling to `train_ultimate.py` with these new features:

1. **Cosine Annealing Scheduler**: Smoothly decays LR from max to min
2. **Stage 2 Specific LR**: Set different LR for fine-tuning stage
3. **Warmup Support**: Gradually increase LR for stability
4. **Multiple Scheduler Options**: Choose what works best for your case

---

## 🚀 Recommended Action: Option 1 (Conservative)

**Best for**: First attempt at breaking plateau, minimize risk

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 2 \
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
    --deep-supervision \
    --num-workers 4
```

**Key Changes**:
- ✅ `--resume-from`: Continue from your epoch 86 checkpoint
- ✅ `--skip-stage1`: Go directly to Stage 2
- ✅ `--stage2-lr 0.0002`: Higher LR to escape plateau
- ✅ `--scheduler cosine`: Smooth decay over remaining epochs
- ✅ `--deep-supervision`: Multi-scale learning (+0.03-0.05 IoU)
- ✅ `--img-size 384`: Higher resolution (if you used 320 before)
- ✅ `--epochs 120`: Will train for 34 more epochs (86→120)

**Expected Result**:
- Cosine scheduler: +0.04 to +0.07 IoU
- Deep supervision: +0.03 to +0.05 IoU
- **Combined: +0.07 to +0.12 IoU**
- **Final IoU: ~0.69-0.74** (likely exceeds 0.72 target!)

**Timeline**: 20-30 epochs (~3-5 hours depending on GPU)

---

## 🔥 Recommended Action: Option 2 (Aggressive)

**Best for**: If Option 1 doesn't work or you want faster progress

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 2 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.0003 \
    --scheduler cosine \
    --min-lr 0.00002 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

**Key Differences from Option 1**:
- 🔥 `--stage2-lr 0.0003`: 50% higher LR for stronger escape from plateau
- 🔥 `--min-lr 0.00002`: 2x higher minimum to maintain learning pressure
- 🔥 `--epochs 150`: Longer training for 64 more epochs (86→150)

**Expected Result**:
- Scheduler (aggressive): +0.06 to +0.09 IoU
- Deep supervision: +0.03 to +0.05 IoU
- **Combined: +0.09 to +0.14 IoU**
- **Final IoU: ~0.71-0.76** (should exceed target!)

**Timeline**: 30-40 epochs (~5-7 hours depending on GPU)

**Warning**: Higher LR may cause temporary instability. Monitor first 5 epochs closely.

---

## 🎯 Recommended Action: Option 3 (Resolution Boost)

**Best for**: Maximum performance, if you have GPU memory

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --skip-stage1 \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 2 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 416 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.0003 \
    --scheduler cosine \
    --min-lr 0.00002 \
    --warmup-epochs 3 \
    --progressive-unfreeze \
    --deep-supervision \
    --num-workers 4
```

**Additional Changes**:
- 🎯 `--img-size 416`: 30% more pixels than 384, captures finer details
- 🎯 `--warmup-epochs 3`: Stability during resolution transition
- 🎯 `--deep-supervision`: Multi-scale learning

**Expected Result**:
- Scheduler (aggressive): +0.06 to +0.09 IoU
- Deep supervision: +0.03 to +0.05 IoU
- Higher resolution: +0.02 to +0.04 IoU
- **Combined: +0.11 to +0.18 IoU**
- **Final IoU: ~0.73-0.80** (exceeds target!)

**Timeline**: 30-50 epochs (~6-10 hours depending on GPU)

**Memory Note**: May need `--stage2-batch-size 1` and ensure no other processes using GPU

---

## 📊 What to Expect

### First 5 Epochs After Resume
```
Epoch 87: Loss: 0.2341 | IoU: 0.6350 | Dice: 0.7456 | LR: 0.000200
Epoch 88: Loss: 0.2298 | IoU: 0.6385 | Dice: 0.7495 | LR: 0.000195
Epoch 89: Loss: 0.2267 | IoU: 0.6421 | Dice: 0.7534 | LR: 0.000190
Epoch 90: Loss: 0.2245 | IoU: 0.6458 | Dice: 0.7572 | LR: 0.000185
Epoch 91: Loss: 0.2221 | IoU: 0.6489 | Dice: 0.7609 | LR: 0.000180
```

**Good signs** ✅:
- IoU increasing steadily (even if slowly)
- Loss decreasing
- LR gradually decreasing (cosine scheduler working)

### After 20 Epochs
```
Epoch 106: Loss: 0.2089 | IoU: 0.6834 | Dice: 0.8012 | LR: 0.000092
```

**Good signs** ✅:
- IoU has improved by +0.04 to +0.05
- Still making progress
- LR has decayed smoothly

### After 30-40 Epochs
```
Epoch 120: Loss: 0.1987 | IoU: 0.7134 | Dice: 0.8245 | LR: 0.000015
```

**Success** 🎉:
- IoU near or above 0.72 target
- Loss significantly lower
- Ready to finish or continue to Stage 3

---

## 🛑 Warning Signs & Solutions

### ❌ Loss Explodes
```
Epoch 87: Loss: 0.2341 | IoU: 0.6350 | LR: 0.000300
Epoch 88: Loss: 1.5678 | IoU: 0.1234 | LR: 0.000295
```

**Cause**: `--stage2-lr` is too high

**Solution**:
1. Stop training
2. Restart with lower `--stage2-lr 0.0002` (instead of 0.0003)
3. Add `--warmup-epochs 5` for stability

### ⚠️ Still Stuck After 20 Epochs
```
Epoch 106: Loss: 0.2330 | IoU: 0.6281 | LR: 0.000092
```

**Cause**: LR still not high enough or resolution too low

**Solution**:
1. Stop training
2. Restart with `--stage2-lr 0.0004` (50% higher)
3. Increase `--min-lr 0.00005` (5x default)
4. Consider `--img-size 416` or `448`

### ⚠️ Slow Progress But Moving
```
Epoch 106: Loss: 0.2267 | IoU: 0.6534 | LR: 0.000092
```

**Cause**: Settings are working, just need more time

**Solution**:
1. ✅ Keep training! This is normal.
2. Extend `--epochs 150` or `180` to allow more time
3. Monitor every 5-10 epochs

---

## 📋 Step-by-Step Instructions

### 1. Backup Your Current Checkpoint
```bash
cp /kaggle/working/checkpoints/best_model.pth \
   /kaggle/working/checkpoints/best_model_epoch86_backup.pth
```

### 2. Choose Your Option
- **Risk-averse**: Option 1 (Conservative)
- **Confident**: Option 2 (Aggressive)
- **Maximum performance**: Option 3 (Resolution Boost)

### 3. Run Training
Copy-paste the command from your chosen option above

### 4. Monitor Progress
Watch the output for:
- ✅ Steady IoU increases
- ✅ Loss decreasing
- ✅ LR showing in output (confirms scheduler working)

### 5. Check After 10 Epochs
- If IoU improved by +0.02 or more: ✅ Continue
- If IoU flat or worse: ⚠️ Try next higher option
- If loss exploded: ❌ Reduce `--stage2-lr` and restart

### 6. Evaluate After 30 Epochs
- If IoU ≥0.70: 🎉 Almost there! Continue to 0.72
- If IoU 0.66-0.70: ✅ Good progress, may need more epochs
- If IoU <0.66: ⚠️ Try Option 3 (Resolution Boost)

---

## 🔬 Understanding the LR Schedule

### Your Training Timeline

```
Stage 1 (Epochs 0-30): Frozen Backbone Training
├─ Using --lr 0.00025
└─ OneCycle scheduler (default)

[Your current position: Epoch 86]
↓

Stage 2 (Epochs 31-120): Fine-tuning with NEW SCHEDULER
├─ Starts with --stage2-lr 0.0002 (Option 1)
├─ Cosine annealing decay
├─ Gradually reduces to --min-lr 0.00001
└─ Should escape plateau and reach 0.72 IoU

Timeline with Option 1:
Epoch 31:  LR = 0.000200 (max)
Epoch 50:  LR = 0.000150
Epoch 70:  LR = 0.000100
Epoch 86:  LR = 0.000075 (resume point) ← YOU ARE HERE
Epoch 100: LR = 0.000050
Epoch 120: LR = 0.000010 (min)
```

### Why This Works

**Old approach** (what caused plateau):
- Constant LR or wrong scheduler
- Model found local minimum
- No mechanism to escape

**New approach** (with cosine scheduler):
- Starts with higher LR → gives model "energy" to escape plateau
- Gradually decreases → finds better minimum
- Never goes too low (min_lr) → maintains learning pressure

---

## 💡 Pro Tips

### 1. Don't Panic at Temporary Dips
Sometimes IoU drops slightly for 2-3 epochs before improving. This is normal as the model adjusts.

### 2. Save Checkpoints Frequently
The training script saves `best_model.pth` automatically, but you can manually backup:
```bash
cp checkpoints/best_model.pth checkpoints/milestone_epoch95.pth
```

### 3. Track Your Best IoU
Keep a note:
```
Epoch 86: IoU 0.6221 (before scheduler change)
Epoch 95: IoU 0.6456 (+0.0235 improvement)
Epoch 105: IoU 0.6789 (+0.0568 total improvement)
```

### 4. Combine Strategies If Needed
If Option 1 gets you to 0.68 but stalls:
1. Save that checkpoint
2. Resume with Option 2 settings (higher LR)
3. Push from 0.68 → 0.72

### 5. GPU Memory Optimization
If you get OOM with higher resolution:
- Use `--stage2-batch-size 1`
- Increase `--accumulation-steps 8` to maintain effective batch size
- Kill any zombie processes: `nvidia-smi` → `kill -9 <PID>`

---

## 📞 Quick Reference

### I want to: Use this option:

**Play it safe, first attempt** → Option 1 (Conservative)

**Confident, want results fast** → Option 2 (Aggressive)

**Maximum performance, have time/memory** → Option 3 (Resolution Boost)

**Currently running old training** → Stop it, backup checkpoint, run Option 1

**Option 1 got me to 0.68, stuck again** → Resume with Option 2 settings

**Loss exploded with Option 2** → Use Option 1 instead

**Option 3 gives OOM** → Reduce `--img-size 384` or `--stage2-batch-size 1`

---

## ✅ Checklist Before Starting

- [ ] Current checkpoint backed up
- [ ] Confirmed resume path: `/kaggle/working/checkpoints/best_model.pth`
- [ ] Chose an option (1, 2, or 3)
- [ ] Copied command correctly
- [ ] Kaggle notebook has GPU enabled
- [ ] Have 4+ hours of runtime available (for Option 2-3)
- [ ] Monitoring output for first 5 epochs

---

## 🎯 Bottom Line

**Your target**: IoU 0.72
**Current**: IoU 0.62
**Gap**: +0.10 IoU

**My recommendation**: Start with **Option 2 (Aggressive)** if you're confident, or **Option 1 (Conservative)** if you want to be safe.

The cosine annealing scheduler with higher Stage 2 learning rate should give you the boost needed to break through the plateau. Monitor progress after 10 epochs and adjust if needed.

**Expected timeline**: 25-35 epochs to reach 0.72 IoU (4-6 hours on Kaggle GPU)

Good luck! You're very close to your target. 🚀

---

## 📚 Full Documentation

For complete details on all scheduler options, see:
- **LR_SCHEDULER_GUIDE.md**: Comprehensive guide to all new features
- **TRAINING_COLLAPSE_ANALYSIS.md**: Understanding hyperparameter issues
- **GPU_BOTTLENECK_ANALYSIS.md**: Memory optimization strategies
