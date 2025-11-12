# Learning Rate Scheduler Guide

## 🎯 Problem: Training Plateau

When training deep learning models, you may encounter plateaus where the model stops improving despite continued training. This often manifests as:

- IoU/Dice scores stagnating for many epochs
- Loss decreasing very slowly or oscillating
- Need for higher performance but current approach isn't working

**Example**: IoU stuck at ~0.62 for 17+ epochs, need to reach 0.72

The new flexible learning rate scheduling system addresses this by allowing dynamic learning rate adjustment throughout training.

---

## 📚 New Parameters

### `--stage2-lr FLOAT`
**Purpose**: Set a different learning rate for Stage 2 (fine-tuning)

**Default**: Same as `--lr` (Stage 1 learning rate)

**Why use it**: Stage 2 unfreezes the backbone, which may benefit from a different learning rate than Stage 1's decoder-only training. Often you want a higher LR in Stage 2 to escape plateaus.

**Example**:
```bash
--lr 0.00025 --stage2-lr 0.0002  # Stage 1: 0.00025, Stage 2: 0.0002
```

### `--scheduler {onecycle,cosine,cosine_restart,none}`
**Purpose**: Choose learning rate scheduling strategy

**Default**: `onecycle`

**Options**:
- `onecycle`: One cycle learning rate policy (good for stable training)
- `cosine`: Cosine annealing (smooth decay from max to min LR)
- `cosine_restart`: Cosine annealing with warm restarts (periodic LR resets)
- `none`: No scheduling (constant LR)

**Example**:
```bash
--scheduler cosine  # Use cosine annealing
```

### `--min-lr FLOAT`
**Purpose**: Minimum learning rate for cosine schedulers

**Default**: `1e-6`

**Why use it**: Prevents LR from becoming too small, which can cause training to stall. The cosine scheduler will decay from your initial LR down to this minimum.

**Example**:
```bash
--scheduler cosine --min-lr 0.00001  # LR decays to 0.00001
```

### `--warmup-epochs INT`
**Purpose**: Number of warmup epochs at the start of Stage 2

**Default**: `0` (no warmup)

**Why use it**: When transitioning from Stage 1 (frozen backbone) to Stage 2 (unfrozen), a warmup period gradually increases the learning rate, providing stability as the model adjusts to training all parameters.

**Example**:
```bash
--warmup-epochs 5  # Gradually increase LR over first 5 epochs of Stage 2
```

### `--t-mult INT`
**Purpose**: T_mult parameter for cosine_restart scheduler

**Default**: `2`

**Why use it**: Controls how quickly restart periods increase. With `t_mult=2`, each restart cycle is 2x longer than the previous.

**Example**:
```bash
--scheduler cosine_restart --t-mult 2  # Restart periods: T₀, 2T₀, 4T₀, ...
```

---

## 🔄 Scheduler Comparison

### OneCycle (Default)
**Best for**: Stable, predictable training from scratch

**How it works**:
1. LR increases from 0 to max over first 10% of training
2. LR decreases back to near 0 over remaining 90%

**Pros**:
- Very stable
- Works well for most cases
- Good default choice

**Cons**:
- Not ideal for resuming from checkpoints
- Can't easily escape plateaus mid-training

**Use when**: Starting fresh training with good hyperparameters

### Cosine Annealing
**Best for**: Escaping plateaus, smooth convergence

**How it works**:
- LR follows a cosine curve from max to min over the entire training period
- Smooth, gradual decrease

**Pros**:
- Helps escape local minima with initial high LR
- Smooth convergence at the end
- Great for fine-tuning
- Works well when resuming training

**Cons**:
- No LR restarts (once it decays, it stays low)

**Use when**:
- Model is stuck at a plateau
- Fine-tuning in Stage 2
- You want smooth, predictable LR decay

### Cosine with Warm Restarts
**Best for**: Long training, avoiding stagnation

**How it works**:
- LR follows cosine curve but periodically resets to max
- Each cycle can be longer than the previous (controlled by `t_mult`)

**Pros**:
- Periodic high LR helps jump out of local minima
- Good for very long training runs
- Can recover from temporary plateaus

**Cons**:
- More complex behavior
- May cause temporary instability during restarts

**Use when**:
- Training for 100+ epochs
- Want insurance against getting stuck
- Previous runs showed multiple plateaus

### None
**Best for**: Manual control, debugging

**How it works**: LR stays constant throughout training

**Use when**:
- You want full manual control
- Debugging issues
- Very short training runs

---

## 🚀 Recommended Configurations

### Scenario 1: Breaking Through IoU Plateau (Your Case)

**Problem**: Model stuck at IoU 0.62, needs to reach 0.72

**Solution**: Use cosine annealing with higher Stage 2 LR

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
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
    --warmup-epochs 3 \
    --progressive-unfreeze \
    --num-workers 4
```

**Why this works**:
- `--stage2-lr 0.0002`: Higher than stage 1, helps escape plateau
- `--scheduler cosine`: Smooth decay prevents oscillations
- `--min-lr 0.00001`: Prevents LR from becoming too small
- `--warmup-epochs 3`: Stable transition from Stage 1 to Stage 2
- `--img-size 384`: Higher resolution captures fine details (vs 320)

**Expected improvement**: +0.05 to +0.10 IoU over 20-30 epochs

### Scenario 2: Resuming from Checkpoint at Plateau

**Problem**: Already trained for 86 epochs, stuck at 0.62, want to continue with new scheduler

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
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
    --min-lr 0.00001 \
    --skip-stage1 \
    --num-workers 4
```

**Key changes**:
- `--resume-from`: Load existing checkpoint
- `--skip-stage1`: Go directly to Stage 2
- `--stage2-lr 0.0003`: Even higher LR to break plateau
- `--epochs 150`: Extend total training (150-86=64 more epochs)

### Scenario 3: Fresh Training with Optimal Settings

**Problem**: Starting from scratch, want best chance of success

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_optimal \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 2 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 120 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.00025 \
    --scheduler cosine_restart \
    --min-lr 0.00001 \
    --warmup-epochs 5 \
    --t-mult 2 \
    --progressive-unfreeze \
    --num-workers 4
```

**Why this works**:
- `cosine_restart`: Insurance against plateaus over long training
- `--warmup-epochs 5`: Smooth Stage 1→2 transition
- `--progressive-unfreeze`: Memory efficient, gradual adaptation

### Scenario 4: Aggressive Plateau Breaking

**Problem**: Tried cosine annealing, still stuck. Need maximum learning boost.

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --resume-from /kaggle/working/checkpoints/best_model.pth \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 2 \
    --stage2-batch-size 1 \
    --accumulation-steps 4 \
    --img-size 384 \
    --epochs 140 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --stage2-lr 0.0005 \
    --scheduler cosine \
    --min-lr 0.00005 \
    --skip-stage1 \
    --num-workers 4
```

**Changes**:
- `--stage2-lr 0.0005`: 2x higher than default (aggressive)
- `--min-lr 0.00005`: 5x higher minimum to maintain learning pressure
- No warmup: Resume assumes model is already stable

**Warning**: Monitor for instability. If loss explodes, reduce `--stage2-lr` to 0.0003

---

## 📊 How Learning Rate Changes During Training

### Cosine Annealing Example

```
Epochs:  31    40    50    60    70    80    90   100   110   120
         |-----|-----|-----|-----|-----|-----|-----|-----|-----|
LR:   0.0002 ──────────────────────────────────────────── 0.00001
         ╱‾‾‾‾╲                                        ╱‾‾╲
        ╱       ╲                                     ╱     ╲
       ╱         ╲___________________________________╱       ╲
    Start      Cosine decay from max to min                  End
```

**Stage 1 (epochs 0-30)**: Uses `--lr` with onecycle scheduler
**Stage 2 (epochs 31-120)**: Uses `--stage2-lr` with cosine decay to `--min-lr`

### Cosine with Warmup Example

```
Epochs:  31  32  33  34  35  36 ...  80  ...  120
         |---|---|---|---|---|---|-----------|-----|
LR:   0.00004                 0.0002         0.00001
         ──→──→──→──→──→  (warmup)  ──────→  (cosine)
```

**Warmup period (epochs 31-35)**: LR gradually increases from `min_lr` to `stage2_lr`
**Main training (epochs 36-120)**: Cosine decay from `stage2_lr` to `min_lr`

---

## 🔍 Monitoring and Debugging

### What to Watch

Look for these patterns in your training output:

**Good signs** ✅:
```
Epoch 87: Loss: 0.2341 | IoU: 0.6350 | Dice: 0.7456 | LR: 0.000180
Epoch 88: Loss: 0.2298 | IoU: 0.6421 | Dice: 0.7523 | LR: 0.000175
Epoch 89: Loss: 0.2267 | IoU: 0.6489 | Dice: 0.7601 | LR: 0.000170
```
- IoU steadily increasing
- Loss decreasing
- LR gradually decreasing (cosine)

**Warning signs** ⚠️:
```
Epoch 87: Loss: 0.2341 | IoU: 0.6350 | Dice: 0.7456 | LR: 0.000180
Epoch 88: Loss: 0.2299 | IoU: 0.6348 | Dice: 0.7453 | LR: 0.000175
Epoch 89: Loss: 0.2301 | IoU: 0.6351 | Dice: 0.7455 | LR: 0.000170
```
- IoU oscillating within narrow range
- Still at plateau despite LR changes
- May need higher `--stage2-lr` or higher `--min-lr`

**Bad signs** ❌:
```
Epoch 87: Loss: 0.2341 | IoU: 0.6350 | Dice: 0.7456 | LR: 0.000500
Epoch 88: Loss: 1.5678 | IoU: 0.1234 | Dice: 0.2345 | LR: 0.000495
Epoch 89: Loss: 2.3456 | IoU: 0.0456 | Dice: 0.0987 | LR: 0.000490
```
- Loss exploding
- IoU collapsing
- LR is too high - reduce `--stage2-lr`

### Interpreting Results

**If IoU improves slowly but steadily**:
- ✅ Good! Let it continue
- Current settings are working
- May take 20-30 epochs to see +0.05-0.10 improvement

**If IoU stays flat**:
- Increase `--stage2-lr` (try 1.5x to 2x current value)
- Increase `--min-lr` to maintain learning pressure longer
- Consider increasing `--img-size` for better feature resolution

**If training becomes unstable**:
- Reduce `--stage2-lr` (try 0.7x current value)
- Add/increase `--warmup-epochs` (try 5-10)
- Check that batch size isn't too large

---

## 🛠️ Troubleshooting

### Q: My training is still stuck after using cosine scheduler

**A**: Try these escalations:

1. **Increase Stage 2 LR**: Current * 1.5
2. **Increase minimum LR**: `--min-lr 0.00005` (5x default)
3. **Increase resolution**: `--img-size 416` or `448`
4. **Check batch size/LR ratio**: Should maintain `LR / effective_batch ≈ 0.00003`

### Q: Loss exploded when I increased LR

**A**: LR was too aggressive:

1. **Reduce Stage 2 LR**: Try 0.0002 instead of 0.0003
2. **Add warmup**: `--warmup-epochs 5` to ease into higher LR
3. **Check gradient clipping**: Default is 1.0, may need adjustment
4. **Verify batch size**: Very small batches (1-2) can be unstable with high LR

### Q: Which scheduler should I use?

**A**: Decision tree:

- **Fresh training, uncertain settings** → `onecycle` (default)
- **Stuck at plateau, want to improve** → `cosine`
- **Very long training (150+ epochs)** → `cosine_restart`
- **Debugging or manual control** → `none`

### Q: Should I use warmup for Stage 2?

**A**: Use warmup if:

- ✅ Stage 2 LR is significantly higher than Stage 1 LR
- ✅ You've seen instability at the Stage 1→2 transition
- ✅ Using very high LR (>0.0003)

Skip warmup if:
- ❌ Stage 2 LR is same or lower than Stage 1
- ❌ Training has been stable
- ❌ Resuming from Stage 2 checkpoint (already stable)

### Q: Can I change scheduler mid-training?

**A**: Yes! When resuming:

```bash
# Original training
python train_ultimate.py train ... --scheduler onecycle

# Resume with different scheduler
python train_ultimate.py train ... \
    --resume-from checkpoints/best_model.pth \
    --skip-stage1 \
    --scheduler cosine \
    --stage2-lr 0.0002
```

The new scheduler will take effect from the resumed epoch forward.

---

## 📈 Expected Performance Gains

Based on typical scenarios:

| Starting IoU | Target IoU | Recommended Action | Expected Gain | Time |
|--------------|------------|-------------------|---------------|------|
| 0.54 | 0.60 | Continue with onecycle | +0.06 | 10-15 epochs |
| 0.60 | 0.65 | Switch to cosine, lr=0.0002 | +0.05 | 15-20 epochs |
| 0.62 (plateau) | 0.68 | Cosine, lr=0.0002, img=384 | +0.06 | 20-30 epochs |
| 0.62 (plateau) | 0.72 | Cosine, lr=0.0003, img=416 | +0.10 | 30-40 epochs |
| 0.68 | 0.72 | Cosine_restart, lr=0.00025 | +0.04 | 20-30 epochs |

**Note**: Gains are approximate and depend on dataset, model, and other factors.

---

## 🎓 Advanced Tips

### 1. Combining with Progressive Unfreezing

Progressive unfreezing works great with warmup:

```bash
--progressive-unfreeze --warmup-epochs 5
```

This gradually unfreezes backbone layers while ramping up LR, providing double stability.

### 2. Resolution Scaling

If scheduler changes aren't enough, try increasing image size:

```bash
--img-size 384  # Standard
--img-size 416  # Better (if memory allows)
--img-size 448  # Best (may need --stage2-batch-size 1)
```

Higher resolution = more details = better camouflage detection

### 3. Finding Optimal Stage 2 LR

If unsure what `--stage2-lr` to use:

1. Start with same as Stage 1: `--stage2-lr` = `--lr`
2. If plateau persists after 10 epochs, increase by 50%
3. Keep increasing until you see improvement OR instability
4. Back off 20% if you hit instability

Example progression:
```
Try 1: --stage2-lr 0.00025  (same as --lr)
Try 2: --stage2-lr 0.0002   (if still stuck)
Try 3: --stage2-lr 0.0003   (if still stuck)
Try 4: --stage2-lr 0.0002   (if 0.0003 was unstable)
```

### 4. Batch Size Considerations

Learning rate and batch size are linked:

**Rule of thumb**: `LR ∝ sqrt(batch_size)`

If you change batch size:
```bash
# Original
--batch-size 2 --stage2-batch-size 1 --stage2-lr 0.0002

# Doubled batch size
--batch-size 4 --stage2-batch-size 2 --stage2-lr 0.0003  # ~1.4x LR
```

---

## 📝 Summary

**For breaking through IoU plateau**:

1. Use `--scheduler cosine` for smooth LR decay
2. Set `--stage2-lr` 0.8x to 1.2x your Stage 1 LR initially
3. Add `--warmup-epochs 3` for stability
4. Set `--min-lr 0.00001` to prevent LR from becoming too small
5. Monitor LR in training output
6. Adjust `--stage2-lr` up if still stuck, down if unstable

**Quick commands**:

```bash
# Standard plateau breaking
--scheduler cosine --stage2-lr 0.0002 --min-lr 0.00001 --warmup-epochs 3

# Aggressive plateau breaking
--scheduler cosine --stage2-lr 0.0003 --min-lr 0.00005

# Long training with insurance
--scheduler cosine_restart --stage2-lr 0.00025 --min-lr 0.00001 --warmup-epochs 5
```

**Remember**: Patience is key. Improvements from scheduler changes may take 20-30 epochs to fully manifest. Monitor the LR in output and watch for steady (even if slow) IoU increases.

Good luck breaking through that plateau! 🚀
