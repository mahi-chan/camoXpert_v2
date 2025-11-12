# How to Resume Stage 2 Training from Checkpoint on Kaggle

## 🎯 Quick Start - Resume Stage 2 Training

After Stage 1 completed successfully (IoU: 0.5480), use this command to continue with Stage 2:

```bash
python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 1 \
  --gradient-checkpointing \
  --progressive-unfreeze \
  --batch-size 2 \
  --accumulation-steps 4
```

### ⚡ Key Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--resume-from` | `checkpoints/best_model.pth` | Load your Stage 1 checkpoint |
| `--skip-stage1` | flag | Jump directly to Stage 2 (epoch 31+) |
| `--stage2-batch-size` | `1` | Reduced batch for memory (default: 1) |
| `--gradient-checkpointing` | flag | Save ~25% memory |
| `--progressive-unfreeze` | flag | Gradual backbone unfreezing |
| `--batch-size` | `2` | Original Stage 1 batch size (for reference) |
| `--accumulation-steps` | `4` | Effective batch = 1×4 = 4 |

## 📋 Step-by-Step Instructions for Kaggle

### Step 1: Verify Your Checkpoint Exists

```python
import os
checkpoint_path = 'checkpoints/best_model.pth'
if os.path.exists(checkpoint_path):
    print(f"✅ Checkpoint found: {checkpoint_path}")
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"   Epoch: {ckpt['epoch']}")
    print(f"   Best IoU: {ckpt['best_iou']:.4f}")
else:
    print(f"❌ Checkpoint not found!")
```

### Step 2: Install Requirements (if needed)

```bash
!pip install timm segmentation-models-pytorch albumentations -q
```

### Step 3: Run Stage 2 Training

```python
!python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 1 \
  --gradient-checkpointing \
  --progressive-unfreeze
```

### Step 4: Monitor Training

You should see:
```
======================================================================
LOADING CHECKPOINT: checkpoints/best_model.pth
======================================================================
✓ Loaded checkpoint from epoch 29
✓ Best IoU so far: 0.5480
✓ Resuming from epoch 30
⚠️  WARNING: Checkpoint is from epoch 29 (Stage 1)
   You requested --skip-stage1, jumping to epoch 30
======================================================================

⏩ Skipping Stage 1 (resuming from epoch 30)

🧹 Cleaning up memory before Stage 2...
GPU Memory: X.XX GB allocated | X.XX GB reserved

======================================================================
STAGE 2: FULL FINE-TUNING
======================================================================
🔧 Reducing batch size: 2 → 1
📈 Using progressive unfreezing strategy
✓ Backbone: Last 1/4 layers unfrozen
   Trainable parameters: XX.XM

Epoch 31/120:  ...
```

## 🔧 Troubleshooting

### Problem: "Checkpoint not found"

**Solution**: Check checkpoint location
```python
# List checkpoint directory
!ls -lh checkpoints/

# If checkpoints are in a different location:
!python train_ultimate.py train \
  --resume-from /path/to/your/checkpoint.pth \
  --skip-stage1
```

### Problem: Still getting OOM

**Solution 1**: Use even smaller batch size
```bash
--stage2-batch-size 1 --accumulation-steps 8  # Effective batch = 8
```

**Solution 2**: Profile memory first
```bash
python memory_profiler.py --find-optimal --stage stage2
```

**Solution 3**: Reduce image size
```bash
--img-size 352  # Instead of 384
```

**Solution 4**: Fewer experts
```bash
--num-experts 5  # Instead of 7
```

### Problem: Training starts from epoch 0

**Issue**: Missing `--skip-stage1` flag or wrong checkpoint path

**Solution**: Always use both flags together:
```bash
--resume-from checkpoints/best_model.pth --skip-stage1
```

## 🎛️ Memory Optimization Strategies

### Conservative (Guaranteed to work on 15GB GPU)
```bash
python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 1 \
  --accumulation-steps 8 \
  --gradient-checkpointing \
  --progressive-unfreeze
```
- Effective batch: 1 × 8 = 8
- Memory: ~11-12 GB
- Speed: Slower but safe

### Balanced (Recommended)
```bash
python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 1 \
  --accumulation-steps 4 \
  --gradient-checkpointing
```
- Effective batch: 1 × 4 = 4
- Memory: ~13 GB
- Speed: Good balance

### Aggressive (Test with profiler first!)
```bash
python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 2 \
  --accumulation-steps 4
```
- Effective batch: 2 × 4 = 8
- Memory: ~14.5 GB
- Speed: Fastest (if it fits)

## 📊 What to Expect

### Stage 2 Training Timeline

| Epochs | Phase | Backbone Status | Expected IoU |
|--------|-------|-----------------|--------------|
| 31-60 | Progressive 1/3 | Last layer unfrozen | 0.55-0.62 |
| 61-90 | Progressive 2/3 | Last 2 layers unfrozen | 0.62-0.68 |
| 91-120 | Progressive 3/3 | All layers unfrozen | 0.68-0.72+ |

### Progressive Unfreezing Timeline

With `--progressive-unfreeze`:
- **Epochs 31-60**: Only last backbone layer trains (~20% of params)
- **Epochs 61-90**: Last 2 backbone layers train (~40% of params)
- **Epochs 91-120**: Full backbone trains (100% of params)

Without `--progressive-unfreeze`:
- **Epochs 31-120**: Full backbone trains immediately (higher memory)

## 🔍 Verifying It's Working

### Check 1: Checkpoint Loaded
```
✓ Loaded checkpoint from epoch 29
✓ Best IoU so far: 0.5480
```

### Check 2: Stage 1 Skipped
```
⏩ Skipping Stage 1 (resuming from epoch 30)
```

### Check 3: Stage 2 Started
```
STAGE 2: FULL FINE-TUNING
🔓 Unfreezing all parameters
```

### Check 4: Training Progressing
```
Epoch 31/120: 100%|████████| 250/250 [07:30<00:00,  3.60s/it, loss=4.6xxx]
```

## 💾 Saving Your Work on Kaggle

### Auto-save Checkpoints
The script automatically saves to `checkpoints/best_model.pth` whenever IoU improves.

### Manual Backup (Recommended)
```python
# After each milestone, copy checkpoint to output
!cp checkpoints/best_model.pth /kaggle/working/stage2_epoch60.pth
```

### Download Checkpoints
```python
from IPython.display import FileLink
FileLink('checkpoints/best_model.pth')
```

## 🚀 Complete Kaggle Notebook Example

```python
# Cell 1: Setup
!git clone https://github.com/your-repo/camoXpert.git
%cd camoXpert
!pip install -r requirements.txt -q

# Cell 2: Verify Checkpoint
import torch
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=False)
print(f"Checkpoint Epoch: {ckpt['epoch']}, IoU: {ckpt['best_iou']:.4f}")

# Cell 3: Resume Training
!python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 1 \
  --gradient-checkpointing \
  --progressive-unfreeze \
  --accumulation-steps 4

# Cell 4: Save Results
!cp checkpoints/best_model.pth /kaggle/working/final_model.pth
!cp checkpoints/history.json /kaggle/working/training_history.json
```

## 📞 Need Help?

**If training fails:**
1. Run memory profiler: `python memory_profiler.py --find-optimal --stage stage2`
2. Check GPU memory: `nvidia-smi`
3. Use conservative settings above
4. Review `GPU_BOTTLENECK_ANALYSIS.md` for detailed troubleshooting

**Common fixes:**
- OOM → Reduce `--stage2-batch-size` to 1, increase `--accumulation-steps`
- Too slow → Increase `--stage2-batch-size`, enable `--gradient-checkpointing`
- Not improving → Make sure `--skip-stage1` is set, check checkpoint loaded correctly
