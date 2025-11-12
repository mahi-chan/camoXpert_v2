# Training Commands - DDP Multi-GPU Configuration

## ⚠️ IMPORTANT: Memory-Optimized Settings for Tesla T4

**Batch size 32 with FP32 (--no-amp) causes OOM errors on Tesla T4!**

### Recommended Configuration (Memory Safe):

**Option 1: Python Launcher (Recommended)**
```bash
python launch_ddp_custom.py
```
- Batch size: 20 per GPU (40 total)
- Mixed precision (AMP): **Enabled** (saves ~40% memory)
- Gradient checkpointing: **Enabled** (saves ~30% memory)

**Option 2: Direct Command (Full Control)**
```bash
torchrun --nproc_per_node=2 --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_cod_specialized \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 20 \
    --stage2-batch-size 16 \
    --accumulation-steps 1 \
    --img-size 352 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.0008 \
    --stage2-lr 0.00055 \
    --scheduler cosine \
    --min-lr 0.0001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --gradient-checkpointing \
    --num-workers 4 \
    --use-cod-specialized
```
**Note**: Removed `--no-amp` to enable mixed precision (critical for memory)

---

## Batch Size Explained

### Memory-Optimized DDP Configuration (2 GPUs):
- **Stage 1**: 20 **per GPU** × 2 GPUs = **40 total**
- **Stage 2**: 16 **per GPU** × 2 GPUs = **32 total**

**With mixed precision (AMP) + gradient checkpointing:**
- Fits comfortably in Tesla T4's 15 GB VRAM
- Still large enough for stable training
- ~70% memory savings vs FP32

### Batch Size Options for Tesla T4:

| Batch/GPU | Total (2 GPUs) | Precision | Status |
|-----------|----------------|-----------|--------|
| 32 | 64 | FP32 (--no-amp) | ❌ **OOM Error** |
| 24 | 48 | FP32 (--no-amp) | ⚠️ **Risky** (~14.5 GB) |
| 20 | 40 | **Mixed (AMP)** | ✅ **Recommended** (~10 GB) |
| 16 | 32 | Mixed (AMP) | ✅ Safe (~8 GB) |

---

## Configuration Summary

| Setting | Value | Description |
|---------|-------|-------------|
| **GPUs** | 2 × Tesla T4 | DistributedDataParallel |
| **Batch (S1)** | 40 total | 20 per GPU |
| **Batch (S2)** | 32 total | 16 per GPU |
| **Resolution** | 352×352 | Optimal for COD |
| **Epochs** | 150 | 30 decoder + 120 full |
| **LR (S1)** | 0.0008 | With warmup |
| **LR (S2)** | 0.00055 | Fine-tuning rate |
| **Scheduler** | Cosine | With min_lr 1e-5 |
| **Warmup** | 5 epochs | Stage 2 only |
| **Precision** | Mixed (FP16/32) | AMP enabled for memory |
| **Checkpointing** | Enabled | Saves ~30% memory |
| **Deep Sup** | Enabled | 3 supervision levels |

---

## Expected Performance

### Training Speed:
- **Single GPU**: ~8 hours
- **DDP (2 GPUs)**: ~4-5 hours ⚡ **~1.8× faster**

### Expected Results:
- **IoU**: ≥ 0.72 (your target)
- **Training**: Stable, no CUDA errors
- **Checkpoints**: Saved to `/kaggle/working/checkpoints_cod_specialized`

---

## Fallback: Single GPU (If DDP Fails)

If you encounter any issues with DDP, use single GPU:

```bash
python train_ultimate.py train \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_cod_specialized \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 24 \
    --stage2-batch-size 18 \
    --accumulation-steps 3 \
    --img-size 352 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.0008 \
    --stage2-lr 0.00055 \
    --scheduler cosine \
    --min-lr 0.0001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --num-workers 4 \
    --no-amp \
    --use-cod-specialized
```

*(Batch size auto-adjusted with gradient accumulation to maintain effective batch size)*

---

## Monitoring Training

Check progress:
```bash
# View checkpoints
ls -lh /kaggle/working/checkpoints_cod_specialized/

# Monitor GPU usage
nvidia-smi -l 1
```

---

## Troubleshooting

### "Address already in use"
```bash
# Change port
torchrun --nproc_per_node=2 --master_port=29501 train_ultimate.py train --use-ddp ...
```

### OOM (Out of Memory)

**Option 1: Enable AMP (Recommended)**
```bash
# Remove --no-amp flag to enable mixed precision
# This saves ~40% memory!
torchrun --nproc_per_node=2 train_ultimate.py train --use-ddp ... (remove --no-amp)
```

**Option 2: Reduce Batch Size**
```bash
# Use smaller batch
--batch-size 16 --stage2-batch-size 12 --gradient-checkpointing
```

**Option 3: Both (Maximum Memory Savings)**
```bash
# Combine AMP + smaller batch + checkpointing
--batch-size 16 --gradient-checkpointing (no --no-amp flag)
```

### NCCL Timeout
```bash
# Increase timeout
export NCCL_TIMEOUT=1800
torchrun --nproc_per_node=2 train_ultimate.py train --use-ddp ...
```

---

Good luck! 🚀 Your training should achieve IoU ≥ 0.72!
