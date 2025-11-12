"""
Custom DDP launcher with memory-optimized configuration for Tesla T4
Run with: python launch_ddp_custom.py

High-resolution settings targeting IoU 0.75:
- Resolution: 416px (vs 352px baseline)
- Sparse MoE: Enabled (6 experts, top-2 selection, 35-40% faster)
- Batch size: 12 per GPU (24 total with 2 GPUs) - OOM-safe
- Accumulation: 2 steps (effective batch = 48)
- Epochs: 200 (40 Stage 1 + 160 Stage 2)
- Mixed precision (AMP) enabled for 40% memory savings
- Gradient checkpointing enabled
"""
import os
import torch

# Set DDP environment variables
ngpus = torch.cuda.device_count()
batch_size = 12  # Memory-safe for 416px on Tesla T4
stage2_batch = 8  # Even safer for stage 2 with high resolution

print(f"🚀 Launching DDP training with {ngpus} GPUs...")
print(f"   Resolution: 416px (targeting IoU 0.75)")
print(f"   Sparse MoE: Enabled (6 experts, top-2 selection)")
print(f"   Expert routing: Learned (adapts to image type)")
print(f"   Total batch size: Stage 1 = {batch_size * ngpus}, Stage 2 = {stage2_batch * ngpus}")
print(f"   Effective batch (w/ accumulation): Stage 1 = {batch_size * ngpus * 2}, Stage 2 = {stage2_batch * ngpus * 2}")
print(f"   Total epochs: 200 (40 Stage 1 + 160 Stage 2)")
print(f"   Mixed precision: Enabled (AMP)")
print(f"   Gradient checkpointing: Enabled")
print(f"   Memory optimization: Tesla T4 optimized\n")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Launch with memory-optimized configuration
cmd = f"""
torchrun --nproc_per_node={ngpus} --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_sparse_moe \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size {batch_size} \
    --stage2-batch-size {stage2_batch} \
    --accumulation-steps 2 \
    --img-size 416 \
    --epochs 200 \
    --stage1-epochs 40 \
    --lr 0.0008 \
    --stage2-lr 0.0006 \
    --scheduler cosine \
    --min-lr 0.00001 \
    --warmup-epochs 5 \
    --deep-supervision \
    --gradient-checkpointing \
    --num-workers 4 \
    --use-cod-specialized \
    --use-sparse-moe \
    --moe-num-experts 6 \
    --moe-top-k 2
"""

print(f"\nCommand:\n{cmd}\n")
print("=" * 70)
os.system(cmd)
