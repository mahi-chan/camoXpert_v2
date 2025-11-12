"""
Simple DDP launcher for Kaggle/Jupyter environments
Run with: python launch_ddp.py
"""
import os
import torch

# Set DDP environment variables
ngpus = torch.cuda.device_count()
print(f"Launching DDP training with {ngpus} GPUs...")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Launch with torchrun (recommended for PyTorch 1.10+)
cmd = f"""
torchrun --nproc_per_node={ngpus} --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --use-cod-specialized \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --batch-size 32 \
    --accumulation-steps 1 \
    --img-size 352 \
    --epochs 150 \
    --stage1-epochs 30 \
    --lr 0.00025 \
    --backbone edgenext_base \
    --num-experts 7 \
    --scheduler onecycle \
    --gradient-checkpointing \
    --use-ema \
    --ema-decay 0.9999
"""

print(f"Command: {cmd}\n")
os.system(cmd)
