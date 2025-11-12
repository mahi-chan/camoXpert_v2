#!/bin/bash
# DDP Multi-GPU Training Launcher for CamoXpert
# Usage: bash train_ddp.sh [args...]

# Number of GPUs to use
NGPUS=2

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    train_ultimate.py \
    train \
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
    --ema-decay 0.9999 \
    "$@"

echo "DDP training completed!"
