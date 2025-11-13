#!/bin/bash
# Multi-GPU Training with Sparse MoE (requires both --use-cod-specialized and --use-sparse-moe)
# Usage: bash run_training_sparse_moe.sh

torchrun --nproc_per_node=2 train_ultimate.py train \
    --use-ddp \
    --use-cod-specialized \
    --use-sparse-moe \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --backbone pvt_v2_b2 \
    --moe-num-experts 6 \
    --moe-top-k 2 \
    --batch-size 8 \
    --stage2-batch-size 8 \
    --accumulation-steps 4 \
    --img-size 416 \
    --epochs 200 \
    --stage1-epochs 40 \
    --lr 0.00025 \
    --stage2-lr 0.0004 \
    --scheduler cosine \
    --deep-supervision \
    --min-lr 0.00001 \
    --progressive-unfreeze \
    --num-workers 4
