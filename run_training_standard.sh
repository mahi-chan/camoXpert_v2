#!/bin/bash
# Multi-GPU Training with Standard CamoXpert (Dense MoE, 7 experts)
# Usage: bash run_training_standard.sh

torchrun --nproc_per_node=2 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints \
    --backbone pvt_v2_b2 \
    --num-experts 7 \
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
