#!/bin/bash
# High-Resolution DDP Training Script with Sparse MoE - Targeting IoU 0.75
# Run with: bash train_ddp_custom.sh

echo "=================================================="
echo "CAMOXPERT SPARSE MOE TRAINING - TARGETING IoU 0.75"
echo "=================================================="
echo "GPUs: 2 × Tesla T4"
echo "Resolution: 416px (vs 352px baseline)"
echo "Sparse MoE: Enabled (6 experts, top-2 selection)"
echo "Expert Routing: Learned (adapts to image type)"
echo "Total Batch: Stage 1 = 24, Stage 2 = 16"
echo "Effective Batch (w/ grad accumulation): 48 / 32"
echo "Mixed Precision: Enabled (AMP)"
echo "Gradient Checkpointing: Enabled"
echo "Epochs: 200 (40 Stage 1 + 160 Stage 2)"
echo "Expected: 2.0-2.2 min/epoch (35-40% faster)"
echo "=================================================="
echo ""

torchrun --nproc_per_node=2 --master_port=29500 train_ultimate.py train \
    --use-ddp \
    --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 \
    --checkpoint-dir /kaggle/working/checkpoints_sparse_moe \
    --backbone edgenext_base \
    --num-experts 7 \
    --batch-size 12 \
    --stage2-batch-size 8 \
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

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
