#!/bin/bash
# Quick helper script to resume Stage 2 training from checkpoint

set -e  # Exit on error

# Default values
CHECKPOINT="checkpoints/best_model.pth"
DATASET="/kaggle/input/cod10k"
BATCH_SIZE=1
ACCUM_STEPS=4

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "                   CAMOXPERT STAGE 2 RESUME HELPER"
echo "========================================================================"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}❌ ERROR: Checkpoint not found at $CHECKPOINT${NC}"
    echo ""
    echo "Available checkpoints:"
    ls -lh checkpoints/ 2>/dev/null || echo "  No checkpoints directory found"
    echo ""
    echo "Usage: bash resume_stage2.sh [checkpoint_path] [dataset_path]"
    exit 1
fi

# Load checkpoint info
echo -e "${GREEN}✓ Checkpoint found: $CHECKPOINT${NC}"
python3 -c "
import torch
import sys
try:
    ckpt = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
    print(f\"  Epoch: {ckpt['epoch']}\")
    print(f\"  Best IoU: {ckpt['best_iou']:.4f}\")
    if ckpt['epoch'] >= 30:
        print('  Stage: 1 (Decoder training complete)')
    else:
        print('  Stage: 1 (Still in decoder training)')
        print('')
        print('  ⚠️  WARNING: This checkpoint is from Stage 1.')
        print('     Consider completing Stage 1 first, or use --skip-stage1 to force Stage 2.')
except Exception as e:
    print(f'Error loading checkpoint: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "========================================================================"
echo "                        CONFIGURATION"
echo "========================================================================"
echo "Checkpoint:          $CHECKPOINT"
echo "Dataset:             $DATASET"
echo "Stage 2 Batch Size:  $BATCH_SIZE"
echo "Accumulation Steps:  $ACCUM_STEPS"
echo "Effective Batch:     $((BATCH_SIZE * ACCUM_STEPS))"
echo "Memory Optimization: Gradient Checkpointing + Progressive Unfreeze"
echo "========================================================================"
echo ""

# Check GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}⚠️  nvidia-smi not found, skipping GPU check${NC}"
    echo ""
fi

# Confirm
echo -e "${YELLOW}Ready to resume Stage 2 training. This will:${NC}"
echo "  1. Load checkpoint from $CHECKPOINT"
echo "  2. Skip Stage 1 (decoder training)"
echo "  3. Start Stage 2 (full fine-tuning) with memory optimizations"
echo "  4. Use batch_size=$BATCH_SIZE with gradient accumulation"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "========================================================================"
echo "                    STARTING STAGE 2 TRAINING"
echo "========================================================================"
echo ""

# Run training
python3 train_ultimate.py train \
  --dataset-path "$DATASET" \
  --resume-from "$CHECKPOINT" \
  --skip-stage1 \
  --stage2-batch-size "$BATCH_SIZE" \
  --accumulation-steps "$ACCUM_STEPS" \
  --gradient-checkpointing \
  --progressive-unfreeze \
  --batch-size 2

echo ""
echo "========================================================================"
echo "                    TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Check your results:"
echo "  - Best model: checkpoints/best_model.pth"
echo "  - Training history: checkpoints/history.json"
echo ""
