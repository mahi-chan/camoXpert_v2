# CamoXpert Training Commands for Kaggle

## Option 1: Standard CamoXpert (Dense MoE, 7 experts) - RECOMMENDED

### Using bash script:
```bash
!bash /kaggle/working/camoXpert_v2/run_training_standard.sh
```

### Or use this single-line command in Kaggle:
```python
!torchrun --nproc_per_node=2 train_ultimate.py train --use-ddp --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 --checkpoint-dir /kaggle/working/checkpoints --backbone pvt_v2_b2 --num-experts 7 --batch-size 8 --stage2-batch-size 8 --accumulation-steps 4 --img-size 416 --epochs 200 --stage1-epochs 40 --lr 0.00025 --stage2-lr 0.0004 --scheduler cosine --deep-supervision --min-lr 0.00001 --progressive-unfreeze --num-workers 4
```

---

## Option 2: Sparse MoE (requires --use-cod-specialized flag)

### Using bash script:
```bash
!bash /kaggle/working/camoXpert_v2/run_training_sparse_moe.sh
```

### Or use this single-line command in Kaggle:
```python
!torchrun --nproc_per_node=2 train_ultimate.py train --use-ddp --use-cod-specialized --use-sparse-moe --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 --checkpoint-dir /kaggle/working/checkpoints --backbone pvt_v2_b2 --moe-num-experts 6 --moe-top-k 2 --batch-size 8 --stage2-batch-size 8 --accumulation-steps 4 --img-size 416 --epochs 200 --stage1-epochs 40 --lr 0.00025 --stage2-lr 0.0004 --scheduler cosine --deep-supervision --min-lr 0.00001 --progressive-unfreeze --num-workers 4
```

---

## Option 3: Single GPU (No DDP)

If you have issues with DDP, train on a single GPU:

```python
!python train_ultimate.py train --dataset-path /kaggle/input/cod10k-dataset/COD10K-v3 --checkpoint-dir /kaggle/working/checkpoints --backbone pvt_v2_b2 --num-experts 7 --batch-size 4 --accumulation-steps 8 --img-size 416 --epochs 200 --stage1-epochs 40 --lr 0.00025 --stage2-lr 0.0004 --scheduler cosine --deep-supervision --min-lr 0.00001 --progressive-unfreeze --num-workers 4
```

---

## Important Notes:

1. **Sparse MoE requires BOTH flags**: `--use-cod-specialized` AND `--use-sparse-moe`
2. **Don't use backslash `\` for line continuation in Kaggle** - use single-line commands or bash scripts
3. **Batch size with DDP**: With 2 GPUs, effective batch = 8 × 2 × 4 = 64
4. **Memory**: If you get OOM, reduce `--batch-size` to 4 or 6

---

## Debugging:

If torchrun fails, check:
1. Are you running in Kaggle with 2 GPUs enabled?
2. Remove any trailing spaces in your command
3. Try the single-GPU option first to test your setup
