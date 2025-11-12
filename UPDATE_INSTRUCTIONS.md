# 🔄 How to Update to Latest Version with Resume Feature

## Issue
You're seeing: `error: unrecognized arguments: --resume-from --skip-stage1`

## Solution
You need to pull the latest code that includes the checkpoint resume feature.

## 📥 **Method 1: Pull Latest Changes (If you have git access)**

```bash
cd /kaggle/working/camoXpert  # or wherever your code is
git pull origin claude/investigate-gpu-bottleneck-011CUdzKFPf87kvDNa4Za2Y2
```

## 📥 **Method 2: Download Updated File Directly**

If you're on Kaggle and can't pull, download the updated `train_ultimate.py`:

```python
# In Kaggle notebook
import requests

# Download updated training script
url = "https://raw.githubusercontent.com/YOUR_REPO/camoXpert/claude/investigate-gpu-bottleneck-011CUdzKFPf87kvDNa4Za2Y2/train_ultimate.py"
response = requests.get(url)
with open('train_ultimate.py', 'w') as f:
    f.write(response.text)

print("✓ Updated train_ultimate.py downloaded!")
```

## 📥 **Method 3: Use Simplified Command (No Resume)**

If you can't update the file, here's a workaround to manually start Stage 2:

### Step 1: Modify the script to start at epoch 30

```python
# Add this at the top of your notebook to patch the script
with open('train_ultimate.py', 'r') as f:
    code = f.read()

# Replace stage 1 epochs with 0 to skip it
code = code.replace('--stage1-epochs', type=int, default=30',
                   '--stage1-epochs', type=int, default=0')

with open('train_ultimate.py', 'w') as f:
    f.write(code)
```

Then load checkpoint manually:

```python
import torch

# Load model
model = CamoXpert(3, 1, pretrained=True, backbone='edgenext_base', num_experts=7).cuda()

# Load checkpoint weights
checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"✓ Best IoU: {checkpoint['best_iou']:.4f}")
```

## ✅ **Method 4: Fresh Clone (Cleanest)**

Start fresh with the latest code:

```bash
# Backup your checkpoint first!
cp checkpoints/best_model.pth /kaggle/working/backup_checkpoint.pth

# Clone latest version
cd /kaggle/working
rm -rf camoXpert
git clone -b claude/investigate-gpu-bottleneck-011CUdzKFPf87kvDNa4Za2Y2 \
  https://github.com/YOUR_REPO/camoXpert.git
cd camoXpert

# Restore checkpoint
mkdir -p checkpoints
cp /kaggle/working/backup_checkpoint.pth checkpoints/best_model.pth

# Now run with resume
python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 1 \
  --gradient-checkpointing
```

## 🔍 **Verify You Have Latest Version**

Run this to check:

```python
# Check if resume arguments exist
with open('train_ultimate.py', 'r') as f:
    content = f.read()

if '--resume-from' in content and '--skip-stage1' in content:
    print("✅ You have the latest version with resume support!")
else:
    print("❌ Old version - needs update")
    print("   Use one of the update methods above")
```

## 📌 **Alternative: Manual Stage 2 Start (No Code Update Needed)**

If you can't update the code, here's a complete workaround:

```python
# manual_stage2.py - Save this as a new file
import torch
from train_ultimate import *

# Setup
args = parse_args()
args.dataset_path = '/kaggle/input/cod10k'
args.stage1_epochs = 0  # Skip Stage 1
args.stage2_batch_size = 1
args.gradient_checkpointing = True

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)

# Create model and load weights
model = CamoXpert(3, 1, pretrained=False, backbone=args.backbone,
                  num_experts=args.num_experts).cuda()
model.load_state_dict(checkpoint['model_state_dict'])

# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Continue with Stage 2 training
# ... (use train() function but start from stage 2)
```

## 🎯 **Once Updated, Run:**

```bash
python train_ultimate.py train \
  --dataset-path /kaggle/input/cod10k \
  --resume-from checkpoints/best_model.pth \
  --skip-stage1 \
  --stage2-batch-size 1 \
  --gradient-checkpointing \
  --progressive-unfreeze
```

## 📞 **Still Having Issues?**

If none of these work, please share:
1. Your current git branch: `git branch`
2. Last commit: `git log -1 --oneline`
3. File modification date: `ls -lh train_ultimate.py`

This will help diagnose the issue!
