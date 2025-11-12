#!/usr/bin/env python3
"""
Quick version checker for CamoXpert training script
"""
import os
import sys

def check_resume_support():
    """Check if train_ultimate.py has resume support"""
    if not os.path.exists('train_ultimate.py'):
        print("❌ ERROR: train_ultimate.py not found in current directory")
        print(f"   Current directory: {os.getcwd()}")
        return False

    with open('train_ultimate.py', 'r') as f:
        content = f.read()

    has_resume = '--resume-from' in content
    has_skip = '--skip-stage1' in content
    has_stage2_batch = '--stage2-batch-size' in content
    has_progressive = '--progressive-unfreeze' in content

    print("="*70)
    print("CAMOXPERT VERSION CHECK")
    print("="*70)
    print(f"{'Feature':<30} {'Status':<10}")
    print("-"*70)
    print(f"{'Checkpoint Resume':<30} {'✅ Yes' if has_resume else '❌ No':<10}")
    print(f"{'Skip Stage 1':<30} {'✅ Yes' if has_skip else '❌ No':<10}")
    print(f"{'Stage 2 Batch Size':<30} {'✅ Yes' if has_stage2_batch else '❌ No':<10}")
    print(f"{'Progressive Unfreezing':<30} {'✅ Yes' if has_progressive else '❌ No':<10}")
    print("="*70)

    if all([has_resume, has_skip, has_stage2_batch, has_progressive]):
        print("\n✅ You have the LATEST VERSION with all features!")
        print("\nYou can now run:")
        print("  python train_ultimate.py train \\")
        print("    --dataset-path /kaggle/input/cod10k \\")
        print("    --resume-from checkpoints/best_model.pth \\")
        print("    --skip-stage1 \\")
        print("    --stage2-batch-size 1 \\")
        print("    --gradient-checkpointing \\")
        print("    --progressive-unfreeze")
        return True
    else:
        print("\n❌ You have an OLD VERSION")
        print("\n🔄 To update:")
        print("  1. Pull latest changes:")
        print("     git pull origin claude/investigate-gpu-bottleneck-011CUdzKFPf87kvDNa4Za2Y2")
        print("\n  2. Or re-clone:")
        print("     git clone -b claude/investigate-gpu-bottleneck-011CUdzKFPf87kvDNa4Za2Y2 \\")
        print("       <repo_url> camoXpert")
        print("\n  3. Or see UPDATE_INSTRUCTIONS.md for more options")
        return False

def check_checkpoint():
    """Check if checkpoint exists"""
    checkpoint_paths = [
        'checkpoints/best_model.pth',
        '/kaggle/working/checkpoints/best_model.pth',
        'best_model.pth'
    ]

    print("\n" + "="*70)
    print("CHECKPOINT CHECK")
    print("="*70)

    found = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✅ Checkpoint found: {path}")

            try:
                import torch
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                print(f"   Epoch: {ckpt.get('epoch', 'unknown')}")
                print(f"   Best IoU: {ckpt.get('best_iou', 'unknown'):.4f}")

                if ckpt.get('epoch', 0) >= 29:
                    print(f"   Status: ✅ Ready for Stage 2!")
                else:
                    print(f"   Status: ⚠️  Still in Stage 1 (epoch {ckpt.get('epoch', 0)})")

                found = True
                break
            except Exception as e:
                print(f"   ⚠️  Could not load checkpoint: {e}")
        else:
            print(f"❌ Not found: {path}")

    if not found:
        print("\n⚠️  No checkpoint found!")
        print("   Make sure your checkpoint is in one of these locations:")
        for path in checkpoint_paths:
            print(f"   - {path}")

    print("="*70)
    return found

def check_gpu():
    """Check GPU availability"""
    print("\n" + "="*70)
    print("GPU CHECK")
    print("="*70)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   Total Memory: {gpu_memory:.2f} GB")

            if gpu_memory < 13:
                print(f"   ⚠️  WARNING: Low GPU memory. Use --stage2-batch-size 1")
            else:
                print(f"   ✅ Sufficient memory for training")
        else:
            print("❌ No GPU available")
            print("   Training will be very slow on CPU")
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")

    print("="*70)

def main():
    print("\n" + "="*70)
    print("🔍 CAMOXPERT TRAINING READINESS CHECK")
    print("="*70 + "\n")

    has_latest = check_resume_support()
    has_checkpoint = check_checkpoint()
    check_gpu()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if has_latest and has_checkpoint:
        print("✅ ALL SYSTEMS GO!")
        print("\n🚀 Ready to resume Stage 2 training!")
        print("\nRun this command:")
        print("  python train_ultimate.py train \\")
        print("    --dataset-path /kaggle/input/cod10k \\")
        print("    --resume-from checkpoints/best_model.pth \\")
        print("    --skip-stage1 \\")
        print("    --stage2-batch-size 1 \\")
        print("    --gradient-checkpointing \\")
        print("    --progressive-unfreeze")
    else:
        print("⚠️  NOT READY")
        if not has_latest:
            print("   - Update your code (see UPDATE_INSTRUCTIONS.md)")
        if not has_checkpoint:
            print("   - Locate or create your checkpoint")
        print("\nSee UPDATE_INSTRUCTIONS.md for help!")

    print("="*70 + "\n")

if __name__ == '__main__':
    main()
