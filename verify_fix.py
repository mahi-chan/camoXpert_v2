"""
Verification script to debug the dimension mismatch issue
Run this in Kaggle to check if the fix is working
"""

import sys
sys.path.insert(0, '/kaggle/working/camoXpert')

print("="*70)
print("VERIFICATION SCRIPT - Checking for dimension detection fix")
print("="*70)

# 1. Check if we're using the right code
print("\n[1/5] Checking git commit...")
import subprocess
result = subprocess.run(['git', '-C', '/kaggle/working/camoXpert', 'log', '-1', '--oneline'],
                       capture_output=True, text=True)
print(f"Current commit: {result.stdout.strip()}")
if '8bc9182' in result.stdout or 'Fix backbone dimension' in result.stdout:
    print("✅ Correct commit (has dimension auto-detection fix)")
else:
    print("❌ OLD commit (missing fix)")
    print("Please re-run: git pull origin claude/investigate-gpu-bottleneck-011CUdzKFPf87kvDNa4Za2Y2")

# 2. Check if the auto-detection code exists in the file
print("\n[2/5] Checking if auto-detection code exists...")
with open('/kaggle/working/camoXpert/models/camoxpert.py', 'r') as f:
    content = f.read()

if 'Detecting backbone feature dimensions' in content:
    print("✅ Auto-detection code found in camoxpert.py")
else:
    print("❌ Auto-detection code NOT found!")
    print("The file might not have been updated properly")

# 3. Check for cached .pyc files
print("\n[3/5] Checking for cached Python files...")
import os
import glob

pyc_files = glob.glob('/kaggle/working/camoXpert/**/*.pyc', recursive=True)
pycache_dirs = glob.glob('/kaggle/working/camoXpert/**/__pycache__', recursive=True)

if pyc_files or pycache_dirs:
    print(f"⚠️ Found {len(pyc_files)} .pyc files and {len(pycache_dirs)} __pycache__ dirs")
    print("Removing cached files...")
    for f in pyc_files:
        os.remove(f)
    for d in pycache_dirs:
        import shutil
        shutil.rmtree(d)
    print("✅ Cached files removed")
else:
    print("✅ No cached files found")

# 4. Try to import and create model
print("\n[4/5] Testing model initialization...")
try:
    import torch
    from models.camoxpert import CamoXpert

    print("\nCreating model with edgenext_base...")
    print("-" * 70)

    model = CamoXpert(3, 1, pretrained=True, backbone='edgenext_base', num_experts=7)

    print("-" * 70)
    print("✅ Model created successfully!")
    print(f"Feature dimensions used: {model.feature_dims}")

    # Test forward pass
    print("\n[5/5] Testing forward pass...")
    dummy_input = torch.randn(2, 3, 320, 320)
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()

    with torch.no_grad():
        pred, aux_loss, deep = model(dummy_input, return_deep_supervision=True)

    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {pred.shape}")
    print(f"   Aux loss: {aux_loss.item():.6f}")
    print(f"   Deep supervision outputs: {len(deep) if deep else 0}")

    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED - Model should work correctly!")
    print("="*70)

except Exception as e:
    print(f"\n❌ ERROR during model creation/forward pass:")
    print(f"   {type(e).__name__}: {e}")
    print("\n" + "="*70)
    print("TROUBLESHOOTING:")
    print("="*70)
    print("1. Make sure you restarted the Kaggle kernel")
    print("2. Re-clone the repository:")
    print("   !rm -rf /kaggle/working/camoXpert")
    print("   !git clone https://github.com/mahi-chan/camoXpert.git /kaggle/working/camoXpert")
    print("   %cd /kaggle/working/camoXpert")
    print("   !git checkout claude/investigate-gpu-bottleneck-011CUdzKFPf87kvDNa4Za2Y2")
    print("3. Run this verification script again")
    print("="*70)

    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
