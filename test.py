"""
CamoXpert Evaluation Script

Evaluates trained models on multiple COD benchmark datasets:
- COD10K (test set)
- CHAMELEON
- CAMO (test set)
- NC4K (test set)

Computes comprehensive metrics:
- S-measure (Structure Measure)
- F-measure (Weighted F-score)
- E-measure (Enhanced-alignment Measure)
- MAE (Mean Absolute Error)
- IoU (Intersection over Union)

Supports Test-Time Augmentation (TTA) for improved performance.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

from models.model_level_moe import ModelLevelMoE
from metrics.cod_metrics import CODMetrics


class CODTestDataset(Dataset):
    """
    Dataset for testing on COD benchmarks.

    Expected directory structure:
    dataset_root/
        Images/
            img1.jpg
            img2.png
            ...
        GT/
            img1.png
            img2.png
            ...
    """

    def __init__(self, root, image_size=352):
        self.root = Path(root)
        self.image_size = image_size

        # Find image and GT directories
        self.image_dir = self.root / 'Images' if (self.root / 'Images').exists() else self.root / 'Imgs'
        self.gt_dir = self.root / 'GT'

        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.gt_dir.exists():
            raise ValueError(f"GT directory not found: {self.gt_dir}")

        # Get image files
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

        # Match GT files
        self.gt_files = []
        for img_file in self.image_files:
            gt_file = self.gt_dir / f"{img_file.stem}.png"
            if not gt_file.exists():
                # Try .jpg extension for GT
                gt_file = self.gt_dir / f"{img_file.stem}.jpg"
            if gt_file.exists():
                self.gt_files.append(gt_file)
            else:
                raise ValueError(f"GT file not found for {img_file.name}")

        print(f"  ✓ Loaded {len(self.image_files)} images from {self.root.name}")

        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_files[idx]).convert('RGB')
        original_size = img.size

        # Load GT
        gt = Image.open(self.gt_files[idx]).convert('L')

        # Apply transforms
        img_tensor = self.img_transform(img)
        gt_tensor = self.gt_transform(gt)

        return {
            'image': img_tensor,
            'gt': gt_tensor,
            'name': self.image_files[idx].stem,
            'original_size': original_size
        }


def load_checkpoint(checkpoint_path, num_experts=4, device='cuda'):
    """
    Load model from checkpoint, handling DDP 'module.' prefix.

    Args:
        checkpoint_path: Path to checkpoint file
        num_experts: Number of experts in the model
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Checkpoint from epoch: {epoch}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Create model
    model = ModelLevelMoE(
        backbone='pvt_v2_b2',
        num_experts=num_experts,
        top_k=2,
        pretrained=False,
        use_deep_supervision=False
    )

    # Load state dict
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"{'='*70}\n")

    return model


def test_time_augmentation(model, image, scales=[0.75, 1.0, 1.25], use_flip=True):
    """
    Apply Test-Time Augmentation (TTA) for improved predictions.

    Args:
        model: Trained model
        image: Input image tensor [1, 3, H, W]
        scales: List of scale factors for multi-scale testing
        use_flip: Whether to use horizontal flip augmentation

    Returns:
        Averaged prediction [1, 1, H, W]
    """
    B, C, H, W = image.shape
    predictions = []

    for scale in scales:
        # Resize image
        if scale != 1.0:
            new_h, new_w = int(H * scale), int(W * scale)
            scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            scaled_img = image

        # Forward pass
        with torch.no_grad():
            pred = model(scaled_img)

            # Resize back to original size
            if scale != 1.0:
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

            predictions.append(pred)

        # Flip augmentation
        if use_flip:
            flipped_img = torch.flip(scaled_img, dims=[3])  # Horizontal flip

            with torch.no_grad():
                pred_flip = model(flipped_img)

                # Resize back
                if scale != 1.0:
                    pred_flip = F.interpolate(pred_flip, size=(H, W), mode='bilinear', align_corners=False)

                # Un-flip prediction
                pred_flip = torch.flip(pred_flip, dims=[3])
                predictions.append(pred_flip)

    # Average all predictions
    final_pred = torch.stack(predictions, dim=0).mean(dim=0)

    return final_pred


def evaluate_dataset(model, dataset, device, use_tta=False, threshold=0.5, output_dir=None):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataset: Test dataset
        device: Device to run evaluation on
        use_tta: Whether to use Test-Time Augmentation
        threshold: Threshold for binary prediction
        output_dir: Optional directory to save prediction masks

    Returns:
        Dictionary of computed metrics
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    metrics = CODMetrics()

    # Create output directory if saving predictions
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Evaluating {dataset.root.name}"):
            image = batch['image'].to(device)
            gt = batch['gt'].to(device)
            name = batch['name'][0]

            # Forward pass
            if use_tta:
                pred = test_time_augmentation(model, image)
            else:
                pred = model(image)

            # Apply sigmoid
            pred = torch.sigmoid(pred)

            # Update metrics
            metrics.update(pred, gt, threshold=threshold)

            # Save prediction if requested
            if output_dir is not None:
                pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pred_img = Image.fromarray(pred_np)
                pred_img.save(output_dir / f"{name}.png")

    # Compute final metrics
    final_metrics = metrics.compute()

    return final_metrics


def save_results_json(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results saved to: {output_path}")


def save_results_markdown(results, output_path, checkpoint_path, use_tta):
    """Save results to Markdown file with formatted tables."""
    with open(output_path, 'w') as f:
        f.write("# CamoXpert Evaluation Results\n\n")
        f.write(f"**Checkpoint**: `{checkpoint_path}`\n\n")
        f.write(f"**Test-Time Augmentation**: {'Enabled' if use_tta else 'Disabled'}\n\n")

        # Main metrics table
        f.write("## Primary Metrics\n\n")
        f.write("| Dataset | S-measure ⭐ | F-measure | E-measure | MAE ↓ | IoU |\n")
        f.write("|---------|-------------|-----------|-----------|-------|-----|\n")

        for dataset_name, metrics in results.items():
            if dataset_name != 'average':
                f.write(f"| {dataset_name:11} | {metrics['S-measure']:.4f} | "
                       f"{metrics['F-measure']:.4f} | {metrics['E-measure']:.4f} | "
                       f"{metrics['MAE']:.4f} | {metrics['IoU']:.4f} |\n")

        # Average row
        if 'average' in results:
            avg = results['average']
            f.write(f"| **Average** | **{avg['S-measure']:.4f}** | "
                   f"**{avg['F-measure']:.4f}** | **{avg['E-measure']:.4f}** | "
                   f"**{avg['MAE']:.4f}** | **{avg['IoU']:.4f}** |\n")

        # Additional metrics table
        f.write("\n## Additional Metrics\n\n")
        f.write("| Dataset | Precision | Recall | Dice Score | Pixel Acc | Specificity |\n")
        f.write("|---------|-----------|--------|------------|-----------|-------------|\n")

        for dataset_name, metrics in results.items():
            if dataset_name != 'average':
                f.write(f"| {dataset_name:11} | {metrics['Precision']:.4f} | "
                       f"{metrics['Recall']:.4f} | {metrics['Dice_Score']:.4f} | "
                       f"{metrics['Pixel_Accuracy']:.4f} | {metrics['Specificity']:.4f} |\n")

        # Average row
        if 'average' in results:
            avg = results['average']
            f.write(f"| **Average** | **{avg['Precision']:.4f}** | "
                   f"**{avg['Recall']:.4f}** | **{avg['Dice_Score']:.4f}** | "
                   f"**{avg['Pixel_Accuracy']:.4f}** | **{avg['Specificity']:.4f}** |\n")

        # Notes
        f.write("\n## Notes\n\n")
        f.write("- ⭐ S-measure is the primary metric for COD evaluation\n")
        f.write("- ↓ indicates lower is better\n")
        f.write("- All other metrics: higher is better\n")

    print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CamoXpert Evaluation')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-experts', type=int, default=4,
                       help='Number of experts in the model (default: 4)')

    # Data arguments
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing test datasets')
    parser.add_argument('--datasets', nargs='+',
                       default=['COD10K', 'CHAMELEON', 'CAMO', 'NC4K'],
                       help='Datasets to evaluate on (default: all)')
    parser.add_argument('--image-size', type=int, default=352,
                       help='Input image size (default: 352)')

    # Evaluation arguments
    parser.add_argument('--tta', action='store_true',
                       help='Enable Test-Time Augmentation (multi-scale + flip)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary prediction (default: 0.5)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save prediction masks to output directory')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Output directory for results (default: ./test_results)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("CAMOEXPERT EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"TTA: {'Enabled' if args.tta else 'Disabled'}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    # Load model
    model = load_checkpoint(args.checkpoint, num_experts=args.num_experts, device=device)

    # Evaluate on each dataset
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        # Find dataset path
        dataset_path = Path(args.data_root) / dataset_name

        # Handle different naming conventions
        if not dataset_path.exists():
            # Try with -v3 suffix (e.g., COD10K-v3)
            dataset_path = Path(args.data_root) / f"{dataset_name}-v3"

        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            print(f"    Skipping {dataset_name}...")
            continue

        # For COD10K, CAMO, NC4K, use Test subdirectory
        if dataset_name in ['COD10K', 'CAMO', 'NC4K']:
            test_path = dataset_path / 'Test'
            if test_path.exists():
                dataset_path = test_path

        try:
            # Create dataset
            dataset = CODTestDataset(dataset_path, image_size=args.image_size)

            # Evaluate
            pred_dir = output_dir / 'predictions' / dataset_name if args.save_predictions else None
            metrics = evaluate_dataset(
                model, dataset, device,
                use_tta=args.tta,
                threshold=args.threshold,
                output_dir=pred_dir
            )

            # Store results
            all_results[dataset_name] = metrics

            # Print results
            print(f"\n  Results for {dataset_name}:")
            print(f"    S-measure: {metrics['S-measure']:.4f} ⭐")
            print(f"    F-measure: {metrics['F-measure']:.4f}")
            print(f"    E-measure: {metrics['E-measure']:.4f}")
            print(f"    MAE:       {metrics['MAE']:.4f}")
            print(f"    IoU:       {metrics['IoU']:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            continue

    # Compute average metrics
    if len(all_results) > 0:
        avg_metrics = {}
        for metric_name in all_results[list(all_results.keys())[0]].keys():
            avg_metrics[metric_name] = np.mean([
                results[metric_name] for results in all_results.values()
            ])
        all_results['average'] = avg_metrics

        print(f"\n{'='*70}")
        print("AVERAGE METRICS")
        print(f"{'='*70}")
        print(f"  S-measure: {avg_metrics['S-measure']:.4f} ⭐")
        print(f"  F-measure: {avg_metrics['F-measure']:.4f}")
        print(f"  E-measure: {avg_metrics['E-measure']:.4f}")
        print(f"  MAE:       {avg_metrics['MAE']:.4f}")
        print(f"  IoU:       {avg_metrics['IoU']:.4f}")
        print(f"{'='*70}\n")

    # Save results
    save_results_json(all_results, output_dir / 'results.json')
    save_results_markdown(all_results, output_dir / 'results.md', args.checkpoint, args.tta)

    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}\n")


if __name__ == '__main__':
    main()
