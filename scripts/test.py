import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.camoxpert import CamoXpert
from data.dataset import COD10KDataset
from metrics.cod_metrics import CODMetrics
from models.utils import load_checkpoint, count_parameters


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test dataset and compute metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    metrics = CODMetrics()
    all_metrics = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            predictions, _ = model(images)
            batch_metrics = metrics.compute_all(predictions, masks)
            all_metrics.append(batch_metrics)

    # Aggregate metrics
    aggregated_metrics = {key: sum(d[key] for d in all_metrics) / len(all_metrics) for key in all_metrics[0]}
    return aggregated_metrics


def print_test_results(metrics):
    """Print test results in a formatted way"""
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print("\n📊 ACCURACY METRICS")
    print("-" * 80)
    print(f"  Pixel Accuracy:        {metrics['Pixel_Accuracy']:.4f}  ({metrics['Pixel_Accuracy'] * 100:.2f}%)")
    print(f"  Precision:             {metrics['Precision']:.4f}  ({metrics['Precision'] * 100:.2f}%)")
    print(f"  Recall (Sensitivity):  {metrics['Recall']:.4f}  ({metrics['Recall'] * 100:.2f}%)")
    print(f"  Specificity:           {metrics['Specificity']:.4f}  ({metrics['Specificity'] * 100:.2f}%)")

    print("\n📐 SEGMENTATION METRICS")
    print("-" * 80)
    print(f"  IoU (Jaccard Index):   {metrics['IoU']:.4f}  ({metrics['IoU'] * 100:.2f}%)")
    print(f"  Dice Score (F1):       {metrics['Dice_Score']:.4f}  ({metrics['Dice_Score'] * 100:.2f}%)")
    print(f"  F-measure (β=0.3):     {metrics['F-measure']:.4f}  ({metrics['F-measure'] * 100:.2f}%)")

    print("\n🎯 CAMOUFLAGE-SPECIFIC METRICS")
    print("-" * 80)
    print(f"  S-measure:             {metrics['S-measure']:.4f}  (Structure similarity)")
    print(f"  E-measure:             {metrics['E-measure']:.4f}  (Enhanced alignment)")
    print(f"  MAE:                   {metrics['MAE']:.4f}  (Mean Absolute Error)")

    print("\n" + "=" * 80)

    # Performance summary
    print("\n📈 PERFORMANCE SUMMARY")
    print("-" * 80)
    avg_detection = (metrics['Precision'] + metrics['Recall']) / 2
    avg_segmentation = (metrics['IoU'] + metrics['Dice_Score']) / 2

    print(f"  Average Detection Performance:     {avg_detection:.4f}")
    print(f"  Average Segmentation Performance:  {avg_segmentation:.4f}")
    print(f"  Overall Score:                     {(avg_detection + avg_segmentation) / 2:.4f}")
    print("=" * 80 + "\n")


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 80}")
    print(f"CamoXpert Test Script")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'=' * 80}\n")

    # Load dataset
    print("Loading test dataset...")
    dataset = COD10KDataset(root_dir=args.dataset_path, split="test", img_size=args.img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"✓ Test samples: {len(dataset)}\n")

    # Load model
    print("Loading model...")
    model = CamoXpert(in_channels=3, num_classes=1)
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    print("✓ Model loaded successfully\n")

    # Print model statistics
    total_params, trainable_params = count_parameters(model)
    print("Model Statistics:")
    print(f"  Total Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable_params:,}")
    print(f"  Model Size:            {total_params * 4 / 1024 / 1024:.2f} MB (FP32)\n")

    # Evaluate model
    print("Evaluating model on test set...")
    print("-" * 80)
    metrics = evaluate_model(model, dataloader, device)

    # Print results
    print_test_results(metrics)

    # Save results to file
    import json
    results_file = os.path.join(os.path.dirname(args.checkpoint), 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {results_file}\n")


# Create parser at module level
parser = argparse.ArgumentParser(description="CamoXpert Test Script")
parser.add_argument("--dataset-path", type=str, required=True, help="Path to the test dataset")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size for testing")
parser.add_argument("--img-size", type=int, default=352, help="Image size for testing")
parser.add_argument("--device", type=str, default="cuda", help="Device to run testing on")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)