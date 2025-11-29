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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
from scipy.ndimage import distance_transform_edt

from models.model_level_moe import ModelLevelMoE


class CODMetrics:
    """
    Camouflaged Object Detection Metrics.

    All methods accept numpy arrays [H, W] with values in [0, 1].
    Implements standard COD evaluation metrics from literature.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.total_metrics = {}
        self.num_samples = 0

    def s_measure(self, pred, gt, alpha=0.5):
        """
        Structure Measure (S-measure).

        Evaluates structural similarity between prediction and ground truth.
        Combines object-level and region-level assessments.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            alpha: Balance between object and region scores (default: 0.5)

        Returns:
            S-measure score (higher is better, range [0, 1])
        """
        y = np.mean(gt)

        if y == 0:  # No object in GT
            return 1.0 - np.mean(pred)
        elif y == 1:  # Entire image is object
            return np.mean(pred)
        else:
            # Object-level score
            So = self._s_object(pred, gt)
            # Region-level score
            Sr = self._s_region(pred, gt)
            # Combined score
            return alpha * So + (1 - alpha) * Sr

    def _s_object(self, pred, gt):
        """Compute object-level structure similarity."""
        # Foreground
        pred_fg = pred * gt
        O_fg = self._object_score(pred_fg, gt)

        # Background
        pred_bg = (1 - pred) * (1 - gt)
        O_bg = self._object_score(pred_bg, 1 - gt)

        # Weighted combination
        u = np.mean(gt)
        return u * O_fg + (1 - u) * O_bg

    def _object_score(self, pred, gt):
        """Compute object score."""
        gt_sum = np.sum(gt)
        if gt_sum == 0:
            return 0.0

        pred_mean = np.sum(pred) / (gt_sum + 1e-8)
        sigma = np.sum((pred - pred_mean) ** 2) / (gt_sum + 1e-8)

        return 2.0 * pred_mean / (pred_mean ** 2 + 1.0 + sigma + 1e-8)

    def _s_region(self, pred, gt):
        """Compute region-level structure similarity."""
        # Find centroid
        X, Y = self._centroid(gt)

        # Divide into 4 regions
        pred1 = pred[:Y, :X]
        pred2 = pred[:Y, X:]
        pred3 = pred[Y:, :X]
        pred4 = pred[Y:, X:]

        gt1 = gt[:Y, :X]
        gt2 = gt[:Y, X:]
        gt3 = gt[Y:, :X]
        gt4 = gt[Y:, X:]

        # Compute SSIM for each region
        Q1 = self._ssim(pred1, gt1)
        Q2 = self._ssim(pred2, gt2)
        Q3 = self._ssim(pred3, gt3)
        Q4 = self._ssim(pred4, gt4)

        # Compute weights
        H, W = gt.shape
        w1 = X * Y / (H * W + 1e-8)
        w2 = (W - X) * Y / (H * W + 1e-8)
        w3 = X * (H - Y) / (H * W + 1e-8)
        w4 = (W - X) * (H - Y) / (H * W + 1e-8)

        # Weighted combination
        return w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    def _centroid(self, gt):
        """Compute centroid of ground truth mask."""
        H, W = gt.shape
        rows = np.arange(H)
        cols = np.arange(W)

        total = np.sum(gt) + 1e-8

        # Column centroid
        X = int(np.sum(np.sum(gt, axis=0) * cols) / total)
        # Row centroid
        Y = int(np.sum(np.sum(gt, axis=1) * rows) / total)

        # Clamp to valid range
        X = max(1, min(X, W - 1))
        Y = max(1, min(Y, H - 1))

        return X, Y

    def _ssim(self, pred, gt):
        """Compute structural similarity (SSIM)."""
        H, W = pred.shape

        if H < 2 or W < 2:
            return 0.0

        N = H * W

        # Means
        x = np.mean(pred)
        y = np.mean(gt)

        # Variances and covariance
        sigma_x2 = np.sum((pred - x) ** 2) / (N - 1 + 1e-8)
        sigma_y2 = np.sum((gt - y) ** 2) / (N - 1 + 1e-8)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1 + 1e-8)

        # SSIM formula
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            return alpha / (beta + 1e-8)
        elif beta == 0:
            return 1.0
        else:
            return 0.0

    def f_measure(self, pred, gt, threshold=0.5, beta2=0.3):
        """
        F-measure (weighted F-score).

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            threshold: Binary threshold (default: 0.5)
            beta2: Beta squared for F-beta score (default: 0.3)

        Returns:
            F-measure score (higher is better, range [0, 1])
        """
        # Binarize
        pred_bin = (pred > threshold).astype(np.float32)
        gt_bin = (gt > threshold).astype(np.float32)

        # True positives, false positives, false negatives
        tp = np.sum(pred_bin * gt_bin)
        fp = np.sum(pred_bin * (1 - gt_bin))
        fn = np.sum((1 - pred_bin) * gt_bin)

        # Precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # F-measure
        f_score = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

        return f_score

    def e_measure(self, pred, gt):
        """
        Enhanced-alignment Measure (E-measure).

        Evaluates pixel-level and image-level alignment.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]

        Returns:
            E-measure score (higher is better, range [0, 1])
        """
        if np.sum(gt) == 0:  # No object
            return 1.0 - np.mean(pred)

        # Enhanced alignment matrix
        enhanced = self._enhanced_alignment_matrix(pred, gt)

        return np.mean(enhanced)

    def _enhanced_alignment_matrix(self, pred, gt):
        """Compute enhanced alignment matrix."""
        # Binarize GT for foreground/background
        gt_fg = (gt > 0.5).astype(np.float32)
        gt_bg = 1 - gt_fg

        # Compute alignment
        pred_mean = np.mean(pred)

        alignment = np.zeros_like(pred)

        # Foreground alignment
        if np.sum(gt_fg) > 0:
            alignment += ((pred - pred_mean) ** 2) * gt_fg / (np.sum(gt_fg) + 1e-8)

        # Background alignment
        if np.sum(gt_bg) > 0:
            alignment += ((pred - pred_mean) ** 2) * gt_bg / (np.sum(gt_bg) + 1e-8)

        # Enhanced matrix
        enhanced = 2 * alignment

        # Normalize to [0, 1]
        enhanced = 1.0 / (1.0 + enhanced)

        return enhanced

    def mae(self, pred, gt):
        """
        Mean Absolute Error.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]

        Returns:
            MAE score (lower is better, range [0, 1])
        """
        return np.mean(np.abs(pred - gt))

    def iou(self, pred, gt, threshold=0.5):
        """
        Intersection over Union (IoU).

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            threshold: Binary threshold (default: 0.5)

        Returns:
            IoU score (higher is better, range [0, 1])
        """
        # Binarize
        pred_bin = (pred > threshold).astype(np.float32)
        gt_bin = (gt > threshold).astype(np.float32)

        # Intersection and union
        intersection = np.sum(pred_bin * gt_bin)
        union = np.sum(pred_bin) + np.sum(gt_bin) - intersection

        return intersection / (union + 1e-8)

    def weighted_f_measure(self, pred, gt, threshold=0.5):
        """
        Distance-weighted F-measure.

        Weights errors based on distance to object boundary.
        Errors near boundaries are penalized more heavily.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            threshold: Binary threshold (default: 0.5)

        Returns:
            Weighted F-measure score (higher is better, range [0, 1])
        """
        from scipy.ndimage import distance_transform_edt

        # Binarize
        pred_bin = (pred > threshold).astype(np.uint8)
        gt_bin = (gt > threshold).astype(np.uint8)

        # Compute distance transforms
        # Distance to nearest boundary
        dt_gt = distance_transform_edt(gt_bin) + distance_transform_edt(1 - gt_bin)

        # Inverse distance weighting (closer to boundary = higher weight)
        weights = 1.0 / (dt_gt + 1.0)
        weights = weights / np.max(weights)  # Normalize

        # Weighted true positives, false positives, false negatives
        tp = np.sum(weights * pred_bin * gt_bin)
        fp = np.sum(weights * pred_bin * (1 - gt_bin))
        fn = np.sum(weights * (1 - pred_bin) * gt_bin)

        # Weighted precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # Weighted F-measure
        f_score = 2 * precision * recall / (precision + recall + 1e-8)

        return f_score

    def update(self, pred, gt, threshold=0.5):
        """
        Update metrics with a new sample.

        Args:
            pred: Prediction tensor or array [B, 1, H, W] or [H, W]
            gt: Ground truth tensor or array [B, 1, H, W] or [H, W]
            threshold: Binary threshold
        """
        # Convert to numpy if tensor
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        # Squeeze to [H, W]
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)

        # Compute all metrics
        metrics = {
            'S-measure': self.s_measure(pred, gt),
            'F-measure': self.f_measure(pred, gt, threshold),
            'E-measure': self.e_measure(pred, gt),
            'MAE': self.mae(pred, gt),
            'IoU': self.iou(pred, gt, threshold),
            'Weighted-F': self.weighted_f_measure(pred, gt, threshold)
        }

        # Accumulate
        if self.num_samples == 0:
            self.total_metrics = metrics
        else:
            for k, v in metrics.items():
                self.total_metrics[k] += v

        self.num_samples += 1

    def compute(self):
        """
        Compute average metrics.

        Returns:
            Dictionary of averaged metrics
        """
        if self.num_samples == 0:
            return {}

        return {k: v / self.num_samples for k, v in self.total_metrics.items()}


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


class Visualizer:
    """
    Visualization utilities for COD predictions.

    Creates various visualization outputs:
    - Overlay with TP/FP/FN (Green/Red/Blue)
    - Boundary comparison
    - Error heatmap
    - Segmented output (object on white background)
    - Cutout with transparent background (RGBA)
    - Comprehensive comparison figure
    """

    def __init__(self):
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    def denormalize_image(self, img_tensor):
        """
        Denormalize image tensor back to [0, 255] RGB.

        Args:
            img_tensor: [C, H, W] normalized tensor

        Returns:
            numpy array [H, W, 3] in uint8 format
        """
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        img = img_tensor.cpu().numpy()
        img = img * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

        return img

    def create_overlay(self, image, pred, gt, threshold=0.5):
        """
        Create TP/FP/FN overlay visualization.

        Green: True Positives
        Red: False Positives
        Blue: False Negatives

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image with overlay
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize predictions and GT
        pred_bin = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        gt_bin = (gt.squeeze().cpu().numpy() > threshold).astype(np.uint8)

        # Compute TP, FP, FN
        tp = (pred_bin == 1) & (gt_bin == 1)
        fp = (pred_bin == 1) & (gt_bin == 0)
        fn = (pred_bin == 0) & (gt_bin == 1)

        # Create overlay
        overlay = img.copy()

        # Green for TP
        overlay[tp] = (overlay[tp] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        # Red for FP
        overlay[fp] = (overlay[fp] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        # Blue for FN
        overlay[fn] = (overlay[fn] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

        return Image.fromarray(overlay)

    def create_boundary_overlay(self, image, pred, gt, threshold=0.5):
        """
        Create boundary comparison overlay.

        GT boundary: Green
        Pred boundary: Red

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image with boundary overlay
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize
        pred_bin = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
        gt_bin = (gt.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

        # Extract boundaries using Canny edge detection
        pred_boundary = cv2.Canny(pred_bin, 50, 150)
        gt_boundary = cv2.Canny(gt_bin, 50, 150)

        # Create overlay
        overlay = img.copy()

        # Green for GT boundary
        overlay[gt_boundary > 0] = [0, 255, 0]

        # Red for Pred boundary
        overlay[pred_boundary > 0] = [255, 0, 0]

        return Image.fromarray(overlay)

    def create_error_map(self, pred, gt):
        """
        Create error heatmap showing prediction errors.

        Args:
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)

        Returns:
            PIL Image with error heatmap
        """
        # Compute absolute error
        error = torch.abs(pred - gt).squeeze().cpu().numpy()

        # Create heatmap using matplotlib colormap
        cmap = plt.cm.jet
        error_colored = (cmap(error)[:, :, :3] * 255).astype(np.uint8)

        return Image.fromarray(error_colored)

    def create_segmented_output(self, image, pred, threshold=0.5):
        """
        Create segmented output: predicted object on white background.

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image with object on white background
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize prediction
        mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)

        # Create white background
        output = np.ones_like(img) * 255

        # Paste object
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        output = np.where(mask_3ch, img, output).astype(np.uint8)

        return Image.fromarray(output)

    def create_cutout(self, image, pred, threshold=0.5):
        """
        Create cutout with transparent background (RGBA).

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image in RGBA mode with transparent background
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize prediction
        mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

        # Create RGBA image
        rgba = np.dstack([img, mask])

        return Image.fromarray(rgba, mode='RGBA')

    def create_comparison_figure(self, image, pred, gt, metrics, name, threshold=0.5):
        """
        Create comprehensive 12-panel comparison figure.

        Layout:
        Row 1: Original | GT | Prediction | Overlay (TP/FP/FN)
        Row 2: Boundary Overlay | Error Map | Segmented | Cutout
        Row 3: Pred Heatmap | GT Binary | Pred Binary | Metrics

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            metrics: Dictionary of computed metrics
            name: Sample name
            threshold: Binary threshold

        Returns:
            matplotlib Figure
        """
        # Create figure with GridSpec for better layout
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Denormalize image
        img = self.denormalize_image(image)
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()

        # Row 1: Original | GT | Prediction | Overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt_np, cmap='gray')
        ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred_np, cmap='gray')
        ax3.set_title('Prediction', fontsize=14, fontweight='bold')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        overlay = np.array(self.create_overlay(image, pred, gt, threshold))
        ax4.imshow(overlay)
        ax4.set_title('TP/FP/FN Overlay', fontsize=14, fontweight='bold')
        ax4.axis('off')
        # Add legend
        green_patch = mpatches.Patch(color='green', label='True Positive')
        red_patch = mpatches.Patch(color='red', label='False Positive')
        blue_patch = mpatches.Patch(color='blue', label='False Negative')
        ax4.legend(handles=[green_patch, red_patch, blue_patch], loc='upper right', fontsize=10)

        # Row 2: Boundary | Error Map | Segmented | Cutout
        ax5 = fig.add_subplot(gs[1, 0])
        boundary = np.array(self.create_boundary_overlay(image, pred, gt, threshold))
        ax5.imshow(boundary)
        ax5.set_title('Boundary Comparison', fontsize=14, fontweight='bold')
        ax5.axis('off')
        # Add legend
        green_line = mpatches.Patch(color='green', label='GT Boundary')
        red_line = mpatches.Patch(color='red', label='Pred Boundary')
        ax5.legend(handles=[green_line, red_line], loc='upper right', fontsize=10)

        ax6 = fig.add_subplot(gs[1, 1])
        error_map = np.array(self.create_error_map(pred, gt))
        im = ax6.imshow(error_map)
        ax6.set_title('Error Heatmap', fontsize=14, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

        ax7 = fig.add_subplot(gs[1, 2])
        segmented = np.array(self.create_segmented_output(image, pred, threshold))
        ax7.imshow(segmented)
        ax7.set_title('Segmented Output', fontsize=14, fontweight='bold')
        ax7.axis('off')

        ax8 = fig.add_subplot(gs[1, 3])
        cutout = np.array(self.create_cutout(image, pred, threshold))
        # Create checkered background for transparency visualization
        checker = np.indices((cutout.shape[0], cutout.shape[1])).sum(axis=0) % 20 < 10
        checker_bg = np.ones((cutout.shape[0], cutout.shape[1], 3)) * 200
        checker_bg[checker] = 150
        # Blend with alpha
        alpha = cutout[:, :, 3:4] / 255.0
        blended = cutout[:, :, :3] * alpha + checker_bg * (1 - alpha)
        ax8.imshow(blended.astype(np.uint8))
        ax8.set_title('Cutout (RGBA)', fontsize=14, fontweight='bold')
        ax8.axis('off')

        # Row 3: Heatmaps and Metrics
        ax9 = fig.add_subplot(gs[2, 0])
        im = ax9.imshow(pred_np, cmap='hot')
        ax9.set_title('Prediction Heatmap', fontsize=14, fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)

        ax10 = fig.add_subplot(gs[2, 1])
        pred_bin = (pred_np > threshold).astype(float)
        ax10.imshow(pred_bin, cmap='gray')
        ax10.set_title('Prediction Binary', fontsize=14, fontweight='bold')
        ax10.axis('off')

        ax11 = fig.add_subplot(gs[2, 2])
        gt_bin = (gt_np > threshold).astype(float)
        ax11.imshow(gt_bin, cmap='gray')
        ax11.set_title('GT Binary', fontsize=14, fontweight='bold')
        ax11.axis('off')

        # Metrics panel
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')

        metrics_text = f"""
        Sample: {name}

        Primary Metrics:
        ━━━━━━━━━━━━━━━━━━━━
        S-measure: {metrics['S-measure']:.4f} ⭐
        F-measure: {metrics['F-measure']:.4f}
        E-measure: {metrics['E-measure']:.4f}
        MAE:       {metrics['MAE']:.4f}
        IoU:       {metrics['IoU']:.4f}

        Accuracy Metrics:
        ━━━━━━━━━━━━━━━━━━━━
        Precision: {metrics['Precision']:.4f}
        Recall:    {metrics['Recall']:.4f}
        Dice:      {metrics['Dice_Score']:.4f}
        Pixel Acc: {metrics['Pixel_Accuracy']:.4f}
        Specific:  {metrics['Specificity']:.4f}
        """

        ax12.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                 verticalalignment='center', transform=ax12.transAxes)

        # Overall title
        fig.suptitle(f'CamoXpert Evaluation: {name}', fontsize=18, fontweight='bold', y=0.98)

        plt.tight_layout()

        return fig

    def save_individual_outputs(self, image, pred, gt, name, save_dir, threshold=0.5):
        """
        Save all individual visualizations to files.

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            name: Sample name
            save_dir: Directory to save outputs
            threshold: Binary threshold

        Returns:
            Dictionary mapping output type to file path
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Original image
        img = self.denormalize_image(image)
        img_pil = Image.fromarray(img)
        img_path = save_dir / f"{name}_original.png"
        img_pil.save(img_path)
        outputs['original'] = img_path

        # Ground truth
        gt_np = (gt.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt_pil = Image.fromarray(gt_np, mode='L')
        gt_path = save_dir / f"{name}_gt.png"
        gt_pil.save(gt_path)
        outputs['gt'] = gt_path

        # Prediction
        pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_np, mode='L')
        pred_path = save_dir / f"{name}_pred.png"
        pred_pil.save(pred_path)
        outputs['pred'] = pred_path

        # Overlay
        overlay_path = save_dir / f"{name}_overlay.png"
        self.create_overlay(image, pred, gt, threshold).save(overlay_path)
        outputs['overlay'] = overlay_path

        # Boundary overlay
        boundary_path = save_dir / f"{name}_boundary.png"
        self.create_boundary_overlay(image, pred, gt, threshold).save(boundary_path)
        outputs['boundary'] = boundary_path

        # Error map
        error_path = save_dir / f"{name}_error.png"
        self.create_error_map(pred, gt).save(error_path)
        outputs['error'] = error_path

        # Segmented output
        segmented_path = save_dir / f"{name}_segmented.png"
        self.create_segmented_output(image, pred, threshold).save(segmented_path)
        outputs['segmented'] = segmented_path

        # Cutout (RGBA)
        cutout_path = save_dir / f"{name}_cutout.png"
        self.create_cutout(image, pred, threshold).save(cutout_path)
        outputs['cutout'] = cutout_path

        return outputs


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


def evaluate_dataset(model, dataset, device, use_tta=False, threshold=0.5, output_dir=None,
                     save_visualizations=False, num_vis_samples=None):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataset: Test dataset
        device: Device to run evaluation on
        use_tta: Whether to use Test-Time Augmentation
        threshold: Threshold for binary prediction
        output_dir: Optional directory to save prediction masks
        save_visualizations: Whether to save visualizations
        num_vis_samples: Number of samples to visualize (None = all)

    Returns:
        Dictionary of computed metrics
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    metrics = CODMetrics()

    # Create output directory if saving predictions
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer if needed
    visualizer = Visualizer() if save_visualizations else None
    vis_count = 0

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

            # Compute metrics for this sample
            sample_metrics = CODMetrics()
            sample_metrics.update(pred, gt, threshold=threshold)
            sample_metrics_dict = sample_metrics.compute()

            # Update overall metrics
            metrics.update(pred, gt, threshold=threshold)

            # Save prediction if requested
            if output_dir is not None:
                pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pred_img = Image.fromarray(pred_np)
                pred_img.save(output_dir / f"{name}.png")

            # Save visualizations if requested
            if save_visualizations and (num_vis_samples is None or vis_count < num_vis_samples):
                vis_dir = output_dir.parent / 'visualizations' / dataset.root.name if output_dir else Path('visualizations') / dataset.root.name

                # Save comparison figure
                fig = visualizer.create_comparison_figure(
                    image.squeeze(0), pred, gt, sample_metrics_dict, name, threshold
                )
                fig_path = vis_dir / 'figures'
                fig_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path / f"{name}_comparison.png", dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Save individual outputs
                individual_dir = vis_dir / 'individual' / name
                visualizer.save_individual_outputs(
                    image.squeeze(0), pred, gt, name, individual_dir, threshold
                )

                vis_count += 1

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
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save comprehensive visualizations (overlays, boundaries, error maps, etc.)')
    parser.add_argument('--num-vis-samples', type=int, default=None,
                       help='Number of samples to visualize per dataset (default: all)')

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
    print(f"Visualizations: {'Enabled' if args.save_visualizations else 'Disabled'}")
    if args.save_visualizations:
        print(f"  Vis samples per dataset: {args.num_vis_samples if args.num_vis_samples else 'All'}")
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
                output_dir=pred_dir,
                save_visualizations=args.save_visualizations,
                num_vis_samples=args.num_vis_samples
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
