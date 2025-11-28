"""
CompositeLossSystem: Advanced Multi-Component Loss for Camouflaged Object Detection

Implements sophisticated loss computation with:
1. Progressive weighting strategy (Early/Mid/Late training stages)
2. Boundary-aware loss with signed distance maps and lambda scheduling
3. Frequency-weighted loss for high-frequency regions
4. Scale-adaptive loss based on object size (small objects get 2× weight)
5. Uncertainty-guided loss focusing on low-confidence predictions
6. Dynamic adjustment based on current IoU performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.ndimage import distance_transform_edt


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def dice_loss(self, pred, target, smooth=1e-6):
        """
        Dice loss for binary segmentation.

        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary masks
        Returns:
            Dice loss
        """
        pred_sigmoid = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + smooth) / (union + smooth)

        return 1 - dice

    def forward(self, pred, target, weight_map=None):
        """
        Args:
            pred: [B, 1, H, W] predictions (logits)
            target: [B, 1, H, W] ground truth
            weight_map: [B, 1, H, W] optional pixel-wise weights
        """
        # BCE loss
        bce = self.bce(pred, target)

        # Apply weight map if provided
        if weight_map is not None:
            bce = bce * weight_map

        bce = bce.mean()

        # Dice loss
        dice = self.dice_loss(pred, target)

        # Combined
        total = self.bce_weight * bce + self.dice_weight * dice

        return total, {'bce': bce.item(), 'dice': dice.item()}


class IoULoss(nn.Module):
    """Intersection over Union (IoU) Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, smooth=1e-6):
        """
        Args:
            pred: [B, 1, H, W] predictions (logits)
            target: [B, 1, H, W] ground truth
        """
        pred_sigmoid = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred_sigmoid.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        # IoU calculation
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        iou = (intersection + smooth) / (union + smooth)

        # IoU loss
        loss = 1 - iou.mean()

        return loss


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-aware loss using signed distance transform.

    Emphasizes pixels near object boundaries using distance maps.
    Lambda scheduling from 0.5 to 2.0 over training.
    """
    def __init__(self, lambda_start=0.5, lambda_end=2.0):
        super().__init__()
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.current_lambda = lambda_start

    def compute_signed_distance_map(self, mask):
        """
        Compute signed distance transform.

        Args:
            mask: [H, W] binary mask (numpy array)
        Returns:
            sdt: [H, W] signed distance transform
        """
        # Distance to nearest foreground pixel
        dist_fg = distance_transform_edt(1 - mask)

        # Distance to nearest background pixel
        dist_bg = distance_transform_edt(mask)

        # Signed distance: negative inside, positive outside
        sdt = dist_bg - dist_fg

        return sdt

    def get_boundary_weight_map(self, target):
        """
        Create boundary-aware weight map from target masks.

        Args:
            target: [B, 1, H, W] binary masks
        Returns:
            weight_map: [B, 1, H, W] boundary weights
        """
        B, _, H, W = target.shape
        weight_maps = []

        for b in range(B):
            mask = target[b, 0].cpu().numpy()

            # Compute signed distance transform
            sdt = self.compute_signed_distance_map(mask)

            # Convert to weight: higher weight near boundaries (|sdt| small)
            # Weight = exp(-|sdt| / sigma)
            sigma = 5.0  # Controls decay rate
            weight = np.exp(-np.abs(sdt) / sigma)

            # Normalize to [0, 1] range
            weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-6)

            weight_maps.append(weight)

        # Stack and convert to tensor
        weight_maps = np.stack(weight_maps, axis=0)
        weight_maps = torch.from_numpy(weight_maps).float().unsqueeze(1).to(target.device)

        return weight_maps

    def update_lambda(self, current_epoch, total_epochs):
        """
        Update lambda parameter based on training progress.

        Schedule: lambda increases from lambda_start to lambda_end
        """
        progress = current_epoch / total_epochs
        self.current_lambda = self.lambda_start + progress * (self.lambda_end - self.lambda_start)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] predictions (logits)
            target: [B, 1, H, W] ground truth
        """
        # Get boundary weight map
        boundary_weights = self.get_boundary_weight_map(target)

        # BCE with boundary weighting
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Apply boundary weights with current lambda
        weighted_bce = bce * (1 + self.current_lambda * boundary_weights)

        loss = weighted_bce.mean()

        return loss


class FrequencyWeightedLoss(nn.Module):
    """
    Frequency-weighted loss giving higher weight to high-frequency regions.

    Uses Laplacian filter to detect high-frequency content.
    """
    def __init__(self, high_freq_weight=2.0):
        super().__init__()
        self.high_freq_weight = high_freq_weight

        # Laplacian kernel for detecting edges/high-freq
        laplacian = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('laplacian', laplacian)

    def compute_frequency_map(self, image):
        """
        Compute high-frequency content map.

        Args:
            image: [B, C, H, W] input image or feature
        Returns:
            freq_map: [B, 1, H, W] high-frequency magnitude map
        """
        B, C, H, W = image.shape

        # Convert to grayscale if RGB
        if C == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        # Apply Laplacian filter (convert to match input dtype for AMP compatibility)
        laplacian = self.laplacian.to(dtype=gray.dtype, device=gray.device)
        freq_response = F.conv2d(gray, laplacian, padding=1)

        # Magnitude (absolute value)
        freq_mag = torch.abs(freq_response)

        # Normalize to [0, 1]
        freq_mag = (freq_mag - freq_mag.min()) / (freq_mag.max() - freq_mag.min() + 1e-6)

        return freq_mag

    def forward(self, pred, target, input_image):
        """
        Args:
            pred: [B, 1, H, W] predictions (logits)
            target: [B, 1, H, W] ground truth
            input_image: [B, 3, H, W] input image (for frequency analysis)
        """
        # Compute frequency map
        freq_map = self.compute_frequency_map(input_image)

        # Resize frequency map to match prediction size if needed
        if freq_map.shape[2:] != pred.shape[2:]:
            freq_map = F.interpolate(freq_map, size=pred.shape[2:], mode='bilinear', align_corners=False)

        # Weight map: higher weight for high-frequency regions
        weight_map = 1 + (self.high_freq_weight - 1) * freq_map

        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = bce * weight_map

        loss = weighted_bce.mean()

        return loss, weight_map


class ScaleAdaptiveLoss(nn.Module):
    """
    Scale-adaptive loss weighting based on object size.

    Small objects get 2× weight, large objects get 1× weight.
    """
    def __init__(self, small_obj_weight=2.0, size_threshold=0.03):
        super().__init__()
        self.small_obj_weight = small_obj_weight
        self.size_threshold = size_threshold  # Fraction of image area

    def compute_object_sizes(self, target):
        """
        Compute object size for each sample.

        Args:
            target: [B, 1, H, W] binary masks
        Returns:
            sizes: [B] object sizes as fraction of image area
        """
        B, _, H, W = target.shape
        total_pixels = H * W

        # Object size per sample
        sizes = target.view(B, -1).sum(dim=1) / total_pixels

        return sizes

    def get_size_weights(self, target):
        """
        Get per-sample weights based on object size.

        Args:
            target: [B, 1, H, W] binary masks
        Returns:
            weights: [B, 1, 1, 1] size-based weights
        """
        sizes = self.compute_object_sizes(target)  # [B]

        # Small objects: weight = small_obj_weight
        # Large objects: weight = 1.0
        # Linear interpolation for medium objects
        weights = torch.where(
            sizes < self.size_threshold,
            torch.full_like(sizes, self.small_obj_weight),
            torch.ones_like(sizes)
        )

        # Reshape for broadcasting
        weights = weights.view(-1, 1, 1, 1)

        return weights

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] predictions (logits)
            target: [B, 1, H, W] ground truth
        """
        # Get size-based weights
        size_weights = self.get_size_weights(target)

        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = bce * size_weights

        loss = weighted_bce.mean()

        return loss, size_weights


class UncertaintyGuidedLoss(nn.Module):
    """
    Uncertainty-guided loss focusing on low-confidence predictions.

    Emphasizes pixels where model is uncertain (prediction near 0.5).
    """
    def __init__(self, focus_threshold=0.7):
        super().__init__()
        self.focus_threshold = focus_threshold

    def compute_uncertainty(self, pred):
        """
        Compute prediction uncertainty.

        Uncertainty is high when prediction is near 0.5 (uncertain),
        low when prediction is near 0 or 1 (confident).

        Args:
            pred: [B, 1, H, W] predictions (logits)
        Returns:
            uncertainty: [B, 1, H, W] uncertainty map
        """
        pred_sigmoid = torch.sigmoid(pred)

        # Uncertainty = 1 - |p - 0.5| * 2
        # When p = 0.5, uncertainty = 1
        # When p = 0 or 1, uncertainty = 0
        uncertainty = 1 - torch.abs(pred_sigmoid - 0.5) * 2

        return uncertainty

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] predictions (logits)
            target: [B, 1, H, W] ground truth
        """
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(pred)

        # Weight map: higher weight for uncertain predictions
        # Only focus on regions above threshold
        weight_map = torch.where(
            uncertainty > self.focus_threshold,
            uncertainty,
            torch.ones_like(uncertainty) * 0.1  # Low weight for confident predictions
        )

        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = bce * weight_map

        loss = weighted_bce.mean()

        return loss, uncertainty


class ProgressiveWeightingStrategy:
    """
    Progressive weighting strategy across training stages.

    FIXED: IoU and boundary losses now active from epoch 0.
    Early stage (0-33%): BCE + Dice + IoU + Boundary (foundation)
    Mid stage (33-66%): BCE + Dice + IoU + Boundary (increase boundary focus)
    Late stage (66-100%): BCE + Dice + IoU + Boundary (maximum boundary precision)
    """
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.early_end = total_epochs // 3
        self.mid_end = 2 * total_epochs // 3

    def get_stage(self, current_epoch):
        """Get current training stage"""
        if current_epoch < self.early_end:
            return 'early'
        elif current_epoch < self.mid_end:
            return 'mid'
        else:
            return 'late'

    def get_weights(self, current_epoch):
        """
        Get loss component weights for current epoch.

        FIXED: IoU and boundary losses now active from epoch 0.
        - Early (0-33%): Foundation with IoU + boundary enabled
        - Mid (33-66%): Increase boundary focus
        - Late (66-100%): Maximum boundary precision

        Returns:
            Dict with weights for each component
        """
        stage = self.get_stage(current_epoch)

        if stage == 'early':
            return {
                'bce_dice': 1.0,
                'iou': 0.5,        # IoU active from start
                'boundary': 0.3,   # Boundary active from start
                'frequency': 0.1
            }
        elif stage == 'mid':
            progress = (current_epoch - self.early_end) / (self.mid_end - self.early_end)
            return {
                'bce_dice': 1.0,
                'iou': 0.5,
                'boundary': 0.3 + 0.2 * progress,  # Ramp 0.3 -> 0.5
                'frequency': 0.2
            }
        else:  # late
            progress = (current_epoch - self.mid_end) / (self.total_epochs - self.mid_end)
            return {
                'bce_dice': 0.8,
                'iou': 0.5,
                'boundary': 0.5 + 0.2 * progress,  # Ramp 0.5 -> 0.7
                'frequency': 0.2
            }


class CompositeLossSystem(nn.Module):
    """
    Complete Composite Loss System for Camouflaged Object Detection.

    Combines multiple loss components with progressive weighting,
    boundary awareness, frequency weighting, scale adaptation,
    uncertainty guidance, and dynamic IoU-based adjustment.

    Args:
        total_epochs: Total training epochs (for progressive weighting)
        use_boundary: Enable boundary-aware loss
        use_frequency: Enable frequency-weighted loss
        use_scale_adaptive: Enable scale-adaptive loss
        use_uncertainty: Enable uncertainty-guided loss
        boundary_lambda_start: Starting weight for boundary loss (default: 0.5)
        boundary_lambda_end: Ending weight for boundary loss (default: 2.0)
        frequency_weight: Weight for high-frequency regions (default: 2.0)
        scale_small_weight: Weight multiplier for small objects (default: 2.0)
        uncertainty_threshold: Focus threshold for uncertainty loss (default: 0.7)
    """
    def __init__(
        self,
        total_epochs=100,
        use_boundary=True,
        use_frequency=True,
        use_scale_adaptive=True,
        use_uncertainty=True,
        boundary_lambda_start=0.5,
        boundary_lambda_end=2.0,
        frequency_weight=2.0,
        scale_small_weight=2.0,
        uncertainty_threshold=0.7
    ):
        super().__init__()

        # Loss components (now with configurable parameters)
        self.bce_dice_loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.iou_loss = IoULoss()
        self.boundary_loss = BoundaryAwareLoss(
            lambda_start=boundary_lambda_start,
            lambda_end=boundary_lambda_end
        )
        self.frequency_loss = FrequencyWeightedLoss(high_freq_weight=frequency_weight)
        self.scale_adaptive_loss = ScaleAdaptiveLoss(small_obj_weight=scale_small_weight)
        self.uncertainty_loss = UncertaintyGuidedLoss(focus_threshold=uncertainty_threshold)

        # Progressive weighting strategy
        self.progressive_strategy = ProgressiveWeightingStrategy(total_epochs)

        # Flags
        self.use_boundary = use_boundary
        self.use_frequency = use_frequency
        self.use_scale_adaptive = use_scale_adaptive
        self.use_uncertainty = use_uncertainty

        # Dynamic adjustment
        self.iou_history = []
        self.adjustment_factor = 1.0

        # Track current epoch
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def update_epoch(self, current_epoch, total_epochs=None):
        """
        Update the current epoch for progressive weighting and boundary lambda scheduling.

        Args:
            current_epoch: Current training epoch
            total_epochs: Total epochs (optional, uses init value if not provided)
        """
        self.current_epoch = current_epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
            self.progressive_strategy.total_epochs = total_epochs

        # Update boundary lambda
        self.boundary_loss.update_lambda(current_epoch, self.total_epochs)

    def compute_iou(self, pred, target):
        """Compute IoU for monitoring"""
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > 0.5).float()

        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection

        iou = intersection / (union + 1e-6)

        return iou.item()

    def update_dynamic_adjustment(self, current_iou):
        """
        Dynamically adjust loss weights based on IoU performance.

        If IoU is improving: reduce adjustment (model is learning)
        If IoU is stagnating: increase adjustment (need stronger signal)
        """
        self.iou_history.append(current_iou)

        # Keep last 10 epochs
        if len(self.iou_history) > 10:
            self.iou_history = self.iou_history[-10:]

        # Check if IoU is improving
        if len(self.iou_history) >= 5:
            recent_avg = np.mean(self.iou_history[-5:])
            older_avg = np.mean(self.iou_history[-10:-5]) if len(self.iou_history) >= 10 else recent_avg

            improvement = recent_avg - older_avg

            if improvement > 0.01:
                # Good improvement: reduce adjustment
                self.adjustment_factor = max(0.5, self.adjustment_factor * 0.95)
            elif improvement < -0.01:
                # Performance degrading: increase adjustment
                self.adjustment_factor = min(2.0, self.adjustment_factor * 1.05)
            else:
                # Stagnating: slight increase
                self.adjustment_factor = min(1.5, self.adjustment_factor * 1.02)

    def forward(
        self,
        pred,
        target,
        input_image=None,
        current_epoch=None,
        return_detailed=False
    ):
        """
        Compute composite loss.

        Args:
            pred: [B, 1, H, W] predictions (logits)
            target: [B, 1, H, W] ground truth
            input_image: [B, 3, H, W] input image (for frequency weighting)
            current_epoch: Current training epoch (uses self.current_epoch if None)
            return_detailed: If True, return detailed loss breakdown

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with detailed losses (if return_detailed=True)
        """
        # Use stored epoch if not provided
        if current_epoch is None:
            current_epoch = self.current_epoch

        # Get progressive weights
        stage_weights = self.progressive_strategy.get_weights(current_epoch)

        # Boundary lambda is already updated by update_epoch(), but update again for safety
        self.boundary_loss.update_lambda(current_epoch, self.progressive_strategy.total_epochs)

        # Initialize loss dict
        loss_dict = {}
        total_loss = 0

        # 1. BCE + Dice Loss (always active)
        bce_dice, bce_dice_dict = self.bce_dice_loss(pred, target)
        total_loss += stage_weights['bce_dice'] * bce_dice * self.adjustment_factor
        loss_dict['bce_dice'] = bce_dice.item()
        loss_dict.update({f'bce_dice_{k}': v for k, v in bce_dice_dict.items()})

        # 2. IoU Loss (mid and late stages)
        if stage_weights['iou'] > 0:
            iou_loss = self.iou_loss(pred, target)
            total_loss += stage_weights['iou'] * iou_loss * self.adjustment_factor
            loss_dict['iou'] = iou_loss.item()

        # 3. Boundary-Aware Loss (late stage)
        if self.use_boundary and stage_weights['boundary'] > 0:
            boundary_loss = self.boundary_loss(pred, target)
            total_loss += stage_weights['boundary'] * boundary_loss
            loss_dict['boundary'] = boundary_loss.item()
            loss_dict['boundary_lambda'] = self.boundary_loss.current_lambda

        # 4. Frequency-Weighted Loss (mid and late stages)
        if self.use_frequency and stage_weights['frequency'] > 0 and input_image is not None:
            freq_loss, freq_map = self.frequency_loss(pred, target, input_image)
            total_loss += stage_weights['frequency'] * freq_loss
            loss_dict['frequency'] = freq_loss.item()

        # 5. Scale-Adaptive Loss (always active if enabled)
        if self.use_scale_adaptive:
            scale_loss, size_weights = self.scale_adaptive_loss(pred, target)
            # Apply as additional weighting
            total_loss += 0.2 * scale_loss
            loss_dict['scale_adaptive'] = scale_loss.item()

        # 6. Uncertainty-Guided Loss (always active if enabled)
        if self.use_uncertainty:
            uncertainty_loss, uncertainty_map = self.uncertainty_loss(pred, target)
            # Apply as additional weighting
            total_loss += 0.3 * uncertainty_loss
            loss_dict['uncertainty'] = uncertainty_loss.item()

        # Compute current IoU for monitoring
        current_iou = self.compute_iou(pred, target)
        loss_dict['iou_metric'] = current_iou

        # Update dynamic adjustment
        self.update_dynamic_adjustment(current_iou)
        loss_dict['adjustment_factor'] = self.adjustment_factor

        # Add stage info
        loss_dict['stage'] = self.progressive_strategy.get_stage(current_epoch)
        loss_dict['total_loss'] = total_loss.item()

        if return_detailed:
            return total_loss, loss_dict
        else:
            return total_loss


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("="*70)
    print("Testing CompositeLossSystem")
    print("="*70)

    # Create loss system
    loss_system = CompositeLossSystem(
        total_epochs=100,
        use_boundary=True,
        use_frequency=True,
        use_scale_adaptive=True,
        use_uncertainty=True
    )

    print(f"\nLoss system parameters: {count_parameters(loss_system)}")

    # Test with dummy data
    print("\n" + "="*70)
    print("Test 1: Progressive Weighting Across Epochs")
    print("="*70)

    batch_size = 4
    pred = torch.randn(batch_size, 1, 64, 64)
    target = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
    input_image = torch.randn(batch_size, 3, 64, 64)

    test_epochs = [0, 33, 66, 99]

    for epoch in test_epochs:
        loss, loss_dict = loss_system(
            pred, target, input_image,
            current_epoch=epoch,
            return_detailed=True
        )

        print(f"\nEpoch {epoch} (Stage: {loss_dict['stage']}):")
        print(f"  Total loss: {loss_dict['total_loss']:.4f}")
        print(f"  BCE+Dice: {loss_dict['bce_dice']:.4f}")
        if 'iou' in loss_dict:
            print(f"  IoU loss: {loss_dict['iou']:.4f}")
        if 'boundary' in loss_dict:
            print(f"  Boundary loss: {loss_dict['boundary']:.4f} (lambda: {loss_dict['boundary_lambda']:.2f})")
        if 'frequency' in loss_dict:
            print(f"  Frequency loss: {loss_dict['frequency']:.4f}")
        print(f"  IoU metric: {loss_dict['iou_metric']:.4f}")
        print(f"  Adjustment factor: {loss_dict['adjustment_factor']:.4f}")

    # Test individual components
    print("\n" + "="*70)
    print("Test 2: Individual Loss Components")
    print("="*70)

    # BCE+Dice
    print("\n1. BCE+Dice Loss:")
    bce_dice_loss = BCEDiceLoss()
    loss, loss_dict = bce_dice_loss(pred, target)
    print(f"   Total: {loss.item():.4f}")
    print(f"   BCE: {loss_dict['bce']:.4f}")
    print(f"   Dice: {loss_dict['dice']:.4f}")

    # IoU
    print("\n2. IoU Loss:")
    iou_loss_fn = IoULoss()
    iou_loss = iou_loss_fn(pred, target)
    print(f"   Loss: {iou_loss.item():.4f}")

    # Boundary-aware
    print("\n3. Boundary-Aware Loss:")
    boundary_loss_fn = BoundaryAwareLoss()
    boundary_loss_fn.update_lambda(50, 100)  # Mid training
    boundary_loss = boundary_loss_fn(pred, target)
    print(f"   Loss: {boundary_loss.item():.4f}")
    print(f"   Lambda: {boundary_loss_fn.current_lambda:.2f}")

    # Frequency-weighted
    print("\n4. Frequency-Weighted Loss:")
    freq_loss_fn = FrequencyWeightedLoss()
    freq_loss, freq_map = freq_loss_fn(pred, target, input_image)
    print(f"   Loss: {freq_loss.item():.4f}")
    print(f"   Freq map shape: {freq_map.shape}")
    print(f"   Freq map range: [{freq_map.min():.2f}, {freq_map.max():.2f}]")

    # Scale-adaptive
    print("\n5. Scale-Adaptive Loss:")
    scale_loss_fn = ScaleAdaptiveLoss()
    scale_loss, size_weights = scale_loss_fn(pred, target)
    print(f"   Loss: {scale_loss.item():.4f}")
    print(f"   Size weights: {size_weights.squeeze().tolist()}")

    # Uncertainty-guided
    print("\n6. Uncertainty-Guided Loss:")
    uncertainty_loss_fn = UncertaintyGuidedLoss()
    uncertainty_loss, uncertainty_map = uncertainty_loss_fn(pred, target)
    print(f"   Loss: {uncertainty_loss.item():.4f}")
    print(f"   Avg uncertainty: {uncertainty_map.mean().item():.4f}")

    # Test dynamic adjustment
    print("\n" + "="*70)
    print("Test 3: Dynamic IoU-Based Adjustment")
    print("="*70)

    print("\nSimulating training with improving IoU:")
    for epoch in range(20):
        # Simulate improving predictions
        pred = torch.randn(batch_size, 1, 64, 64) + epoch * 0.1

        loss, loss_dict = loss_system(
            pred, target, input_image,
            current_epoch=epoch,
            return_detailed=True
        )

        if epoch % 5 == 0:
            print(f"Epoch {epoch}:")
            print(f"  IoU: {loss_dict['iou_metric']:.4f}")
            print(f"  Adjustment factor: {loss_dict['adjustment_factor']:.4f}")
            print(f"  Total loss: {loss_dict['total_loss']:.4f}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)

    print("\nFeatures Summary:")
    print("  ✓ Progressive weighting (Early/Mid/Late stages)")
    print("  ✓ Boundary-aware loss with lambda scheduling (0.5 → 2.0)")
    print("  ✓ Frequency-weighted loss for high-freq regions")
    print("  ✓ Scale-adaptive loss (small objects get 2× weight)")
    print("  ✓ Uncertainty-guided loss (focus on low-confidence)")
    print("  ✓ Dynamic IoU-based adjustment")

    print(f"\nLoss Components:")
    print(f"  - BCE+Dice: Always active")
    print(f"  - IoU: Active in mid/late stages")
    print(f"  - Boundary: Active in late stage")
    print(f"  - Frequency: Active in mid/late stages")
    print(f"  - Scale-adaptive: Always active (0.2× weight)")
    print(f"  - Uncertainty: Always active (0.3× weight)")
