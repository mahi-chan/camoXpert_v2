"""
Boundary-Aware Loss Functions for Enhanced MoE

Components:
1. BoundaryAwareLoss - Weights segmentation errors higher at boundaries
2. BoundaryPredictionLoss - Supervises boundary prediction
3. DiscontinuitySupervisionLoss - Supervises TDD and GAD outputs
4. PerExpertLoss - Individual supervision for each expert
5. CombinedEnhancedLoss - Combines all losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareLoss(nn.Module):
    """
    Segmentation loss that penalizes boundary errors more heavily.
    """
    def __init__(self, boundary_weight=5.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def _extract_boundary(self, mask, kernel_size=3):
        """Extract boundary from binary mask using morphological gradient"""
        padding = kernel_size // 2

        # Dilate
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        # Erode
        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
        # Boundary = Dilate - Erode
        boundary = dilated - eroded

        # Expand boundary region slightly for more coverage
        boundary = F.max_pool2d(boundary, 5, stride=1, padding=2)

        return boundary

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted segmentation [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
        """
        # Extract boundary from GT
        boundary = self._extract_boundary(target)

        # Per-pixel BCE loss (autocast-safe)
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(pred.float(), target.float(), reduction='none')

        # Weight by boundary proximity
        weight_map = 1.0 + self.boundary_weight * boundary

        # Weighted loss
        weighted_loss = (bce * weight_map).mean()

        return weighted_loss


class DiceLoss(nn.Module):
    """Standard Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BoundaryPredictionLoss(nn.Module):
    """
    Loss for training the Boundary Prior Network.
    Uses BCE + Dice to handle thin boundary class imbalance.
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def _extract_boundary(self, mask, kernel_size=3):
        """Extract GT boundary from segmentation mask"""
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
        boundary = dilated - eroded
        return boundary

    def forward(self, pred_boundary, target_mask):
        """
        Args:
            pred_boundary: Predicted boundary [B, 1, H, W]
            target_mask: GT segmentation mask [B, 1, H, W]
        """
        # Resize if needed
        if pred_boundary.shape[2:] != target_mask.shape[2:]:
            target_mask = F.interpolate(
                target_mask,
                size=pred_boundary.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Extract GT boundary
        target_boundary = self._extract_boundary(target_mask)
        target_boundary = torch.clamp(target_boundary, 0, 1)

        # Positive weight for imbalanced boundary pixels
        pos_ratio = target_boundary.sum() / (target_boundary.numel() + 1e-6)
        pos_weight = torch.clamp((1 - pos_ratio) / (pos_ratio + 1e-6), 1, 50)

        # BCE with positive weighting (autocast-safe)
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(pred_boundary.float(), target_boundary.float(), reduction='none')
        weighted_bce = bce * (1 + (pos_weight - 1) * target_boundary)
        bce_loss = weighted_bce.mean()

        # Dice loss
        dice_loss = self.dice(pred_boundary, target_boundary)

        return bce_loss + dice_loss


class DiscontinuitySupervisionLoss(nn.Module):
    """
    Supervises TDD and GAD outputs.
    Target: Discontinuity should be high at object boundaries.
    """
    def forward(self, discontinuity_map, target_mask):
        """
        Args:
            discontinuity_map: TDD or GAD output [B, 1, H, W]
            target_mask: GT segmentation mask [B, 1, H, W]
        """
        # Resize mask to discontinuity resolution
        if discontinuity_map.shape[2:] != target_mask.shape[2:]:
            target_mask = F.interpolate(
                target_mask,
                size=discontinuity_map.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Extract boundary as supervision target
        padding = 1
        dilated = F.max_pool2d(target_mask, 3, stride=1, padding=padding)
        eroded = -F.max_pool2d(-target_mask, 3, stride=1, padding=padding)
        target_boundary = dilated - eroded

        # Expand boundary region
        target_boundary = F.max_pool2d(target_boundary, 5, stride=1, padding=2)
        target_boundary = torch.clamp(target_boundary, 0, 1)

        # BCE loss (autocast-safe)
        with torch.amp.autocast('cuda', enabled=False):
            loss = F.binary_cross_entropy(discontinuity_map.float(), target_boundary.float())

        return loss


class PerExpertLoss(nn.Module):
    """
    Computes loss for each expert individually.
    Ensures all experts learn to segment, not just relied-upon ones.
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, expert_preds, target):
        """
        Args:
            expert_preds: List of expert predictions [pred1, pred2, pred3]
            target: GT mask [B, 1, H, W]
        """
        total_loss = 0

        for pred in expert_preds:
            # Resize if needed
            if pred.shape[2:] != target.shape[2:]:
                pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

            pred = torch.sigmoid(pred) if pred.min() < 0 else pred  # Apply sigmoid if logits

            # BCE loss (autocast-safe)
            with torch.amp.autocast('cuda', enabled=False):
                bce = F.binary_cross_entropy(pred.float(), target.float())
            dice = self.dice(pred, target)

            total_loss += (bce + dice)

        # Average over experts
        return total_loss / len(expert_preds)


class HardSampleMiningLoss(nn.Module):
    """Focus on hardest pixels"""
    def __init__(self, hard_ratio=0.3):
        super().__init__()
        self.hard_ratio = hard_ratio

    def forward(self, pred, target):
        # Per-pixel loss (autocast-safe)
        with torch.amp.autocast('cuda', enabled=False):
            pixel_loss = F.binary_cross_entropy(pred.float(), target.float(), reduction='none')

        B, C, H, W = pixel_loss.shape
        k = max(int(self.hard_ratio * H * W), 100)

        # Get hardest pixels
        pixel_loss_flat = pixel_loss.view(B, -1)
        hard_loss, _ = torch.topk(pixel_loss_flat, k, dim=1)

        return pixel_loss.mean() + hard_loss.mean()


class CombinedEnhancedLoss(nn.Module):
    """
    Combined loss for enhanced MoE with TDD/GAD/BPN.

    Components:
    - Segmentation: Boundary-aware BCE + Dice
    - Boundary: BPN supervision
    - Discontinuity: TDD + GAD supervision
    - Per-expert: Individual expert losses
    - Hard mining: Focus on difficult pixels
    - Load balance: Router regularization
    """
    def __init__(
        self,
        seg_weight=1.0,
        boundary_weight=2.0,
        discontinuity_weight=0.3,
        expert_weight=0.3,
        hard_mining_weight=0.5,
        load_balance_weight=0.1
    ):
        super().__init__()

        self.seg_weight = seg_weight
        self.boundary_weight = boundary_weight
        self.discontinuity_weight = discontinuity_weight
        self.expert_weight = expert_weight
        self.hard_mining_weight = hard_mining_weight
        self.load_balance_weight = load_balance_weight

        # Loss components
        self.boundary_aware = BoundaryAwareLoss(boundary_weight=5.0)
        self.dice = DiceLoss()
        self.boundary_pred = BoundaryPredictionLoss()
        self.discontinuity = DiscontinuitySupervisionLoss()
        self.per_expert = PerExpertLoss()
        self.hard_mining = HardSampleMiningLoss(hard_ratio=0.3)

        print("\n" + "="*70)
        print("COMBINED ENHANCED LOSS")
        print("="*70)
        print(f"  Segmentation (boundary-aware): {seg_weight}")
        print(f"  Boundary prediction: {boundary_weight}")
        print(f"  Discontinuity supervision: {discontinuity_weight}")
        print(f"  Per-expert supervision: {expert_weight}")
        print(f"  Hard sample mining: {hard_mining_weight}")
        print(f"  Load balance: {load_balance_weight}")
        print("="*70)

    def forward(self, pred, target, aux_outputs=None):
        """
        Args:
            pred: Main prediction [B, 1, H, W]
            target: GT mask [B, 1, H, W]
            aux_outputs: Dict with auxiliary outputs from model

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        loss_dict = {}

        # Ensure pred is sigmoid activated
        if pred.min() < 0:
            pred = torch.sigmoid(pred)

        # 1. Main segmentation loss (boundary-aware)
        seg_loss = self.boundary_aware(pred, target)
        dice_loss = self.dice(pred, target)
        loss_dict['seg_bce'] = seg_loss
        loss_dict['seg_dice'] = dice_loss

        # 2. Hard sample mining
        hard_loss = self.hard_mining(pred, target)
        loss_dict['hard_mining'] = hard_loss

        total_loss = (
            self.seg_weight * (seg_loss + dice_loss) +
            self.hard_mining_weight * hard_loss
        )

        # Additional losses from auxiliary outputs
        if aux_outputs is not None:
            # 3. Boundary prediction loss
            if 'boundary' in aux_outputs and aux_outputs['boundary'] is not None:
                boundary_loss = self.boundary_pred(aux_outputs['boundary'], target)
                loss_dict['boundary'] = boundary_loss
                total_loss += self.boundary_weight * boundary_loss

            # 4. Discontinuity supervision (TDD)
            if 'texture_disc' in aux_outputs and aux_outputs['texture_disc'] is not None:
                tdd_loss = self.discontinuity(aux_outputs['texture_disc'], target)
                loss_dict['tdd'] = tdd_loss
                total_loss += self.discontinuity_weight * tdd_loss

            # 5. Discontinuity supervision (GAD)
            if 'gradient_anomaly' in aux_outputs and aux_outputs['gradient_anomaly'] is not None:
                gad_loss = self.discontinuity(aux_outputs['gradient_anomaly'], target)
                loss_dict['gad'] = gad_loss
                total_loss += self.discontinuity_weight * gad_loss

            # 6. Per-expert supervision
            if 'individual_expert_preds' in aux_outputs and aux_outputs['individual_expert_preds'] is not None:
                expert_loss = self.per_expert(aux_outputs['individual_expert_preds'], target)
                loss_dict['expert'] = expert_loss
                total_loss += self.expert_weight * expert_loss

            # 7. Load balance loss from router
            if 'load_balance_loss' in aux_outputs and aux_outputs['load_balance_loss'] is not None:
                lb_loss = aux_outputs['load_balance_loss']
                loss_dict['load_balance'] = lb_loss
                total_loss += self.load_balance_weight * lb_loss

        loss_dict['total'] = total_loss

        return total_loss, loss_dict


# Test
if __name__ == '__main__':
    print("Testing CombinedEnhancedLoss...")

    criterion = CombinedEnhancedLoss()

    # Create dummy data
    pred = torch.sigmoid(torch.randn(2, 1, 416, 416))
    target = (torch.rand(2, 1, 416, 416) > 0.7).float()

    aux_outputs = {
        'boundary': torch.sigmoid(torch.randn(2, 1, 104, 104)),
        'texture_disc': torch.sigmoid(torch.randn(2, 1, 104, 104)),
        'gradient_anomaly': torch.sigmoid(torch.randn(2, 1, 104, 104)),
        'individual_expert_preds': [
            torch.sigmoid(torch.randn(2, 1, 416, 416)),
            torch.sigmoid(torch.randn(2, 1, 416, 416)),
            torch.sigmoid(torch.randn(2, 1, 416, 416)),
        ],
        'load_balance_loss': torch.tensor(0.1),
    }

    total_loss, loss_dict = criterion(pred, target, aux_outputs)

    print("\nLoss components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")

    # Verify gradients
    total_loss.backward()
    print("\n✓ Gradient flow OK")
    print("✓ Loss test passed!")
