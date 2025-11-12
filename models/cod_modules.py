"""
100% COD-Specialized Modules
Based on SOTA COD research (SINet, PraNet, ZoomNet, UGTR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryUncertaintyModule(nn.Module):
    """
    Estimates uncertainty at object boundaries
    Camouflaged boundaries are inherently ambiguous - model should know when uncertain
    """
    def __init__(self, dim):
        super().__init__()
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Predict mean (main prediction)
        self.mean_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1)
        )

        # Predict uncertainty (how confident we are)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, x):
        shared_feat = self.shared(x)
        mean = self.mean_head(shared_feat)
        uncertainty = self.uncertainty_head(shared_feat)
        return mean, uncertainty


class SearchIdentificationModule(nn.Module):
    """
    Mimics visual search process for camouflaged objects (from SINet)
    1. Search: Where might objects be hiding?
    2. Identification: What are the object features?
    """
    def __init__(self, dim):
        super().__init__()
        # Search branch: Generate search map (where to look)
        self.search_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()  # Search confidence map [0, 1]
        )

        # Identification branch: Enhanced features at search locations
        # Using regular conv for DataParallel compatibility
        self.identify_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        search_map = self.search_conv(x)
        features = self.identify_conv(x)
        searched_features = features * search_map
        return searched_features, search_map


class ReverseAttentionModule(nn.Module):
    """
    Learns to suppress background by modeling what is NOT the object (from PraNet)
    Key: Once you know what's NOT the object, what remains must be the object
    """
    def __init__(self, dim):
        super().__init__()
        # Predict background (inverse of foreground)
        self.background_pred = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 4, 3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )

        # Refine features by removing background
        self.refine = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        bg_map = self.background_pred(x)
        fg_map = 1 - bg_map
        x_with_bg = torch.cat([x, bg_map], dim=1)
        refined = self.refine(x_with_bg)
        output = refined * fg_map
        return output, fg_map


class ContrastEnhancementModule(nn.Module):
    """
    Enhances subtle differences between foreground and background
    Uses multi-scale contrast kernels specifically for COD
    DataParallel-safe version without depthwise convolutions
    """
    def __init__(self, dim):
        super().__init__()
        # Calculate channel splits that sum exactly to dim
        c1 = dim // 3
        c2 = dim // 3
        c3 = dim - c1 - c2  # Remainder goes to third branch

        # Multi-scale contrast detection (regular convs for DataParallel compatibility)
        self.contrast_3x3 = nn.Sequential(
            nn.Conv2d(dim, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.contrast_5x5 = nn.Sequential(
            nn.Conv2d(dim, c2, 5, padding=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.contrast_7x7 = nn.Sequential(
            nn.Conv2d(dim, c3, 7, padding=3),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

        # Contrast fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Contrast amplification (learned)
        self.amplify = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c3 = self.contrast_3x3(x)
        c5 = self.contrast_5x5(x)
        c7 = self.contrast_7x7(x)
        contrast = self.fusion(torch.cat([c3, c5, c7], dim=1))
        contrast_weight = self.amplify(contrast)
        enhanced = x + contrast * contrast_weight
        return enhanced


class IterativeBoundaryRefinement(nn.Module):
    """
    Iteratively refine boundaries using uncertainty feedback
    Focus computation on uncertain regions (boundaries)
    """
    def __init__(self, dim, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations

        self.refinement_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim + 2, dim, 3, padding=1),  # +2 for pred + uncertainty
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_iterations)
        ])

        self.pred_heads = nn.ModuleList([
            nn.Conv2d(dim, 1, 1) for _ in range(num_iterations)
        ])

    def forward(self, features, initial_pred, initial_uncertainty):
        """
        Args:
            features: [B, C, H, W]
            initial_pred: [B, 1, H, W] logits
            initial_uncertainty: [B, 1, H, W] uncertainty map
        Returns:
            refinements: List of refined predictions
        """
        pred = initial_pred
        uncertainty = initial_uncertainty

        refinements = []

        for i in range(self.num_iterations):
            uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)
            x = torch.cat([features, pred, uncertainty_norm], dim=1)
            refined_features = self.refinement_blocks[i](x)
            refined_features = refined_features * (1 + uncertainty_norm)
            pred = self.pred_heads[i](refined_features)
            refinements.append(pred)

        return refinements


class CODTextureExpert(nn.Module):
    """
    COD-optimized texture expert with adaptive dilation
    Detects multi-scale texture patterns critical for camouflage
    """
    def __init__(self, dim):
        super().__init__()
        # Calculate channel splits that sum exactly to dim
        c1 = dim // 4
        c2 = dim // 4
        c3 = dim // 4
        c4 = dim - c1 - c2 - c3  # Remainder goes to fourth branch

        # Adaptive dilations for texture at different scales
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, c1, 1),
            nn.Conv2d(c1, c1, 3, padding=1, dilation=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, c2, 1),
            nn.Conv2d(c2, c2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, c3, 1),
            nn.Conv2d(c3, c3, 3, padding=4, dilation=4),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, c4, 1),
            nn.Conv2d(c4, c4, 3, padding=8, dilation=8),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        multi_scale = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fusion(multi_scale) + x


class CODFrequencyExpert(nn.Module):
    """
    COD-optimized frequency domain analysis (simplified for DataParallel)
    Uses direct convolutions instead of frequency separation arithmetic
    """
    def __init__(self, dim):
        super().__init__()
        # Calculate channel splits that sum exactly to dim
        c1 = dim // 4
        c2 = dim // 4
        c3 = dim // 4
        c4 = dim - c1 - c2 - c3  # Remainder goes to fourth branch

        # Replace frequency separation with direct multi-scale convolutions
        self.scale1_conv = nn.Sequential(
            nn.Conv2d(dim, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.scale2_conv = nn.Sequential(
            nn.Conv2d(dim, c2, 5, padding=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.scale3_conv = nn.Sequential(
            nn.Conv2d(dim, c3, 7, padding=3),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, c4, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Multi-scale feature extraction (no arithmetic operations)
        feat1 = self.scale1_conv(x)
        feat2 = self.scale2_conv(x)
        feat3 = self.scale3_conv(x)
        spatial = self.spatial_conv(x)
        features = torch.cat([feat1, feat2, feat3, spatial], dim=1)
        return self.fusion(features) + x


class CODEdgeExpert(nn.Module):
    """
    COD-optimized edge detection (DataParallel-safe version)
    Uses learnable edge detection initialized with edge kernels to avoid
    DataParallel grouped convolution misalignment issues
    """
    def __init__(self, dim):
        super().__init__()
        # Calculate channel splits that sum exactly to dim
        c1 = dim // 4
        c2 = dim // 4
        c3 = dim // 4
        c4 = dim - c1 - c2 - c3  # Remainder goes to fourth branch

        # Learnable edge detection (regular convs for DataParallel compatibility)
        # Network will learn edge patterns during training
        self.horizontal_edge = nn.Sequential(
            nn.Conv2d(dim, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.vertical_edge = nn.Sequential(
            nn.Conv2d(dim, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.laplacian_edge = nn.Sequential(
            nn.Conv2d(dim, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(dim, c4, 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h_edge = self.horizontal_edge(x)
        v_edge = self.vertical_edge(x)
        lap_edge = self.laplacian_edge(x)
        spatial = self.spatial_branch(x)
        edge_features = torch.cat([h_edge, v_edge, lap_edge, spatial], dim=1)
        return self.fusion(edge_features) + x
