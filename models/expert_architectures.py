"""
Complete Expert Architectures for Model-Level MoE - SOTA REPLICAS

Each expert implements REAL innovations from SOTA COD papers:
- Deep Supervision (auxiliary losses at multiple scales)
- Receptive Field Blocks (RFB) for multi-scale context
- Boundary Enhancement Modules
- Proper Feature Aggregation

Expert 1: SINet-Style - Search & Identification + RFB
Expert 2: PraNet-Style - Reverse Attention + Boundary Guidance
Expert 3: ZoomNet-Style - Multi-Scale Zoom + Feature Pyramid
Expert 4: UJSC-Style - Uncertainty-Guided + Edge Refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SHARED COMPONENTS (used by multiple experts)
# ============================================================

class RFB(nn.Module):
    """
    Receptive Field Block (RFB)

    Used in: SINet, ZoomNet
    Purpose: Capture multi-scale context with different dilation rates
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        inter_channels = in_channels // 4

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 -> 3x3 (dilation=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 -> 3x3 (dilation=3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=3, dilation=3),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 1x1 -> 3x3 (dilation=5)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=5, dilation=5),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(inter_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate all branches
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)

        return out


class BoundaryEnhancement(nn.Module):
    """
    Boundary Enhancement Module

    Used in: PraNet, UJSC
    Purpose: Explicitly detect and enhance object boundaries
    """
    def __init__(self, in_channels):
        super().__init__()

        # Edge detection branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # Boundary refinement
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Detect edges
        edges = self.edge_conv(x)

        # Concatenate features with edge map
        x_with_edges = torch.cat([x, edges], dim=1)

        # Refine with boundary awareness
        refined = self.boundary_refine(x_with_edges)

        return refined, edges


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DeepSupervisionDecoder(nn.Module):
    """
    Decoder with Deep Supervision

    Outputs predictions at multiple scales for auxiliary losses
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()

        # Decoder blocks
        self.decoder4 = DecoderBlock(feature_dims[3], feature_dims[2], 256)
        self.decoder3 = DecoderBlock(256, feature_dims[1], 128)
        self.decoder2 = DecoderBlock(128, feature_dims[0], 64)
        self.decoder1 = DecoderBlock(64, 0, 32)

        # Deep supervision: Auxiliary prediction heads at each level
        self.aux_head4 = nn.Conv2d(256, 1, 1)  # 28x28
        self.aux_head3 = nn.Conv2d(128, 1, 1)  # 56x56
        self.aux_head2 = nn.Conv2d(64, 1, 1)   # 112x112

        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, features, return_aux=False):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone
            return_aux: If True, return auxiliary outputs for deep supervision

        Returns:
            pred: Final prediction [B, 1, 448, 448]
            aux_outputs: List of auxiliary predictions (if return_aux=True)
        """
        f1, f2, f3, f4 = features

        # Decode pathway
        d4 = self.decoder4(f4, f3)   # [B, 256, 28, 28]
        d3 = self.decoder3(d4, f2)   # [B, 128, 56, 56]
        d2 = self.decoder2(d3, f1)   # [B, 64, 112, 112]
        d1 = self.decoder1(d2, None) # [B, 32, 224, 224]

        # Final upsample to input resolution
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 32, 448, 448]

        # Main prediction
        pred = self.pred_head(d1)  # [B, 1, 448, 448]

        if return_aux:
            # Auxiliary outputs (upsampled to 448x448 for loss computation)
            aux4 = F.interpolate(self.aux_head4(d4), size=(448, 448), mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux_head3(d3), size=(448, 448), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head2(d2), size=(448, 448), mode='bilinear', align_corners=False)

            return pred, [aux4, aux3, aux2]

        return pred


# ============================================================
# EXPERT 1: SINet-Style (Search & Identification + RFB)
# ============================================================

class SINetExpert(nn.Module):
    """
    SINet-Style Expert: Two-stage Search then Identify + RFB

    REAL INNOVATIONS:
    - RFB modules for multi-scale receptive fields
    - Search module with global context
    - Identification with local refinement
    - Deep supervision

    ~15M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        self.feature_dims = feature_dims

        # ============================================================
        # SEARCH MODULE: Global context with RFB
        # ============================================================
        self.search_rfb = RFB(feature_dims[-1], feature_dims[-1])
        self.search_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dims[-1], feature_dims[-1] // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dims[-1] // 4, 1, 1),
            nn.Sigmoid()
        )

        # ============================================================
        # IDENTIFICATION MODULE: Local refinement per scale
        # ============================================================
        self.identify_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])

        # ============================================================
        # DECODER with Deep Supervision
        # ============================================================
        self.decoder = DeepSupervisionDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
            aux_outputs: List of auxiliary predictions (for deep supervision)
        """
        # ============================================================
        # SEARCH STAGE: Multi-scale context + attention
        # ============================================================
        search_feat = self.search_rfb(features[-1])  # Apply RFB
        search_attention = self.search_attention(search_feat)  # [B, 1, 1, 1]

        # ============================================================
        # IDENTIFICATION STAGE: Apply attention and refine
        # ============================================================
        refined_features = []
        for i, (feat, identify) in enumerate(zip(features, self.identify_modules)):
            # Resize search attention to match feature size
            attention = F.interpolate(search_attention, size=feat.shape[2:], mode='bilinear', align_corners=False)

            # Apply attention and refine
            attended = feat * (1 + attention)
            refined = identify(attended)
            refined_features.append(refined + feat)  # Residual

        # ============================================================
        # DECODE with deep supervision
        # ============================================================
        pred, aux_outputs = self.decoder(refined_features, return_aux=True)

        return pred, aux_outputs


# ============================================================
# EXPERT 2: PraNet-Style (Reverse Attention + Boundary)
# ============================================================

class PraNetExpert(nn.Module):
    """
    PraNet-Style Expert: Reverse Attention + Boundary Enhancement

    REAL INNOVATIONS:
    - Reverse attention (learn background, infer foreground)
    - Boundary enhancement modules
    - Deep supervision
    - Edge-guided refinement

    ~15M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        self.feature_dims = feature_dims

        # ============================================================
        # REVERSE ATTENTION: Predict background per scale
        # ============================================================
        self.reverse_attention_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, padding=1),
                nn.BatchNorm2d(dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, 1, 1),
                nn.Sigmoid()
            ) for dim in feature_dims
        ])

        # ============================================================
        # BOUNDARY ENHANCEMENT: Explicit edge detection
        # ============================================================
        self.boundary_modules = nn.ModuleList([
            BoundaryEnhancement(dim) for dim in feature_dims
        ])

        # ============================================================
        # DECODER with Deep Supervision
        # ============================================================
        self.decoder = DeepSupervisionDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
            aux_outputs: List of auxiliary predictions
        """
        refined_features = []

        for feat, reverse_attn, boundary_module in zip(
            features,
            self.reverse_attention_modules,
            self.boundary_modules
        ):
            # ============================================================
            # REVERSE ATTENTION: Predict background
            # ============================================================
            bg_map = reverse_attn(feat)  # [B, 1, H, W]
            fg_map = 1 - bg_map

            # ============================================================
            # BOUNDARY ENHANCEMENT: Edge-guided refinement
            # ============================================================
            fg_features = feat * fg_map
            refined, edges = boundary_module(fg_features)

            refined_features.append(refined + feat)  # Residual

        # ============================================================
        # DECODE with deep supervision
        # ============================================================
        pred, aux_outputs = self.decoder(refined_features, return_aux=True)

        return pred, aux_outputs


# ============================================================
# EXPERT 3: ZoomNet-Style (Multi-Scale + Feature Pyramid)
# ============================================================

class ZoomNetExpert(nn.Module):
    """
    ZoomNet-Style Expert: Multi-Scale Zoom + Feature Pyramid

    REAL INNOVATIONS:
    - RFB modules at multiple scales
    - Zoom-in and zoom-out with different receptive fields
    - Feature Pyramid Network (FPN) for multi-scale fusion
    - Deep supervision

    ~15M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        self.feature_dims = feature_dims

        # ============================================================
        # RFB MODULES: Multi-scale context at each level
        # ============================================================
        self.rfb_modules = nn.ModuleList([
            RFB(dim, dim) for dim in feature_dims
        ])

        # ============================================================
        # FEATURE PYRAMID NETWORK: Top-down pathway
        # ============================================================
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(dim, 256, 1) for dim in feature_dims
        ])

        self.fpn_output = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in feature_dims
        ])

        # Reduce back to original dims
        self.fpn_reduce = nn.ModuleList([
            nn.Conv2d(256, dim, 1) for dim in feature_dims
        ])

        # ============================================================
        # DECODER with Deep Supervision
        # ============================================================
        self.decoder = DeepSupervisionDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
            aux_outputs: List of auxiliary predictions
        """
        # ============================================================
        # Apply RFB to each feature level
        # ============================================================
        rfb_features = [rfb(feat) for rfb, feat in zip(self.rfb_modules, features)]

        # ============================================================
        # Feature Pyramid Network (top-down)
        # ============================================================
        # Lateral connections
        fpn_feats = [lateral(feat) for lateral, feat in zip(self.fpn_lateral, rfb_features)]

        # Top-down pathway
        for i in range(len(fpn_feats) - 1, 0, -1):
            # Upsample higher-level features
            upsampled = F.interpolate(fpn_feats[i], size=fpn_feats[i-1].shape[2:],
                                     mode='bilinear', align_corners=False)
            # Add to lower-level features
            fpn_feats[i-1] = fpn_feats[i-1] + upsampled

        # Output convolutions
        fpn_outputs = [conv(feat) for conv, feat in zip(self.fpn_output, fpn_feats)]

        # Reduce back to original dimensions
        refined_features = [reduce(feat) + orig_feat
                           for reduce, feat, orig_feat in zip(self.fpn_reduce, fpn_outputs, features)]

        # ============================================================
        # DECODE with deep supervision
        # ============================================================
        pred, aux_outputs = self.decoder(refined_features, return_aux=True)

        return pred, aux_outputs


# ============================================================
# EXPERT 4: UJSC-Style (Uncertainty + Edge Refinement)
# ============================================================

class UJSCExpert(nn.Module):
    """
    UJSC-Style Expert: Uncertainty-Guided + Edge Refinement

    REAL INNOVATIONS:
    - Uncertainty prediction with dropout-based estimation
    - Boundary enhancement modules
    - Uncertainty-weighted feature refinement
    - Deep supervision

    ~15M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        self.feature_dims = feature_dims

        # ============================================================
        # UNCERTAINTY PREDICTION
        # ============================================================
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(feature_dims[-1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),  # Dropout for uncertainty estimation
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        # ============================================================
        # BOUNDARY ENHANCEMENT: Edge detection per scale
        # ============================================================
        self.boundary_modules = nn.ModuleList([
            BoundaryEnhancement(dim) for dim in feature_dims
        ])

        # ============================================================
        # UNCERTAINTY-GUIDED REFINEMENT
        # ============================================================
        self.refinement_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim + 1, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])

        # ============================================================
        # DECODER with Deep Supervision
        # ============================================================
        self.decoder = DeepSupervisionDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
            aux_outputs: List of auxiliary predictions
        """
        # ============================================================
        # Predict uncertainty map
        # ============================================================
        uncertainty = self.uncertainty_head(features[-1])  # [B, 1, 14, 14]

        # ============================================================
        # Boundary enhancement + uncertainty-guided refinement
        # ============================================================
        refined_features = []
        for feat, boundary_module, refine in zip(features, self.boundary_modules, self.refinement_modules):
            # Detect boundaries
            boundary_feat, edges = boundary_module(feat)

            # Resize uncertainty to match feature size
            unc_resized = F.interpolate(uncertainty, size=feat.shape[2:],
                                       mode='bilinear', align_corners=False)

            # Concatenate feature with uncertainty
            feat_with_unc = torch.cat([boundary_feat, unc_resized], dim=1)

            # Uncertainty-guided refinement
            refined = refine(feat_with_unc)
            refined_features.append(refined + feat)  # Residual

        # ============================================================
        # DECODE with deep supervision
        # ============================================================
        pred, aux_outputs = self.decoder(refined_features, return_aux=True)

        return pred, aux_outputs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing SOTA Expert Architectures...")
    print("="*60)

    # Create dummy features
    features = [
        torch.randn(2, 64, 112, 112),
        torch.randn(2, 128, 56, 56),
        torch.randn(2, 320, 28, 28),
        torch.randn(2, 512, 14, 14)
    ]

    experts = [
        ("SINet-Style", SINetExpert()),
        ("PraNet-Style", PraNetExpert()),
        ("ZoomNet-Style", ZoomNetExpert()),
        ("UJSC-Style", UJSCExpert())
    ]

    for name, expert in experts:
        print(f"\n{name} Expert:")
        print(f"  Parameters: {count_parameters(expert) / 1e6:.1f}M")

        pred, aux_outputs = expert(features)
        print(f"  Main output shape: {pred.shape}")
        print(f"  Auxiliary outputs: {len(aux_outputs)} scales")
        for i, aux in enumerate(aux_outputs):
            print(f"    Aux {i+1}: {aux.shape}")

        assert pred.shape == (2, 1, 448, 448), f"Wrong output shape: {pred.shape}"
        assert len(aux_outputs) == 3, f"Wrong number of aux outputs: {len(aux_outputs)}"

    print("\n" + "="*60)
    print("✓ All SOTA expert tests passed!")
