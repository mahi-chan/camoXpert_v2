"""
Complete Expert Architectures for Model-Level MoE

Each expert is a complete model (~15M params) with:
- SOTA-inspired COD modules
- Full decoder pathway
- Prediction head

Expert 1: SINet-Style - Search & Identification
Expert 2: PraNet-Style - Reverse Attention
Expert 3: ZoomNet-Style - Multi-Scale Zoom Context
Expert 4: UJSC-Style - Uncertainty-Guided Refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared Decoder Module (used by all experts)
# ============================================================

class DecoderBlock(nn.Module):
    """Standard decoder block with upsampling and skip connections"""
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


class FullDecoder(nn.Module):
    """Complete decoder pathway from features to prediction"""
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()

        # Decoder blocks: progressively upsample
        self.decoder4 = DecoderBlock(feature_dims[3], feature_dims[2], 256)
        self.decoder3 = DecoderBlock(256, feature_dims[1], 128)
        self.decoder2 = DecoderBlock(128, feature_dims[0], 64)
        self.decoder1 = DecoderBlock(64, 0, 32)

        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone
                     f1: [B, 64, 112, 112]
                     f2: [B, 128, 56, 56]
                     f3: [B, 320, 28, 28]
                     f4: [B, 512, 14, 14]
        Returns:
            prediction: [B, 1, 448, 448]
        """
        f1, f2, f3, f4 = features

        # Decode pathway
        d4 = self.decoder4(f4, f3)   # [B, 256, 28, 28]
        d3 = self.decoder3(d4, f2)   # [B, 128, 56, 56]
        d2 = self.decoder2(d3, f1)   # [B, 64, 112, 112]
        d1 = self.decoder1(d2, None) # [B, 32, 224, 224]

        # Final upsample to input resolution
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 32, 448, 448]

        # Prediction
        pred = self.pred_head(d1)  # [B, 1, 448, 448]

        return pred


# ============================================================
# EXPERT 1: SINet-Style (Search & Identification)
# ============================================================

class SINetExpert(nn.Module):
    """
    SINet-Style Expert: Two-stage Search then Identify

    Best for: Hard-to-find objects, cluttered backgrounds
    Strategy:
      1. Search stage: Find candidate regions (global attention)
      2. Identification stage: Refine candidates (local features)

    ~15M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        self.feature_dims = feature_dims

        # ============================================================
        # SEARCH MODULE: Find candidate object regions
        # ============================================================
        self.search_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dims[-1], feature_dims[-1], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dims[-1], feature_dims[-1], 1),
            nn.Sigmoid()
        )

        # ============================================================
        # IDENTIFICATION MODULE: Refine features per scale
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
        # DECODER: Full decoder to prediction
        # ============================================================
        self.decoder = FullDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
        """
        # ============================================================
        # SEARCH STAGE: Generate attention map
        # ============================================================
        search_attention = self.search_module(features[-1])  # [B, 512, 1, 1]

        # ============================================================
        # IDENTIFICATION STAGE: Apply attention and refine
        # ============================================================
        refined_features = []
        for i, (feat, identify) in enumerate(zip(features, self.identify_modules)):
            # Resize search attention to match feature size
            attention = F.interpolate(search_attention, size=feat.shape[2:], mode='bilinear', align_corners=False)

            # Apply attention and refine
            attended = feat * (1 + attention)  # Boost attended regions
            refined = identify(attended)
            refined_features.append(refined + feat)  # Residual connection

        # ============================================================
        # DECODE to final prediction
        # ============================================================
        pred = self.decoder(refined_features)

        return pred


# ============================================================
# EXPERT 2: PraNet-Style (Reverse Attention)
# ============================================================

class PraNetExpert(nn.Module):
    """
    PraNet-Style Expert: Reverse Attention Mechanism

    Best for: Clear foreground-background separation
    Strategy:
      1. Learn what's NOT the object (background)
      2. Foreground = 1 - background
      3. Refine foreground features

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
        # FOREGROUND REFINEMENT: Enhance foreground features
        # ============================================================
        self.foreground_refine_modules = nn.ModuleList([
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
        # DECODER
        # ============================================================
        self.decoder = FullDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
        """
        refined_features = []

        for feat, reverse_attn, fg_refine in zip(
            features,
            self.reverse_attention_modules,
            self.foreground_refine_modules
        ):
            # ============================================================
            # Predict BACKGROUND (reverse attention)
            # ============================================================
            bg_map = reverse_attn(feat)  # [B, 1, H, W]

            # ============================================================
            # Compute FOREGROUND
            # ============================================================
            fg_map = 1 - bg_map

            # ============================================================
            # Enhance foreground features
            # ============================================================
            fg_features = feat * fg_map
            refined = fg_refine(fg_features)
            refined_features.append(refined + feat)  # Residual

        # ============================================================
        # DECODE to final prediction
        # ============================================================
        pred = self.decoder(refined_features)

        return pred


# ============================================================
# EXPERT 3: ZoomNet-Style (Multi-Scale Zoom Context)
# ============================================================

class ZoomNetExpert(nn.Module):
    """
    ZoomNet-Style Expert: Multi-Scale Zoom & Context

    Best for: Multi-scale objects, varying sizes
    Strategy:
      1. Zoom-out: Capture large context (downsample)
      2. Zoom-in: Capture fine details (upsample)
      3. Fuse multi-scale information

    ~15M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        self.feature_dims = feature_dims

        # ============================================================
        # ZOOM-OUT MODULE: Large context
        # ============================================================
        self.zoom_out = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(feature_dims[-1], feature_dims[-1], 3, padding=1),
            nn.BatchNorm2d(feature_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dims[-1], feature_dims[-1], 3, padding=1),
            nn.BatchNorm2d(feature_dims[-1]),
            nn.ReLU(inplace=True)
        )

        # ============================================================
        # ZOOM-IN MODULE: Fine details
        # ============================================================
        self.zoom_in = nn.Sequential(
            nn.Conv2d(feature_dims[-1], feature_dims[-1], 3, padding=1),
            nn.BatchNorm2d(feature_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dims[-1], feature_dims[-1], 3, padding=1),
            nn.BatchNorm2d(feature_dims[-1]),
            nn.ReLU(inplace=True)
        )

        # ============================================================
        # MULTI-SCALE FUSION: Combine zoom levels
        # ============================================================
        self.fusion_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])

        # ============================================================
        # DECODER
        # ============================================================
        self.decoder = FullDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
        """
        # ============================================================
        # Multi-scale context from highest features
        # ============================================================
        highest_feat = features[-1]  # [B, 512, 14, 14]

        # Zoom-out: Capture large context
        zoom_out_feat = self.zoom_out(highest_feat)  # [B, 512, 7, 7]
        zoom_out_feat = F.interpolate(zoom_out_feat, size=highest_feat.shape[2:], mode='bilinear', align_corners=False)

        # Zoom-in: Capture fine details
        zoom_in_feat = self.zoom_in(highest_feat)  # [B, 512, 14, 14]

        # Combine zoom levels
        multi_scale_context = torch.cat([zoom_out_feat, zoom_in_feat], dim=1)  # [B, 1024, 14, 14]

        # ============================================================
        # Fuse with all feature scales
        # ============================================================
        refined_features = []
        for i, (feat, fusion) in enumerate(zip(features, self.fusion_modules)):
            # Resize multi-scale context to match feature size
            context = F.interpolate(multi_scale_context, size=feat.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate and fuse
            fused = torch.cat([feat, context[:, :feat.shape[1]]], dim=1)  # Match channels
            refined = fusion(fused)
            refined_features.append(refined + feat)  # Residual

        # ============================================================
        # DECODE to final prediction
        # ============================================================
        pred = self.decoder(refined_features)

        return pred


# ============================================================
# EXPERT 4: UJSC-Style (Uncertainty-Guided Refinement)
# ============================================================

class UJSCExpert(nn.Module):
    """
    UJSC-Style Expert: Uncertainty-Guided Refinement

    Best for: Ambiguous boundaries, difficult edges
    Strategy:
      1. Predict uncertainty map (where model is uncertain)
      2. Use uncertainty to guide feature refinement
      3. Focus more on uncertain regions

    ~15M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        self.feature_dims = feature_dims

        # ============================================================
        # UNCERTAINTY PREDICTION HEAD
        # ============================================================
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(feature_dims[-1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        # ============================================================
        # UNCERTAINTY-GUIDED REFINEMENT (per scale)
        # ============================================================
        self.refinement_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim + 1, dim, 3, padding=1),  # +1 for uncertainty map
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])

        # ============================================================
        # DECODER
        # ============================================================
        self.decoder = FullDecoder(feature_dims)

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone

        Returns:
            prediction: [B, 1, 448, 448]
        """
        # ============================================================
        # Predict uncertainty map
        # ============================================================
        uncertainty = self.uncertainty_head(features[-1])  # [B, 1, 14, 14]

        # ============================================================
        # Use uncertainty to guide refinement at each scale
        # ============================================================
        refined_features = []
        for feat, refine in zip(features, self.refinement_modules):
            # Resize uncertainty to match feature size
            unc_resized = F.interpolate(uncertainty, size=feat.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate feature with uncertainty
            feat_with_unc = torch.cat([feat, unc_resized], dim=1)  # [B, dim+1, H, W]

            # Refine features (focus more on uncertain regions)
            refined = refine(feat_with_unc)
            refined_features.append(refined + feat)  # Residual

        # ============================================================
        # DECODE to final prediction
        # ============================================================
        pred = self.decoder(refined_features)

        return pred


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing Expert Architectures...")
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

        pred = expert(features)
        print(f"  Output shape: {pred.shape}")
        assert pred.shape == (2, 1, 448, 448), f"Wrong output shape: {pred.shape}"

    print("\n" + "="*60)
    print("✓ All expert tests passed!")
