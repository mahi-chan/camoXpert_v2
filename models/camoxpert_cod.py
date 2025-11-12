"""
CamoXpert COD - 100% Camouflaged Object Detection Specialized Architecture
Based on SOTA COD research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CamoXpertCOD(nn.Module):
    """
    100% COD-Specialized Architecture
    Integrates all COD-specific modules for maximum performance
    """
    def __init__(self, in_channels=3, num_classes=1, pretrained=True, backbone='edgenext_base'):
        super().__init__()

        print("\n" + "="*70)
        print("INITIALIZING CAMOXPERT COD - 100% COD-SPECIALIZED")
        print("="*70)

        # Import COD modules
        from models.cod_modules import (
            SearchIdentificationModule,
            ReverseAttentionModule,
            ContrastEnhancementModule,
            BoundaryUncertaintyModule,
            IterativeBoundaryRefinement,
            CODTextureExpert,
            CODFrequencyExpert,
            CODEdgeExpert
        )

        # 1. Backbone
        print("\n[1/7] Loading backbone...")
        self.backbone = self._create_backbone(backbone, pretrained)
        self.feature_dims = [80, 160, 288, 584]  # EdgeNeXt feature dimensions
        print(f"✓ Backbone loaded: {backbone}")
        print(f"✓ Feature dimensions: {self.feature_dims}")

        # 2. COD-Specialized Expert Layers (replace MoE with dedicated COD experts)
        print("\n[2/7] Initializing COD-Specialized Experts...")
        self.texture_experts = nn.ModuleList([
            CODTextureExpert(dim) for dim in self.feature_dims
        ])
        self.frequency_experts = nn.ModuleList([
            CODFrequencyExpert(dim) for dim in self.feature_dims
        ])
        self.edge_experts = nn.ModuleList([
            CODEdgeExpert(dim) for dim in self.feature_dims
        ])
        print(f"✓ Created 3 expert types × 4 stages = 12 expert modules")

        # 3. Contrast Enhancement (replaces generic contrast)
        print("\n[3/7] Initializing Contrast Enhancement...")
        self.contrast_modules = nn.ModuleList([
            ContrastEnhancementModule(dim) for dim in self.feature_dims
        ])
        print(f"✓ Created {len(self.contrast_modules)} contrast enhancement modules")

        # 4. Search & Identification Module
        print("\n[4/7] Initializing Search & Identification Module...")
        self.search_module = SearchIdentificationModule(self.feature_dims[-1])
        print("✓ Search module created")

        # 5. Decoder
        print("\n[5/7] Initializing Decoder...")
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(self.feature_dims[3], self.feature_dims[2]),  # 584->288
            DecoderBlock(self.feature_dims[2] * 2, self.feature_dims[1]),  # 288*2->160 (with skip)
            DecoderBlock(self.feature_dims[1] * 2, self.feature_dims[0]),  # 160*2->80
            DecoderBlock(self.feature_dims[0] * 2, 64),  # 80*2->64
        ])
        print(f"✓ Created {len(self.decoder_blocks)} decoder blocks")

        # 6. Reverse Attention Module
        print("\n[6/7] Initializing Reverse Attention Module...")
        self.reverse_attention = ReverseAttentionModule(64)
        print("✓ Reverse attention module created")

        # 7. Boundary Uncertainty & Iterative Refinement
        print("\n[7/7] Initializing Boundary Uncertainty & Refinement...")
        self.boundary_uncertainty = BoundaryUncertaintyModule(64)
        self.iterative_refinement = IterativeBoundaryRefinement(64, num_iterations=2)
        print("✓ Boundary uncertainty & iterative refinement created")

        # Deep supervision heads
        self.deep_heads = nn.ModuleList([
            nn.Conv2d(self.feature_dims[2], num_classes, kernel_size=1),
            nn.Conv2d(self.feature_dims[1], num_classes, kernel_size=1),
            nn.Conv2d(self.feature_dims[0], num_classes, kernel_size=1),
        ])

        # Summary
        print("\n" + "="*70)
        print("CAMOXPERT COD ARCHITECTURE SUMMARY")
        print("="*70)
        print(f"  Backbone:              {backbone}")
        print(f"  COD Experts:           Texture, Frequency, Edge (×4 stages)")
        print(f"  Search Module:         Enabled")
        print(f"  Reverse Attention:     Enabled")
        print(f"  Boundary Uncertainty:  Enabled")
        print(f"  Iterative Refinement:  2 iterations")
        print(f"  Contrast Enhancement:  Enabled")
        print(f"  Deep Supervision:      3 levels")
        print(f"  Specialization:        100% COD-Optimized")
        print("="*70 + "\n")

    def _create_backbone(self, backbone: str, pretrained: bool):
        """Create EdgeNeXt backbone"""
        import timm
        try:
            model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
            return model
        except Exception as e:
            print(f"⚠️  Error loading '{backbone}': {e}")
            print("Attempting fallback...")
            fallback_map = {
                'edgenext_small': 'edgenext_small',
                'edgenext_base': 'edgenext_base',
                'edgenext_base_usi': 'edgenext_base',
            }
            fallback_name = fallback_map.get(backbone, 'edgenext_small')
            model = timm.create_model(fallback_name, pretrained=pretrained, features_only=True)
            print(f"✓ Fallback successful: {fallback_name}")
            return model

    def forward(self, x, return_deep_supervision=False):
        """
        Forward pass through CamoXpertCOD

        Args:
            x: Input images [B, 3, H, W]
            return_deep_supervision: Whether to return deep supervision outputs

        Returns:
            If return_deep_supervision=False:
                pred: Final prediction logits [B, 1, H, W]
                uncertainty: Uncertainty map [B, 1, H, W]
                auxiliary_outputs: Dict with fg_map, search_map, refinements
            If return_deep_supervision=True:
                Same as above plus deep_outputs: List of intermediate predictions
        """
        B, _, H, W = x.shape

        # 1. Backbone feature extraction
        features = self.backbone(x)  # 4 scales

        # 2. Apply COD experts to all feature scales
        enhanced_features = []
        for i, feat in enumerate(features):
            # Apply texture expert
            feat = self.texture_experts[i](feat)
            # Apply frequency expert
            feat = self.frequency_experts[i](feat)
            # Apply edge expert
            feat = self.edge_experts[i](feat)
            # Apply contrast enhancement
            feat = self.contrast_modules[i](feat)
            enhanced_features.append(feat)

        # 3. Search & Identification on highest-level features
        searched_features, search_map = self.search_module(enhanced_features[-1])

        # 4. Decoder with skip connections
        x4 = searched_features  # 584 channels
        x3 = self.decoder_blocks[0](x4, None)  # 288
        x2 = self.decoder_blocks[1](torch.cat([x3, enhanced_features[2]], dim=1), None)  # 160
        x1 = self.decoder_blocks[2](torch.cat([x2, enhanced_features[1]], dim=1), None)  # 80
        x0 = self.decoder_blocks[3](torch.cat([x1, enhanced_features[0]], dim=1), None)  # 64

        # 5. Reverse Attention
        x0, fg_map = self.reverse_attention(x0)

        # Upsample to original size
        x0 = F.interpolate(x0, size=(H, W), mode='bilinear', align_corners=False)
        fg_map = F.interpolate(fg_map, size=(H, W), mode='bilinear', align_corners=False)
        search_map = F.interpolate(search_map, size=(H, W), mode='bilinear', align_corners=False)

        # 6. Boundary Uncertainty Estimation
        initial_pred, uncertainty = self.boundary_uncertainty(x0)

        # Clamp to prevent NaN
        initial_pred = torch.clamp(initial_pred, min=-15, max=15)
        uncertainty = torch.clamp(uncertainty, min=1e-8, max=10.0)

        # 7. Iterative Boundary Refinement
        refinements = self.iterative_refinement(x0, initial_pred, uncertainty)

        # Final prediction is the last refinement
        final_pred = refinements[-1]
        final_pred = torch.clamp(final_pred, min=-15, max=15)

        # Prepare auxiliary outputs
        auxiliary_outputs = {
            'fg_map': fg_map,
            'search_map': search_map,
            'refinements': refinements[:-1],  # Intermediate refinements (exclude final)
            'uncertainty': uncertainty
        }

        # Deep supervision
        deep_outputs = None
        if return_deep_supervision:
            deep_outputs = [
                self.deep_heads[0](enhanced_features[2]),  # 288
                self.deep_heads[1](enhanced_features[1]),  # 160
                self.deep_heads[2](enhanced_features[0]),  # 80
            ]
            # Clamp deep outputs
            deep_outputs = [torch.clamp(d, min=-15, max=15) for d in deep_outputs]

        if return_deep_supervision:
            return final_pred, auxiliary_outputs, deep_outputs
        else:
            return final_pred, auxiliary_outputs, None
