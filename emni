"""
CamoXpert: Advanced Camouflaged Object Detection Model

Architecture:
- EdgeNeXt backbone for feature extraction
- SDTA (Selective Dual-axis Temporal Attention) enhancement
- Mixture of Experts (MoE) for specialized feature processing
- Progressive decoder with skip connections
- Deep supervision for better training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class CamoXpert(nn.Module):
    """
    CamoXpert: State-of-the-art Camouflaged Object Detection
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        backbone: str = 'edgenext_small',
        num_experts: int = 5
    ):
        """
        Initialize CamoXpert

        Args:
            in_channels: Input image channels (default: 3 for RGB)
            num_classes: Output classes (default: 1 for binary segmentation)
            pretrained: Use pretrained backbone weights
            backbone: Backbone architecture ('edgenext_small', 'edgenext_base', 'edgenext_base_usi')
            num_experts: Number of MoE experts to create (3-7)
        """
        super().__init__()

        print("\n" + "="*70)
        print("INITIALIZING CAMOXPERT")
        print("="*70)
        print(f"Backbone: {backbone}")

        self.num_experts = num_experts
        self.num_classes = num_classes

        # ========================================
        # 1. Backbone: EdgeNeXt
        # ========================================
        self.backbone = self._create_backbone(backbone, pretrained)

        # Get feature dimensions from backbone
        # EdgeNeXt outputs 4 stages with different channel dimensions
        if 'small' in backbone:
            self.feature_dims = [48, 96, 160, 304]
        elif 'base' in backbone:
            self.feature_dims = [80, 160, 288, 584]
        else:
            self.feature_dims = [80, 160, 288, 584]

        print(f"Feature dimensions: {self.feature_dims}")

        # ========================================
        # 2. SDTA Enhancement Blocks
        # ========================================
        print("\nInitializing SDTA enhancement blocks...")
        self.sdta_blocks = nn.ModuleList([
            SDTABlock(dim) for dim in self.feature_dims
        ])
        print(f"✓ {len(self.sdta_blocks)} SDTA blocks created")

        # ========================================
        # 3. Mixture of Experts Layers
        # ========================================
        print("\nInitializing Mixture of Experts layers...")
        from models.experts import MoELayer

        top_k = max(2, num_experts // 2)  # Use half of experts, minimum 2

        self.moe_layers = nn.ModuleList([
            MoELayer(dim, num_experts=num_experts, top_k=top_k)
            for dim in self.feature_dims
        ])
        print(f"✓ {len(self.moe_layers)} MoE layers created")

        # ========================================
        # 4. Decoder
        # ========================================
        print("\nInitializing decoder...")
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(self.feature_dims[3], self.feature_dims[2]),  # 584->288
            DecoderBlock(self.feature_dims[2], self.feature_dims[1]),  # 288->160
            DecoderBlock(self.feature_dims[1], self.feature_dims[0]),  # 160->80
            DecoderBlock(self.feature_dims[0], 64),                    # 80->64
        ])
        print(f"✓ {len(self.decoder_blocks)} decoder blocks created")

        # ========================================
        # 5. Deep Supervision Heads (output LOGITS)
        # ========================================
        self.deep_heads = nn.ModuleList([
            nn.Conv2d(self.feature_dims[2], num_classes, kernel_size=1),  # 288
            nn.Conv2d(self.feature_dims[1], num_classes, kernel_size=1),  # 160
            nn.Conv2d(self.feature_dims[0], num_classes, kernel_size=1),  # 80
        ])

        # ========================================
        # 6. Final Prediction Head (output LOGITS)
        # ========================================
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
            # NO SIGMOID - Loss function handles it
        )

        # ========================================
        # Summary
        # ========================================
        print("\n" + "="*70)
        print("CamoXpert Architecture Summary:")
        print("="*70)
        print(f"  Backbone:        {backbone}")
        print(f"  Expert Count:    {num_experts} (Top-{top_k} selection per sample)")
        print(f"  Encoder Stages:  {len(self.feature_dims)}")
        print(f"  Decoder Stages:  {len(self.decoder_blocks)}")
        print(f"  Deep Supervision: Enabled ({len(self.deep_heads)} heads)")
        print(f"  Output:          LOGITS (no sigmoid)")
        print("="*70)
        print()

    def _create_backbone(self, backbone: str, pretrained: bool):
        """Create EdgeNeXt backbone"""
        try:
            import timm
            model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
            return model
        except Exception as e:
            print(f"⚠️  Error loading backbone: {e}")
            print("Attempting alternative backbone loading...")

            # Fallback
            import timm
            if 'small' in backbone:
                model = timm.create_model('edgenext_small', pretrained=pretrained, features_only=True)
            else:
                model = timm.create_model('edgenext_base', pretrained=pretrained, features_only=True)
            return model

    def forward(self, x, return_deep_supervision=False):
        """
        Forward pass through CamoXpert

        Args:
            x: Input images [B, 3, H, W]
            return_deep_supervision: Whether to return intermediate predictions

        Returns:
            pred: Final prediction LOGITS [B, 1, H, W] (no sigmoid applied)
            aux_loss: Auxiliary MoE load balancing loss
            deep_outputs: List of deep supervision LOGITS (if enabled) or None
        """

        B, _, H, W = x.shape

        # ========================================
        # 1. Encoder: Extract multi-scale features
        # ========================================
        encoder_features = self.backbone(x)

        # encoder_features is a list: [f1, f2, f3, f4]
        # Typical shapes for img_size=288:
        #   f1: [B, 80, 72, 72]
        #   f2: [B, 160, 36, 36]
        #   f3: [B, 288, 18, 18]
        #   f4: [B, 584, 9, 9]

        # ========================================
        # 2. Enhancement: Apply SDTA + MoE
        # ========================================
        enhanced_features = []
        total_aux_loss = 0.0

        for i, (feat, sdta, moe) in enumerate(zip(
            encoder_features,
            self.sdta_blocks,
            self.moe_layers
        )):
            # Apply SDTA enhancement
            feat = sdta(feat)

            # Apply MoE with experts
            # Returns: enhanced_feat, aux_loss, routing_info
            feat, aux_loss, routing_info = moe(feat)

            # Accumulate auxiliary loss
            if aux_loss is not None:
                total_aux_loss += aux_loss

            enhanced_features.append(feat)

        # enhanced_features[0] = f1_enhanced [B, 80, 72, 72]
        # enhanced_features[1] = f2_enhanced [B, 160, 36, 36]
        # enhanced_features[2] = f3_enhanced [B, 288, 18, 18]
        # enhanced_features[3] = f4_enhanced [B, 584, 9, 9]

        # ========================================
        # 3. Decoder: Progressive upsampling
        # ========================================
        deep_outputs = [] if return_deep_supervision else None

        # Start with deepest features
        x4 = enhanced_features[3]  # [B, 584, 9, 9]

        # Decode stage 4 -> 3
        x3 = self.decoder_blocks[0](x4, enhanced_features[2])  # [B, 288, 18, 18]
        if return_deep_supervision:
            deep_pred = self.deep_heads[0](x3)  # LOGITS [B, 1, 18, 18]
            deep_outputs.append(deep_pred)

        # Decode stage 3 -> 2
        x2 = self.decoder_blocks[1](x3, enhanced_features[1])  # [B, 160, 36, 36]
        if return_deep_supervision:
            deep_pred = self.deep_heads[1](x2)  # LOGITS [B, 1, 36, 36]
            deep_outputs.append(deep_pred)

        # Decode stage 2 -> 1
        x1 = self.decoder_blocks[2](x2, enhanced_features[0])  # [B, 80, 72, 72]
        if return_deep_supervision:
            deep_pred = self.deep_heads[2](x1)  # LOGITS [B, 1, 72, 72]
            deep_outputs.append(deep_pred)

        # Decode stage 1 -> output
        x0 = self.decoder_blocks[3](x1, None)  # [B, 64, 144, 144]

        # Upsample to original size
        x0_up = F.interpolate(x0, size=(H, W), mode='bilinear', align_corners=False)

        # ========================================
        # 4. Final prediction (RETURN LOGITS)
        # ========================================
        pred = self.final_conv(x0_up)  # [B, 1, H, W] - LOGITS

        # DO NOT APPLY SIGMOID HERE
        # The loss function (BCEWithLogitsLoss) will handle it

        # ========================================
        # 5. Return outputs
        # ========================================
        if return_deep_supervision:
            return pred, total_aux_loss, deep_outputs
        else:
            return pred, total_aux_loss, None


class SDTABlock(nn.Module):
    """
    Selective Dual-axis Temporal Attention Block
    Enhances features with spatial attention mechanism
    """

    def __init__(self, dim, reduction=8):
        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connections
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Fusion of upsampled and skip features
        if out_channels == 64:
            # Last decoder block (no skip connection)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # With skip connection
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, skip=None):
        """
        Args:
            x: Input features from previous decoder stage
            skip: Skip connection features from encoder (can be None for last stage)
        """
        x = self.upsample(x)

        if skip is not None:
            # Resize if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == '__main__':
    # Test model
    print("Testing CamoXpert...")

    model = CamoXpert(
        in_channels=3,
        num_classes=1,
        pretrained=False,
        backbone='edgenext_small',
        num_experts=5
    )

    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total/1e6:.2f}M")
    print(f"Trainable parameters: {trainable/1e6:.2f}M")

    # Test forward pass
    x = torch.randn(2, 3, 288, 288)

    print("\nTesting forward pass...")
    pred, aux_loss, deep = model(x, return_deep_supervision=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Aux loss: {aux_loss}")
    print(f"Deep supervision outputs: {len(deep) if deep else 0}")

    if deep:
        for i, d in enumerate(deep):
            print(f"  Deep {i+1}: {d.shape}")

    print("\n✓ Model test successful!")

    # Verify outputs are logits (not sigmoid)
    print(f"\nOutput range check:")
    print(f"  Min: {pred.min().item():.4f}")
    print(f"  Max: {pred.max().item():.4f}")
    print(f"  Mean: {pred.mean().item():.4f}")
    print(f"  (Should be unbounded, not [0, 1])")