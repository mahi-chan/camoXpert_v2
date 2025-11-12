import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import LayerNorm2d


class BiLevelFusion(nn.Module):
    """
    Bi-level feature fusion combining low-level and high-level features.
    """
    def __init__(self, dims=[48, 96, 160, 256], out_dim=64):
        super().__init__()
        # Low-level fusion (Stage 1 + 2)
        self.low_fusion = nn.Sequential(
            nn.Conv2d(dims[0] + dims[1], dims[0], kernel_size=1),
            LayerNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        # High-level fusion (Stage 3 + 4)
        self.high_fusion = nn.Sequential(
            nn.Conv2d(dims[2] + dims[3], dims[2], kernel_size=1),
            LayerNorm2d(dims[2]),
            nn.GELU(),
            nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1),
            LayerNorm2d(dims[2]),
            nn.GELU()
        )

        # Cross-level fusion
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(dims[0] + dims[2], out_dim * 2, kernel_size=3, padding=1),
            LayerNorm2d(out_dim * 2),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
            LayerNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, features):
        """
        Args:
            features: List of 4 feature maps from backbone stages

        Returns:
            fused: Fused feature map at 1/4 resolution
        """
        f1, f2, f3, f4 = features

        # Upsample to match resolutions
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)

        # Low and high level fusion
        low_fused = self.low_fusion(torch.cat([f1, f2_up], dim=1))
        high_fused = self.high_fusion(torch.cat([f3, f4_up], dim=1))

        # Upsample high-level to match low-level
        high_up = F.interpolate(high_fused, size=low_fused.shape[2:], mode='bilinear', align_corners=False)

        # Cross-level fusion
        fused = self.cross_fusion(torch.cat([low_fused, high_up], dim=1))

        return fused