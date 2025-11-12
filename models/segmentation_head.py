import torch
import torch.nn as nn
from models.backbone import LayerNorm2d


class SegmentationHead(nn.Module):
    """
    Segmentation head for final mask prediction.

    Progressive upsampling with skip connections for detail preservation.
    """
    def __init__(self, in_dim=64, num_classes=1):
        super().__init__()

        self.decoder = nn.Sequential(
            # First upsampling block (1/4 -> 1/2)
            nn.ConvTranspose2d(in_dim, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            LayerNorm2d(32),
            nn.GELU(),

            # Second upsampling block (1/2 -> 1/1)
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            LayerNorm2d(16),
            nn.GELU(),

            # Final prediction
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)