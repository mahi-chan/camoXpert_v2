import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps (B, C, H, W).
    """

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x


class SDTAEncoder(nn.Module):
    """
    Spatial Dimension Transposed Attention Encoder with Efficient Linear Attention.

    Uses Linear Attention (O(N) complexity) instead of standard attention (O(N²)).
    This dramatically reduces memory and computation for high-resolution features.
    """

    def __init__(self, dim, num_heads=8, drop_path=0.1, use_linear_attention=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_linear_attention = use_linear_attention

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop_path = nn.Dropout(drop_path)

    def linear_attention(self, q, k, v):
        """
        Linear Attention: O(N) complexity instead of O(N²)

        Standard attention: Softmax(Q @ K^T) @ V -> O(N²) for N = H*W
        Linear attention: Q @ (K^T @ V) -> O(N) using kernel trick

        Memory: O(d²) instead of O(N²) where d << N
        Speed: 3-5x faster for high resolution
        """
        # Apply feature map: elu(x) + 1 (ensures positive values)
        q = F.elu(q) + 1  # [B, H, N, D]
        k = F.elu(k) + 1  # [B, H, N, D]

        # Normalize queries and keys for stability
        q = q / q.sum(dim=-1, keepdim=True)
        k = k / k.sum(dim=-1, keepdim=True)

        # Linear attention: Q @ (K^T @ V) instead of (Q @ K^T) @ V
        # This exploits associativity of matrix multiplication
        kv = k.transpose(-2, -1) @ v  # [B, H, D, D] - Much smaller than [B, H, N, N]!
        out = q @ kv  # [B, H, N, D]

        return out

    def standard_attention(self, q, k, v):
        """
        Standard O(N²) attention (kept for compatibility/comparison)
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N] - LARGE!
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, H, N, D]
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)

        # Choose attention mechanism
        if self.use_linear_attention:
            out = self.linear_attention(q, k, v)
        else:
            out = self.standard_attention(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)

        # Project and add residual
        out = self.proj(out)
        out = self.drop_path(out) + x

        return out


class ConvBlock(nn.Module):
    """
    Convolutional block with normalization and activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = LayerNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class EdgeNeXtBlock(nn.Module):
    """
    EdgeNeXt block combining depthwise convolution and attention.
    """

    def __init__(self, dim, drop_path=0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        x = input + self.drop_path(x)
        return x


class EdgeNeXtBackbone(nn.Module):
    """
    EdgeNeXt backbone for multi-scale feature extraction.
    """

    def __init__(self, in_channels=3, depths=[3, 3, 9, 3], dims=[48, 96, 160, 256], drop_path_rate=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0

        for i in range(len(depths)):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(EdgeNeXtBlock(dims[i], dp_rates[cur]))
                cur += 1
            self.stages.append(nn.Sequential(*stage_blocks))

            # Add downsampling between stages (except last)
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
                self.stages.append(downsample)

        self.num_stages = len(depths)
        self.dims = dims

    def forward(self, x):
        features = []
        x = self.stem(x)

        stage_idx = 0
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if i % 2 == 0:  # After each stage block (not downsampling)
                features.append(x)
                stage_idx += 1
                if stage_idx >= self.num_stages:
                    break

        return features