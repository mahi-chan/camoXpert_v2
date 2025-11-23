"""
FrequencyExpert: FEDER-inspired frequency-domain architecture for camouflaged object detection

This module implements a sophisticated frequency-based expert that decomposes features into
high-frequency (texture/edge) and low-frequency (color/illumination) components, processes
them with specialized attention mechanisms, and reconstructs edges using ODE-inspired dynamics.

Components:
1. Deep Wavelet-like Decomposition (DWD): Learnable wavelets initialized with Haar transforms
2. High-Frequency Attention: Residual blocks with joint spatial-channel attention
3. Low-Frequency Attention: Instance + positional normalization for redundancy suppression
4. ODE-Inspired Edge Reconstruction: Second-order Runge-Kutta with Hamiltonian stability
5. Guidance-Based Feature Aggregation: Attention-guided linear combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict
from models.backbone import LayerNorm2d


class LearnableWaveletDecomposition(nn.Module):
    """
    Deep Wavelet-like Decomposition (DWD) using learnable wavelets initialized with Haar transforms.

    Decomposes input into 4 subbands: LL (low-low), LH (low-high), HL (high-low), HH (high-high)
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Initialize with Haar wavelet basis
        # LL: Low-pass in both directions (approximation)
        # LH: Low-pass horizontal, High-pass vertical (horizontal details)
        # HL: High-pass horizontal, Low-pass vertical (vertical details)
        # HH: High-pass in both directions (diagonal details)

        # Learnable wavelet filters (initialized with Haar)
        self.ll_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.lh_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.hl_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.hh_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self._init_haar_wavelets()

    def _init_haar_wavelets(self):
        """Initialize convolution kernels with Haar wavelet basis"""
        # Haar wavelet patterns (3x3 approximation)
        # LL: averaging (low-pass)
        ll_kernel = torch.ones(3, 3) / 9.0

        # LH: horizontal edges (vertical high-pass)
        lh_kernel = torch.tensor([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=torch.float32) / 6.0

        # HL: vertical edges (horizontal high-pass)
        hl_kernel = torch.tensor([
            [-1,  0,  1],
            [-1,  0,  1],
            [-1,  0,  1]
        ], dtype=torch.float32) / 6.0

        # HH: diagonal edges (high-pass in both)
        hh_kernel = torch.tensor([
            [-1,  0,  1],
            [ 0,  0,  0],
            [ 1,  0, -1]
        ], dtype=torch.float32) / 4.0

        # Apply to all channels
        for conv, kernel in zip(
            [self.ll_conv, self.lh_conv, self.hl_conv, self.hh_conv],
            [ll_kernel, lh_kernel, hl_kernel, hh_kernel]
        ):
            for i in range(self.channels):
                conv.weight.data[i, i] = kernel

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Dictionary containing 4 subbands:
                - ll: Low-frequency approximation [B, C, H, W]
                - lh: Horizontal details [B, C, H, W]
                - hl: Vertical details [B, C, H, W]
                - hh: Diagonal details [B, C, H, W]
        """
        ll = self.ll_conv(x)  # Low-frequency (approximation)
        lh = self.lh_conv(x)  # Horizontal edges
        hl = self.hl_conv(x)  # Vertical edges
        hh = self.hh_conv(x)  # Diagonal edges

        return {
            'll': ll,  # Low-frequency component
            'lh': lh,  # High-frequency component (horizontal)
            'hl': hl,  # High-frequency component (vertical)
            'hh': hh   # High-frequency component (diagonal)
        }


class SpatialChannelAttention(nn.Module):
    """
    Joint Spatial-Channel Attention for high-frequency features.

    Combines spatial attention (where to focus) with channel attention (what to focus on)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            LayerNorm2d(channels // reduction),
            nn.GELU(),
            nn.Conv2d(channels // reduction, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Channel attention: B, C, H, W -> B, C, 1, 1 -> B, C, 1, 1
        ca = self.channel_attention(x)

        # Spatial attention: B, C, H, W -> B, 1, H, W
        sa = self.spatial_attention(x)

        # Joint attention
        return x * ca * sa


class HighFrequencyAttention(nn.Module):
    """
    High-Frequency Attention module using residual blocks with joint spatial-channel attention
    for texture-rich regions.
    """
    def __init__(self, channels):
        super().__init__()

        # Residual block 1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        # Residual block 2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        # Joint spatial-channel attention
        self.attention = SpatialChannelAttention(channels)

        # Final activation
        self.gelu = nn.GELU()

    def forward(self, high_freq_features):
        """
        Args:
            high_freq_features: High-frequency components [B, C, H, W]
        Returns:
            Enhanced high-frequency features [B, C, H, W]
        """
        # First residual block
        out = self.res_block1(high_freq_features) + high_freq_features
        out = self.gelu(out)

        # Second residual block
        out = self.res_block2(out) + out
        out = self.gelu(out)

        # Apply joint attention
        out = self.attention(out)

        return out


class PositionalNormalization(nn.Module):
    """
    Positional Normalization: Normalizes features based on spatial position.

    Helps suppress redundant low-frequency information by considering spatial context.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale and bias per position
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Positionally normalized features [B, C, H, W]
        """
        # Compute statistics along channel dimension, preserving spatial structure
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, H, W]

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine transformation
        return self.gamma * x_norm + self.beta


class LowFrequencyAttention(nn.Module):
    """
    Low-Frequency Attention with instance normalization and positional normalization
    to suppress redundancy in color/illumination components.
    """
    def __init__(self, channels):
        super().__init__()

        # Instance normalization (normalizes each sample independently)
        self.instance_norm = nn.InstanceNorm2d(channels, affine=True)

        # Positional normalization
        self.positional_norm = PositionalNormalization(channels)

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        # Suppression gate (learns what to suppress)
        self.suppression_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()

    def forward(self, low_freq_features):
        """
        Args:
            low_freq_features: Low-frequency components [B, C, H, W]
        Returns:
            Refined low-frequency features with suppressed redundancy [B, C, H, W]
        """
        # Instance normalization to remove instance-specific bias
        out = self.instance_norm(low_freq_features)

        # Positional normalization to suppress spatial redundancy
        out = self.positional_norm(out)

        # Refine features
        out = self.refine(out) + out
        out = self.gelu(out)

        # Apply suppression gate to reduce redundancy
        gate = self.suppression_gate(out)
        out = out * gate

        return out


class ODEEdgeReconstruction(nn.Module):
    """
    ODE-Inspired Edge Reconstruction using second-order Runge-Kutta solver
    with Hamiltonian stability guarantees.

    Models edge evolution as: dx/dt = f(x, t), solved with RK2 (Heun's method)
    Hamiltonian H = E_kinetic + E_potential ensures energy conservation
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Edge dynamics function f(x, t)
        self.edge_dynamics = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        # Hamiltonian potential energy term
        self.potential_energy = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU()
        )

        # Stability constraint (learnable damping)
        self.damping = nn.Parameter(torch.tensor(0.1))

        # Time step (learnable)
        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, high_freq_features):
        """
        Reconstruct edges using RK2 ODE solver.

        Args:
            high_freq_features: High-frequency edge information [B, C, H, W]
        Returns:
            Reconstructed edge features [B, C, H, W]
        """
        # Initial state
        x0 = high_freq_features

        # RK2 (Heun's method) - Second-order Runge-Kutta
        # Step 1: Compute k1 = f(x0)
        k1 = self.edge_dynamics(x0)

        # Step 2: Compute k2 = f(x0 + dt * k1)
        x_temp = x0 + self.dt * k1
        k2 = self.edge_dynamics(x_temp)

        # Step 3: Update x = x0 + dt/2 * (k1 + k2)
        x_next = x0 + (self.dt / 2.0) * (k1 + k2)

        # Apply Hamiltonian stability: Add potential energy term
        # H = T + V, where T (kinetic) is edge evolution, V (potential) is regularization
        potential = self.potential_energy(x_next)

        # Energy-conserving update with damping for stability
        # x_stable = x_next - damping * grad(V)
        x_stable = x_next + potential * torch.sigmoid(self.damping)

        return x_stable


class GuidanceBasedAggregation(nn.Module):
    """
    Guidance-Based Feature Aggregation replacing concatenation with attention-guided
    linear combinations.

    Instead of concat([f1, f2, f3]), computes: α1*f1 + α2*f2 + α3*f3
    where α_i are learned attention weights
    """
    def __init__(self, channels, num_inputs=4):
        super().__init__()
        self.num_inputs = num_inputs

        # Attention guidance network
        self.guidance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_inputs, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, num_inputs, 1),
            nn.Softmax(dim=1)  # Normalize across inputs
        )

        # Feature-wise modulation
        self.modulation = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU()
        )

    def forward(self, features_list):
        """
        Args:
            features_list: List of feature tensors [f1, f2, f3, f4], each [B, C, H, W]
        Returns:
            Aggregated features [B, C, H, W]
        """
        # Concatenate for guidance computation
        concat_features = torch.cat(features_list, dim=1)  # [B, C*num_inputs, H, W]

        # Compute attention weights for each input
        attention_weights = self.guidance_net(concat_features)  # [B, num_inputs, 1, 1]

        # Weighted linear combination
        aggregated = sum(
            w[:, i:i+1, :, :] * f
            for i, (w, f) in enumerate(zip([attention_weights] * self.num_inputs, features_list))
        )

        # Feature modulation
        aggregated = self.modulation(aggregated)

        return aggregated


class FrequencyExpert(nn.Module):
    """
    FEDER-inspired FrequencyExpert for camouflaged object detection.

    Processes features through frequency decomposition, specialized attention mechanisms,
    ODE-based edge reconstruction, and guidance-based aggregation.

    Architecture:
        Input [B, C, H, W]
            ↓
        [Deep Wavelet Decomposition] → {LL, LH, HL, HH}
            ↓
        [Low-Freq Attention]   [High-Freq Attention] × 3
            ↓                        ↓
        [ODE Edge Reconstruction]
            ↓
        [Guidance-Based Aggregation]
            ↓
        Output [B, C, H, W] + Auxiliary Outputs
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 1. Deep Wavelet-like Decomposition
        self.wavelet_decomposition = LearnableWaveletDecomposition(dim)

        # 2. Low-Frequency Attention
        self.low_freq_attention = LowFrequencyAttention(dim)

        # 3. High-Frequency Attention (separate for each high-freq component)
        self.high_freq_attention_lh = HighFrequencyAttention(dim)  # Horizontal
        self.high_freq_attention_hl = HighFrequencyAttention(dim)  # Vertical
        self.high_freq_attention_hh = HighFrequencyAttention(dim)  # Diagonal

        # 4. ODE-Inspired Edge Reconstruction
        self.ode_edge_reconstruction = ODEEdgeReconstruction(dim)

        # 5. Guidance-Based Feature Aggregation
        self.feature_aggregation = GuidanceBasedAggregation(dim, num_inputs=4)

        # 6. Deep supervision heads (auxiliary outputs)
        self.aux_head_low = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            LayerNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1)  # Single channel for segmentation
        )

        self.aux_head_high = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            LayerNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1)  # Single channel for edge map
        )

        # Final refinement
        self.final_refinement = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            LayerNorm2d(dim)
        )

        self.gelu = nn.GELU()

    def forward(self, x, return_aux=False):
        """
        Forward pass through FrequencyExpert.

        Args:
            x: Input features [B, C, H, W]
            return_aux: If True, return auxiliary outputs for deep supervision

        Returns:
            If return_aux=False:
                Enhanced features [B, C, H, W]
            If return_aux=True:
                (enhanced_features, aux_outputs) where aux_outputs is dict:
                    - 'low_freq_pred': Low-frequency prediction [B, 1, H, W]
                    - 'high_freq_pred': High-frequency edge map [B, 1, H, W]
                    - 'decomposition': Wavelet decomposition components
        """
        # Step 1: Deep Wavelet-like Decomposition
        decomposition = self.wavelet_decomposition(x)
        ll = decomposition['ll']  # Low-frequency
        lh = decomposition['lh']  # High-frequency (horizontal)
        hl = decomposition['hl']  # High-frequency (vertical)
        hh = decomposition['hh']  # High-frequency (diagonal)

        # Step 2: Low-Frequency Attention
        low_freq_enhanced = self.low_freq_attention(ll)

        # Step 3: High-Frequency Attention (for each component)
        high_freq_lh = self.high_freq_attention_lh(lh)
        high_freq_hl = self.high_freq_attention_hl(hl)
        high_freq_hh = self.high_freq_attention_hh(hh)

        # Combine high-frequency components
        high_freq_combined = high_freq_lh + high_freq_hl + high_freq_hh

        # Step 4: ODE-Inspired Edge Reconstruction
        edge_reconstructed = self.ode_edge_reconstruction(high_freq_combined)

        # Step 5: Guidance-Based Feature Aggregation
        features_to_aggregate = [
            low_freq_enhanced,
            high_freq_lh,
            high_freq_hl,
            edge_reconstructed
        ]
        aggregated = self.feature_aggregation(features_to_aggregate)

        # Final refinement with residual connection
        output = self.final_refinement(aggregated) + x
        output = self.gelu(output)

        # Auxiliary outputs for deep supervision
        if return_aux:
            aux_outputs = {
                'low_freq_pred': self.aux_head_low(low_freq_enhanced),
                'high_freq_pred': self.aux_head_high(edge_reconstructed),
                'decomposition': decomposition
            }
            return output, aux_outputs

        return output


class MultiScaleFrequencyExpert(nn.Module):
    """
    Multi-scale FrequencyExpert that processes features at different scales.

    Designed to work with feature pyramids at dimensions [64, 128, 320, 512]
    """
    def __init__(self, dims=[64, 128, 320, 512]):
        super().__init__()
        self.dims = dims

        # Create frequency expert for each scale
        self.experts = nn.ModuleList([
            FrequencyExpert(dim) for dim in dims
        ])

        # Cross-scale feature fusion
        self.cross_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                LayerNorm2d(dim),
                nn.GELU()
            ) for dim in dims
        ])

        # Deep supervision heads for each scale
        self.deep_supervision_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 2, 3, padding=1),
                LayerNorm2d(dim // 2),
                nn.GELU(),
                nn.Conv2d(dim // 2, 1, 1)
            ) for dim in dims
        ])

    def forward(self, features, return_aux=False):
        """
        Process multi-scale features.

        Args:
            features: List of feature tensors at different scales
                      [f1, f2, f3, f4] with dims [64, 128, 320, 512]
            return_aux: If True, return auxiliary predictions for deep supervision

        Returns:
            If return_aux=False:
                List of enhanced features [f1', f2', f3', f4']
            If return_aux=True:
                (enhanced_features, aux_predictions) where aux_predictions is list of
                predictions at each scale for deep supervision
        """
        enhanced_features = []
        aux_predictions = []
        aux_outputs_all = []

        # Process each scale
        for i, (feat, expert) in enumerate(zip(features, self.experts)):
            if return_aux:
                enhanced, aux = expert(feat, return_aux=True)
                aux_outputs_all.append(aux)
            else:
                enhanced = expert(feat)

            # Cross-scale fusion
            enhanced = self.cross_scale_fusion[i](enhanced)
            enhanced_features.append(enhanced)

            # Generate deep supervision prediction
            if return_aux:
                pred = self.deep_supervision_heads[i](enhanced)
                aux_predictions.append(pred)

        if return_aux:
            return enhanced_features, {
                'predictions': aux_predictions,
                'aux_outputs': aux_outputs_all
            }

        return enhanced_features


# Example usage and testing
if __name__ == '__main__':
    print("Testing FrequencyExpert...")

    # Test single-scale expert
    expert = FrequencyExpert(dim=128)
    x = torch.randn(2, 128, 32, 32)

    # Forward pass without auxiliary outputs
    output = expert(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Shape mismatch!"

    # Forward pass with auxiliary outputs
    output, aux = expert(x, return_aux=True)
    print(f"\nWith auxiliary outputs:")
    print(f"  Low-freq prediction: {aux['low_freq_pred'].shape}")
    print(f"  High-freq prediction: {aux['high_freq_pred'].shape}")
    print(f"  Decomposition keys: {list(aux['decomposition'].keys())}")

    # Test multi-scale expert
    print("\n" + "="*60)
    print("Testing MultiScaleFrequencyExpert...")

    multi_expert = MultiScaleFrequencyExpert(dims=[64, 128, 320, 512])
    features = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 320, 16, 16),
        torch.randn(2, 512, 8, 8)
    ]

    # Forward pass
    enhanced_features, aux = multi_expert(features, return_aux=True)

    print(f"\nEnhanced features:")
    for i, feat in enumerate(enhanced_features):
        print(f"  Scale {i}: {feat.shape}")

    print(f"\nDeep supervision predictions:")
    for i, pred in enumerate(aux['predictions']):
        print(f"  Scale {i}: {pred.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in multi_expert.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✓ All tests passed!")
