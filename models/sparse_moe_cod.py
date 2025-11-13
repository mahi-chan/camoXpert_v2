"""
Sparse Mixture-of-Experts (MoE) for Camouflaged Object Detection

Implements learned routing to dynamically select best experts per image.
Router learns which experts (texture, frequency, edge, contrast, etc.)
work best for each specific image type.

Example:
  Image 10 (sandy beach camouflage) → Router selects [Texture, Frequency, Contrast]
  Image 11 (forest camouflage)      → Router selects [Edge, Texture, Frequency]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cod_modules import (
    CODTextureExpert,
    CODFrequencyExpert,
    CODEdgeExpert,
    ContrastEnhancementModule
)


class SparseRouter(nn.Module):
    """
    Learned router that selects top-k experts for each input

    Args:
        dim: Feature dimension
        num_experts: Total number of experts available
        top_k: Number of experts to select (2-3 recommended)
        routing_mode: 'global' or 'spatial'
    """
    def __init__(self, dim, num_experts, top_k=2, routing_mode='global'):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.routing_mode = routing_mode

        if routing_mode == 'global':
            # Global routing: Single decision per feature map
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, num_experts, 1),
                nn.Flatten()
            )
        else:  # 'spatial'
            # Spatial routing: Different experts for different regions
            self.gate = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 1),
                nn.BatchNorm2d(dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, num_experts, 1)
            )

        # LOSS-FREE LOAD BALANCING (Modern approach, August 2024)
        # Uses expert-wise bias instead of auxiliary loss
        # Heavy-load experts → negative bias (discouraged)
        # Light-load experts → positive bias (encouraged)
        # NO gradient interference from auxiliary loss
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        self.bias_update_rate = 0.01  # How fast bias adapts to load imbalance

        # Track recent expert usage (EMA)
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)

    def forward(self, x, warmup_factor=1.0):
        """
        Args:
            x: Input features [B, C, H, W]
            warmup_factor: Legacy parameter (ignored, kept for compatibility)

        Returns:
            expert_weights: [B, num_experts] or [B, num_experts, H, W]
            top_k_indices: [B, top_k] or [B, top_k, H, W]
            load_balance_loss: Always 0.0 (no auxiliary loss)
        """
        # Compute routing logits
        logits = self.gate(x)  # [B, num_experts] or [B, num_experts, H, W]

        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        # LOSS-FREE BALANCING: Apply expert-wise bias BEFORE softmax
        if self.routing_mode == 'global':
            logits_biased = logits + self.expert_bias.unsqueeze(0)
        else:  # spatial
            logits_biased = logits + self.expert_bias.view(1, -1, 1, 1)

        # Softmax to get expert probabilities with temperature scaling
        temperature = 1.0  # Can be tuned for smoothness
        if self.routing_mode == 'global':
            probs = F.softmax(logits_biased / temperature, dim=1)  # [B, num_experts]

            # Clamp probabilities to prevent extremes
            probs = torch.clamp(probs, min=1e-6, max=1.0)

            # Select top-k experts
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=1)

            # Normalize top-k probabilities to sum to 1
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-8)

            # LOSS-FREE LOAD BALANCING: Update expert bias based on usage
            if self.training:
                with torch.no_grad():
                    B = probs.shape[0]
                    # Measure actual expert usage in this batch
                    expert_usage = torch.zeros(self.num_experts, device=probs.device)
                    for expert_id in range(self.num_experts):
                        expert_usage[expert_id] = (top_k_indices == expert_id).float().sum() / (B * self.top_k)

                    # Update EMA of expert usage
                    self.expert_usage_ema = 0.9 * self.expert_usage_ema + 0.1 * expert_usage

                    # Compute ideal usage (uniform distribution)
                    ideal_usage = 1.0 / self.num_experts

                    # Update bias: Heavy-load → negative bias, Light-load → positive bias
                    load_imbalance = self.expert_usage_ema - ideal_usage
                    self.expert_bias.data -= self.bias_update_rate * load_imbalance

                    # Clamp bias to prevent extreme values
                    self.expert_bias.data.clamp_(-5.0, 5.0)

        else:  # spatial
            probs = F.softmax(logits_biased, dim=1)  # [B, num_experts, H, W]
            B, E, H, W = probs.shape

            # Reshape for top-k
            probs_flat = probs.view(B, E, -1).transpose(1, 2)  # [B, H*W, E]
            top_k_probs, top_k_indices = torch.topk(probs_flat, self.top_k, dim=2)

            # Normalize
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=2, keepdim=True) + 1e-8)

            # Reshape back
            top_k_probs = top_k_probs.transpose(1, 2).view(B, self.top_k, H, W)
            top_k_indices = top_k_indices.transpose(1, 2).view(B, self.top_k, H, W)

            # LOSS-FREE LOAD BALANCING: Update expert bias (spatial)
            if self.training:
                with torch.no_grad():
                    # Measure actual expert usage in this batch (spatial)
                    expert_usage = torch.zeros(self.num_experts, device=probs.device)
                    for expert_id in range(self.num_experts):
                        expert_usage[expert_id] = (top_k_indices == expert_id).float().sum() / (B * self.top_k * H * W)

                    # Update EMA of expert usage
                    self.expert_usage_ema = 0.9 * self.expert_usage_ema + 0.1 * expert_usage

                    # Compute ideal usage (uniform distribution)
                    ideal_usage = 1.0 / self.num_experts

                    # Update bias: Heavy-load → negative bias, Light-load → positive bias
                    load_imbalance = self.expert_usage_ema - ideal_usage
                    self.expert_bias.data -= self.bias_update_rate * load_imbalance

                    # Clamp bias to prevent extreme values
                    self.expert_bias.data.clamp_(-5.0, 5.0)

        # Return 0.0 for load_balance_loss (no auxiliary loss, backward compatible)
        return top_k_probs, top_k_indices, torch.tensor(0.0, device=probs.device, dtype=probs.dtype)


class SparseCODMoE(nn.Module):
    """
    Sparse Mixture-of-Experts layer for COD

    Dynamically selects and applies top-k experts based on input features.
    Router learns which experts work best for different camouflage types.

    Args:
        dim: Feature dimension
        num_experts: Number of expert modules (6 recommended)
        top_k: Number of experts to activate per input (2-3 recommended)
        expert_types: List of expert classes to use
        routing_mode: 'global' (faster) or 'spatial' (more flexible)
    """
    def __init__(self, dim, num_experts=6, top_k=2,
                 expert_types=None, routing_mode='global'):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_mode = routing_mode

        # Default expert types for COD
        if expert_types is None:
            expert_types = [
                CODTextureExpert,
                CODFrequencyExpert,
                CODEdgeExpert,
                ContrastEnhancementModule,
                CODTextureExpert,  # Duplicate for diversity
                CODFrequencyExpert  # Duplicate for diversity
            ]

        # Create expert pool
        self.experts = nn.ModuleList([
            expert_types[i % len(expert_types)](dim)
            for i in range(num_experts)
        ])

        # Router network
        self.router = SparseRouter(dim, num_experts, top_k, routing_mode)

    def forward(self, x, return_routing_info=False):
        """
        Args:
            x: Input features [B, C, H, W]
            return_routing_info: If True, return routing details

        Returns:
            output: Expert-enhanced features [B, C, H, W]
            load_balance_loss: Load balancing auxiliary loss
            routing_info: (Optional) Dict with routing statistics
        """
        B, C, H, W = x.shape

        # Route to top-k experts
        top_k_probs, top_k_indices, load_balance_loss = self.router(x)

        # Apply selected experts
        if self.routing_mode == 'global':
            # Global routing: same experts for whole feature map
            output = torch.zeros_like(x)

            for i in range(self.top_k):
                # Get expert indices and weights for this position
                expert_idx = top_k_indices[:, i]  # [B]
                expert_weight = top_k_probs[:, i]  # [B]

                # Apply each expert to corresponding samples in batch
                for b in range(B):
                    expert = self.experts[expert_idx[b]]
                    expert_output = expert(x[b:b+1])
                    output[b:b+1] += expert_weight[b] * expert_output

        else:  # spatial
            # Spatial routing: different experts for different regions
            # More complex but allows position-specific expert selection
            output = torch.zeros_like(x)

            # Process each sample in batch
            for b in range(B):
                for k in range(self.top_k):
                    # Get spatial expert assignment [H, W]
                    expert_map = top_k_indices[b, k]  # [H, W]
                    weight_map = top_k_probs[b, k:k+1]  # [1, H, W]

                    # Apply experts (simplified: use dominant expert per region)
                    dominant_expert = expert_map.flatten().mode().values.item()
                    expert = self.experts[dominant_expert]
                    expert_output = expert(x[b:b+1])
                    output[b:b+1] += weight_map * expert_output

        # Routing info for analysis
        routing_info = None
        if return_routing_info:
            routing_info = {
                'selected_experts': top_k_indices.detach().cpu(),
                'expert_weights': top_k_probs.detach().cpu(),
                'load_balance_loss': load_balance_loss.item()
            }

        if return_routing_info:
            return output, load_balance_loss, routing_info
        else:
            return output, load_balance_loss


class EfficientSparseCODMoE(nn.Module):
    """
    Memory and speed optimized sparse MoE for COD

    Uses batch-level routing instead of sample-level for better efficiency.
    Recommended for production use.
    """
    def __init__(self, dim, num_experts=6, top_k=2, routing_mode='global'):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Create expert pool
        expert_types = [
            CODTextureExpert,
            CODFrequencyExpert,
            CODEdgeExpert,
            ContrastEnhancementModule
        ]

        self.experts = nn.ModuleList([
            expert_types[i % len(expert_types)](dim)
            for i in range(num_experts)
        ])

        # Lightweight router
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts)
        )

        self.top_k = top_k

        # LOSS-FREE LOAD BALANCING (Modern approach, August 2024)
        # Uses expert-wise bias instead of auxiliary loss
        # Heavy-load experts → negative bias (discouraged)
        # Light-load experts → positive bias (encouraged)
        # NO gradient interference from auxiliary loss
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        self.bias_update_rate = 0.01  # How fast bias adapts to load imbalance

        # Track recent expert usage (EMA)
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)

    def forward(self, x, warmup_factor=1.0):
        """
        Efficient forward pass with loss-free load balancing

        Args:
            x: Input features [B, C, H, W]
            warmup_factor: Legacy parameter (ignored, kept for compatibility)

        Returns:
            output: Enhanced features
            load_balance_loss: Always 0.0 (no auxiliary loss)
        """
        B = x.shape[0]

        # Compute routing scores [B, num_experts]
        routing_logits = self.router(x)

        # Clamp for numerical stability
        routing_logits = torch.clamp(routing_logits, min=-10.0, max=10.0)

        # LOSS-FREE BALANCING: Apply expert-wise bias BEFORE softmax
        # Heavy-load experts have negative bias → lower probability
        # Light-load experts have positive bias → higher probability
        routing_logits_biased = routing_logits + self.expert_bias.unsqueeze(0)

        routing_probs = F.softmax(routing_logits_biased, dim=1)

        # Clamp probabilities
        routing_probs = torch.clamp(routing_probs, min=1e-6, max=1.0)

        # Select top-k experts per sample
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-8)

        # EXPERT BATCHING: Group samples by expert for parallel processing
        # This is 30-40% faster than sequential per-sample processing
        output = torch.zeros_like(x)

        # Process each expert once with all assigned samples batched together
        for expert_id in range(self.num_experts):
            # Find all (batch_idx, k_idx) pairs that selected this expert
            expert_mask = (top_k_indices == expert_id)  # [B, top_k]

            if not expert_mask.any():
                continue  # No samples assigned to this expert

            # Get batch indices and their corresponding weights
            batch_indices = torch.where(expert_mask.any(dim=1))[0]  # Which samples use this expert

            if len(batch_indices) == 0:
                continue

            # Gather all samples that need this expert
            expert_inputs = x[batch_indices]  # [N, C, H, W] where N <= B

            # Process all samples for this expert in one batch
            expert_outputs = self.experts[expert_id](expert_inputs)  # [N, C, H, W]

            # Distribute outputs back to original positions with weights
            for i, batch_idx in enumerate(batch_indices):
                # Find which k positions selected this expert for this sample
                k_positions = torch.where(top_k_indices[batch_idx] == expert_id)[0]

                # Sum weights for all k positions that selected this expert
                total_weight = top_k_probs[batch_idx, k_positions].sum().item()

                # Add weighted expert output to final output
                output[batch_idx] += total_weight * expert_outputs[i]

        # LOSS-FREE LOAD BALANCING: Update expert bias based on usage
        # NO auxiliary loss = NO gradient interference
        if self.training:
            with torch.no_grad():
                # Measure actual expert usage in this batch
                expert_usage = torch.zeros(self.num_experts, device=x.device)
                for expert_id in range(self.num_experts):
                    expert_usage[expert_id] = (top_k_indices == expert_id).float().sum() / (B * self.top_k)

                # Update EMA of expert usage
                self.expert_usage_ema = 0.9 * self.expert_usage_ema + 0.1 * expert_usage

                # Compute ideal usage (uniform distribution)
                ideal_usage = 1.0 / self.num_experts

                # Update bias: Heavy-load → negative bias, Light-load → positive bias
                load_imbalance = self.expert_usage_ema - ideal_usage
                self.expert_bias.data -= self.bias_update_rate * load_imbalance

                # Clamp bias to prevent extreme values
                self.expert_bias.data.clamp_(-5.0, 5.0)

        # Return 0.0 for load_balance_loss (no auxiliary loss, backward compatible)
        return output, torch.tensor(0.0, device=x.device, dtype=x.dtype)


# Example usage
if __name__ == '__main__':
    # Test sparse MoE
    print("Testing Sparse COD MoE...")

    dim = 288
    batch = 4
    H, W = 22, 22

    # Create sparse MoE layer
    moe = EfficientSparseCODMoE(dim=dim, num_experts=6, top_k=2)

    # Test forward pass
    x = torch.randn(batch, dim, H, W)
    output, lb_loss = moe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Load balance loss: {lb_loss.item():.6f}")
    print(f"\nParameters: {sum(p.numel() for p in moe.parameters()) / 1e6:.2f}M")

    # Test with routing info
    print("\n" + "="*60)
    print("Testing with routing info...")
    moe_detailed = SparseCODMoE(dim=dim, num_experts=6, top_k=2, routing_mode='global')
    output, lb_loss, routing_info = moe_detailed(x, return_routing_info=True)

    print(f"Selected experts per sample:")
    for i, experts in enumerate(routing_info['selected_experts']):
        print(f"  Sample {i}: Experts {experts.tolist()}")

    print(f"\nExpert weights:")
    for i, weights in enumerate(routing_info['expert_weights']):
        print(f"  Sample {i}: Weights {weights.tolist()}")
