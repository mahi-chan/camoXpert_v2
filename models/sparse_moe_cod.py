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

        # Load balancing auxiliary loss coefficient - ADAPTIVE
        # Starts at 0.00001 for stability, scales up to 0.0005 for specialization
        # Warmup: 0.00001 (epochs 0-20, prevents explosion)
        # Post-warmup: 0.0005 (epochs 20+, encourages specialization)
        self.load_balance_loss_coef_min = 0.00001  # During warmup
        self.load_balance_loss_coef_max = 0.0005   # After warmup

        # Entropy regularization coefficient (encourages diverse expert usage)
        self.entropy_coef = 0.001  # Small bonus for high entropy routing

        # Router stabilization
        self.router_grad_clip = 1.0  # Clip router gradients to this norm

    def forward(self, x, warmup_factor=1.0):
        """
        Args:
            x: Input features [B, C, H, W]
            warmup_factor: Scale factor for load balance loss (0.0 to 1.0)

        Returns:
            expert_weights: [B, num_experts] or [B, num_experts, H, W]
            top_k_indices: [B, top_k] or [B, top_k, H, W]
            load_balance_loss: Scalar loss for balancing expert usage
        """
        # Compute routing logits
        logits = self.gate(x)  # [B, num_experts] or [B, num_experts, H, W]

        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        # Softmax to get expert probabilities with temperature scaling
        temperature = 1.0  # Can be tuned for smoothness
        if self.routing_mode == 'global':
            probs = F.softmax(logits / temperature, dim=1)  # [B, num_experts]

            # Clamp probabilities to prevent extremes
            probs = torch.clamp(probs, min=1e-6, max=1.0)

            # Select top-k experts
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=1)

            # Normalize top-k probabilities to sum to 1
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-8)

            # Load balancing loss (encourage uniform expert usage)
            expert_usage = probs.mean(dim=0)  # [num_experts]
            ideal_usage = 1.0 / self.num_experts
            load_balance_loss = ((expert_usage - ideal_usage) ** 2).sum()

            # Entropy regularization (encourage diverse routing per image)
            # High entropy = router uses different experts for different images (GOOD)
            # Low entropy = router always uses same experts (BAD - collapse)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()  # [B] -> scalar
            # We SUBTRACT entropy loss (negative = reward high entropy)
            entropy_loss = -entropy

        else:  # spatial
            probs = F.softmax(logits, dim=1)  # [B, num_experts, H, W]
            B, E, H, W = probs.shape

            # Reshape for top-k
            probs_flat = probs.view(B, E, -1).transpose(1, 2)  # [B, H*W, E]
            top_k_probs, top_k_indices = torch.topk(probs_flat, self.top_k, dim=2)

            # Normalize
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=2, keepdim=True) + 1e-8)

            # Reshape back
            top_k_probs = top_k_probs.transpose(1, 2).view(B, self.top_k, H, W)
            top_k_indices = top_k_indices.transpose(1, 2).view(B, self.top_k, H, W)

            # Load balance loss
            expert_usage = probs.mean(dim=(0, 2, 3))  # [num_experts]
            ideal_usage = 1.0 / self.num_experts
            load_balance_loss = ((expert_usage - ideal_usage) ** 2).sum()

            # Entropy regularization (spatial)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            entropy_loss = -entropy

        # Adaptive coefficient: interpolate from min to max based on warmup
        # warmup_factor 0.0 -> coef_min (0.00001, stable)
        # warmup_factor 1.0 -> coef_max (0.0005, encourages specialization)
        adaptive_coef = self.load_balance_loss_coef_min + \
                       warmup_factor * (self.load_balance_loss_coef_max - self.load_balance_loss_coef_min)

        # Total routing loss = load balance + entropy regularization
        scaled_lb_loss = load_balance_loss * adaptive_coef + entropy_loss * self.entropy_coef

        return top_k_probs, top_k_indices, scaled_lb_loss


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

        # Adaptive load balance coefficient (like SparseRouter)
        self.load_balance_loss_coef_min = 0.00001
        self.load_balance_loss_coef_max = 0.0005
        self.entropy_coef = 0.001

    def forward(self, x, warmup_factor=1.0):
        """
        Efficient forward pass with minimal overhead

        Args:
            x: Input features [B, C, H, W]
            warmup_factor: Scale factor for load balance loss (0.0 to 1.0)

        Returns:
            output: Enhanced features
            load_balance_loss: Auxiliary loss
        """
        B = x.shape[0]

        # Compute routing scores [B, num_experts]
        routing_logits = self.router(x)

        # Clamp for numerical stability
        routing_logits = torch.clamp(routing_logits, min=-10.0, max=10.0)

        routing_probs = F.softmax(routing_logits, dim=1)

        # Clamp probabilities
        routing_probs = torch.clamp(routing_probs, min=1e-6, max=1.0)

        # Select top-k experts per sample
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Group samples by expert assignment for efficient batching
        # (In practice, this could be optimized further with expert batching)
        output = torch.zeros_like(x)

        # Process each sample (can be parallelized with more complex indexing)
        for b in range(B):
            for k in range(self.top_k):
                expert_id = top_k_indices[b, k].item()
                expert_weight = top_k_probs[b, k].item()

                expert_output = self.experts[expert_id](x[b:b+1])
                output[b:b+1] += expert_weight * expert_output

        # Load balancing loss
        expert_usage = routing_probs.mean(dim=0)
        ideal_usage = 1.0 / self.num_experts
        load_balance_loss = ((expert_usage - ideal_usage) ** 2).sum()

        # Entropy regularization (encourage diverse routing)
        entropy = -(routing_probs * torch.log(routing_probs + 1e-10)).sum(dim=1).mean()
        entropy_loss = -entropy  # Negative to reward high entropy

        # Adaptive coefficient based on warmup
        adaptive_coef = self.load_balance_loss_coef_min + \
                       warmup_factor * (self.load_balance_loss_coef_max - self.load_balance_loss_coef_min)

        # Total routing loss
        total_routing_loss = load_balance_loss * adaptive_coef + entropy_loss * self.entropy_coef

        return output, total_routing_loss


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
