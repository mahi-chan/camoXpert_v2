"""
Sophisticated Router Network for Model-Level MoE

Analyzes input features to determine which expert architecture
is best suited for the given image type.

Features analyzed:
- Texture complexity (fine vs coarse patterns)
- Edge density (sharp vs smooth boundaries)
- Color distribution (monochrome vs colorful)
- Frequency content (high vs low frequency patterns)
- Context scale (small vs large objects)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SophisticatedRouter(nn.Module):
    """
    Multi-branch router that analyzes different image characteristics
    to intelligently route to the best expert.

    ~8M parameters - sophisticated enough to learn specialization

    Args:
        backbone_dims: Feature dimensions from backbone [64, 128, 320, 512]
        num_experts: Number of expert models (default: 4)
        top_k: How many experts to use (default: 2)
    """
    def __init__(self, backbone_dims=[64, 128, 320, 512], num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Use highest resolution features for detailed analysis
        self.high_dim = backbone_dims[-1]  # 512

        # ============================================================
        # BRANCH 1: Texture Analysis
        # Detects texture complexity and patterns
        # ============================================================
        self.texture_branch = nn.Sequential(
            # Multi-scale texture extraction
            nn.Conv2d(self.high_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # ============================================================
        # BRANCH 2: Edge Complexity Analysis
        # Detects edge density and boundary characteristics
        # ============================================================
        self.edge_branch = nn.Sequential(
            # Edge-sensitive convolutions
            nn.Conv2d(self.high_dim, 256, 3, padding=1, groups=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # ============================================================
        # BRANCH 3: Multi-Scale Context Analysis
        # Analyzes object scale and context complexity
        # ============================================================
        self.context_branch = nn.Sequential(
            # Large receptive field for context
            nn.Conv2d(self.high_dim, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # ============================================================
        # BRANCH 4: Frequency Analysis
        # Detects high/low frequency patterns
        # ============================================================
        self.frequency_branch = nn.Sequential(
            # Small kernels for high-freq, large for low-freq
            nn.Conv2d(self.high_dim, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # ============================================================
        # Global Feature Integration
        # Considers all feature scales
        # ============================================================
        self.multi_scale_integration = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, 64, 1)
            ) for dim in backbone_dims
        ])

        # ============================================================
        # Decision Network
        # Combines all branches to make routing decision
        # ============================================================
        # Total input: 256 + 256 + 256 + 128 + (64*4) = 1152
        total_features = 256 + 256 + 256 + 128 + (64 * len(backbone_dims))

        self.decision_network = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_experts)
        )

        # Temperature parameter for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features):
        """
        Args:
            features: List of backbone features [f1, f2, f3, f4]
                     where f4 is the highest-level feature

        Returns:
            expert_weights: [B, num_experts] - Probability distribution over experts
            top_k_indices: [B, top_k] - Indices of selected experts
            top_k_weights: [B, top_k] - Weights for selected experts (sum to 1)
        """
        B = features[-1].shape[0]
        highest_features = features[-1]  # [B, 512, H, W]

        # ============================================================
        # Extract characteristics from different branches
        # ============================================================
        texture_feat = self.texture_branch(highest_features).view(B, -1)  # [B, 256]
        edge_feat = self.edge_branch(highest_features).view(B, -1)        # [B, 256]
        context_feat = self.context_branch(highest_features).view(B, -1)  # [B, 256]
        freq_feat = self.frequency_branch(highest_features).view(B, -1)   # [B, 128]

        # Integrate multi-scale features
        multi_scale_feats = []
        for feat, integrator in zip(features, self.multi_scale_integration):
            ms_feat = integrator(feat).view(B, -1)  # [B, 64]
            multi_scale_feats.append(ms_feat)
        multi_scale_feat = torch.cat(multi_scale_feats, dim=1)  # [B, 256]

        # ============================================================
        # Combine all branches
        # ============================================================
        combined = torch.cat([
            texture_feat,      # 256
            edge_feat,         # 256
            context_feat,      # 256
            freq_feat,         # 128
            multi_scale_feat   # 256
        ], dim=1)  # [B, 1152]

        # ============================================================
        # Make routing decision
        # ============================================================
        logits = self.decision_network(combined)  # [B, num_experts]

        # Apply temperature scaling for smoother or sharper decisions
        # Higher temperature = more uniform, Lower = more peaked
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        logits = logits / temp

        # Get probability distribution over experts
        expert_probs = F.softmax(logits, dim=1)  # [B, num_experts]

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(expert_probs, self.top_k, dim=1)

        # Renormalize top-k probabilities to sum to 1
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-8)

        return expert_probs, top_k_indices, top_k_weights

    def get_expert_usage_stats(self, expert_probs):
        """
        Analyze which experts are being used

        Args:
            expert_probs: [B, num_experts]

        Returns:
            Dictionary with usage statistics
        """
        with torch.no_grad():
            avg_probs = expert_probs.mean(dim=0)  # [num_experts]
            max_expert = expert_probs.argmax(dim=1)  # [B]

            stats = {
                'avg_expert_probs': avg_probs.cpu().numpy(),
                'expert_selection_counts': torch.bincount(max_expert, minlength=self.num_experts).cpu().numpy(),
                'entropy': -(expert_probs * torch.log(expert_probs + 1e-8)).sum(dim=1).mean().item()
            }

        return stats


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the router
    print("Testing Sophisticated Router...")

    router = SophisticatedRouter(
        backbone_dims=[64, 128, 320, 512],
        num_experts=4,
        top_k=2
    )

    print(f"Router parameters: {count_parameters(router) / 1e6:.1f}M")

    # Create dummy features
    features = [
        torch.randn(2, 64, 112, 112),
        torch.randn(2, 128, 56, 56),
        torch.randn(2, 320, 28, 28),
        torch.randn(2, 512, 14, 14)
    ]

    expert_probs, top_k_indices, top_k_weights = router(features)

    print(f"Expert probabilities shape: {expert_probs.shape}")
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Top-k weights shape: {top_k_weights.shape}")
    print(f"\nExample routing:")
    print(f"Expert probs: {expert_probs[0]}")
    print(f"Selected experts: {top_k_indices[0]}")
    print(f"Expert weights: {top_k_weights[0]}")
    print(f"Weights sum to: {top_k_weights[0].sum()}")

    stats = router.get_expert_usage_stats(expert_probs)
    print(f"\nUsage stats:")
    print(f"Average expert probabilities: {stats['avg_expert_probs']}")
    print(f"Routing entropy: {stats['entropy']:.3f}")

    print("\n✓ Router test passed!")
