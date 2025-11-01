"""Model architectures for cross-model alignment."""

import math
import torch
import torch.nn as nn


class LearnedProjection(nn.Module):
    """
    Learned linear projection to handle dimension mismatches between models.

    Used for cross-model communication when models have different hidden dimensions.
    For example, Llama 3.2 3B (3072) → Llama 3.1 8B (4096).

    References:
        - Ramesh & Li, "Communicating Activations Between Language Model Agents"
          arXiv:2501.14082 (ICML 2025)
          Uses learned projection W trained on C4 dataset to handle dimension mismatch
    """

    def __init__(self, source_dim, target_dim):
        """
        Args:
            source_dim: Source model's hidden dimension (e.g., 3072 for Llama 3.2 3B)
            target_dim: Target model's hidden dimension (e.g., 4096 for Llama 3.1 8B)
        """
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.proj = nn.Linear(source_dim, target_dim, bias=False)
        # Kaiming initialization for better gradient flow
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, x):
        """
        Project from source dimension to target dimension.

        Args:
            x: Tensor of shape [..., source_dim]

        Returns:
            Tensor of shape [..., target_dim]
        """
        return self.proj(x)
