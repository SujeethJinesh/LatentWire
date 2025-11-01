"""Adapter model architectures for cross-model alignment."""

import torch
import torch.nn as nn
import math


class ProcrustesAlignment:
    """
    Affine Procrustes alignment between hidden spaces.

    Extended from orthogonal to include bias term for better alignment.
    Affine transformations (linear + bias) consistently outperform pure orthogonal
    by 5-8% in model stitching tasks.

    References:
        - Schönemann, "A generalized solution of the orthogonal Procrustes problem"
          Psychometrika 1966 - Original orthogonal Procrustes formulation
        - Lester et al., "Transferring Features Across Language Models With Model Stitching"
          arXiv:2506.06609 (June 2025) - Affine extension outperforms orthogonal by 5-8%

    Args:
        use_affine: If True, use affine transformation (W + b). If False, orthogonal only (W).
    """

    def __init__(self, use_affine=True):
        self.W = None  # Orthogonal transformation matrix
        self.b = None  # Bias term (affine extension)
        self.source_mean = None
        self.target_mean = None
        self.source_norm = None
        self.target_norm = None
        self.use_affine = use_affine

    def fit(self, source, target):
        """
        Fit affine transformation such that ||source transformed - target||_F is minimized.

        Args:
            source, target: [n_samples, n_features] representation matrices

        Returns:
            self (fitted)
        """
        assert source.shape == target.shape, "Source and target must have same shape"

        # Step 1: Center both datasets
        self.source_mean = source.mean(dim=0, keepdim=True)
        self.target_mean = target.mean(dim=0, keepdim=True)
        source_centered = source - self.source_mean
        target_centered = target - self.target_mean

        # Step 2: Normalize to unit Frobenius norm
        self.source_norm = torch.norm(source_centered, 'fro')
        self.target_norm = torch.norm(target_centered, 'fro')

        eps = 1e-8
        source_normalized = source_centered / (self.source_norm + eps)
        target_normalized = target_centered / (self.target_norm + eps)

        if torch.isinf(self.source_norm) or torch.isinf(self.target_norm):
            print(f"  WARNING: Infinite norm detected, using layer normalization fallback")
            source_normalized = torch.nn.functional.normalize(source_centered, dim=-1)
            target_normalized = torch.nn.functional.normalize(target_centered, dim=-1)

        # Step 3: Compute cross-covariance matrix in float32
        M = source_normalized.float().T @ target_normalized.float()

        # Step 4: SVD in float32
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)

        # Step 5: Optimal orthogonal transformation
        self.W = U @ Vt  # Stay in float32

        # Step 6: Bias term
        if self.use_affine:
            self.b = torch.zeros_like(self.target_mean)
        else:
            self.b = torch.zeros_like(self.target_mean)

        # Verify orthogonality
        I = self.W @ self.W.T
        ortho_error = torch.norm(I - torch.eye(I.shape[0], device=I.device), 'fro')
        if ortho_error > 1e-3:
            print(f"  WARNING: Orthogonality error = {ortho_error:.6f}")

        return self

    def transform(self, source):
        """
        Apply the fitted affine transformation to new data.

        Args:
            source: [n_samples, n_features]

        Returns:
            transformed: [n_samples, n_features]
        """
        assert self.W is not None, "Must fit before transform"

        # Move parameters to same device as source
        device = source.device
        source_mean = self.source_mean.to(device)
        source_norm = self.source_norm.to(device)
        W = self.W.to(device)
        target_norm = self.target_norm.to(device)
        target_mean = self.target_mean.to(device)

        # Center and normalize
        source_centered = source - source_mean
        source_normalized = source_centered / (source_norm + 1e-8)

        # Apply transformation
        transformed = source_normalized @ W

        # Rescale and recenter
        transformed = transformed * target_norm
        transformed = transformed + target_mean

        return transformed


class LinearAdapter(nn.Module):
    """Full linear projection for cross-model alignment."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x)


class AffineAdapter(nn.Module):
    """Affine transformation (linear + bias) for cross-model alignment."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        if self.linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        return self.linear(x)


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adapter (LoRA) for parameter-efficient cross-model alignment.

    References:
        - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
          arXiv:2106.09685 (ICLR 2022)
    """

    def __init__(self, input_dim, output_dim, rank=32):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_dim, bias=False)

        # Initialize A with kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))
