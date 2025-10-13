"""
Embedding space experiments to address the arbitrary embedding distribution problem.

The core issue: Frozen LLMs expect embeddings from their discrete token vocabulary,
but learned encoders produce arbitrary continuous embeddings. This module implements
various techniques to bridge this distribution mismatch.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticalMatcher(nn.Module):
    """
    Experiment 1B: Statistical Matching
    Matches statistical properties (mean, variance, RMS) of learned embeddings to real token embeddings.
    """

    def __init__(self, mode: str = "per_example_rms"):
        """
        Args:
            mode:
                - "per_example_rms": Match RMS per example
                - "batch_distribution": Match mean+std across batch
                - "none": Pass through (baseline)
        """
        super().__init__()
        self.mode = mode

    def forward(self, embeddings: torch.Tensor, target_stats: Optional[dict] = None) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, seq, d_model] - learned embeddings
            target_stats: Optional dict with 'mean', 'std', 'rms' from real token embeddings

        Returns:
            Calibrated embeddings
        """
        if self.mode == "none":
            return embeddings

        if self.mode == "per_example_rms":
            # Match RMS per example (most conservative)
            current_rms = embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt()
            if target_stats is not None and 'rms' in target_stats:
                target_rms = target_stats['rms']  # [batch, seq, 1]
            else:
                # Fallback: use empirical target (Llama ~30-50 range)
                target_rms = torch.ones_like(current_rms) * 35.0

            scale = target_rms / (current_rms + 1e-8)
            return embeddings * scale

        elif self.mode == "batch_distribution":
            # Match mean and std across entire batch
            current_mean = embeddings.mean()
            current_std = embeddings.std()

            target_mean = target_stats.get('mean', 0.0) if target_stats else 0.0
            target_std = target_stats.get('std', 35.0) if target_stats else 35.0

            # Standardize then rescale
            embeddings = (embeddings - current_mean) / (current_std + 1e-8)
            embeddings = embeddings * target_std + target_mean
            return embeddings

        return embeddings


class NearestKProjector(nn.Module):
    """
    Experiment 1A: Nearest-K Embedding Projection
    Projects learned embeddings onto the manifold of real token embeddings.
    """

    def __init__(self, vocab_embeddings: torch.Tensor, k: int = 5, alpha: float = 0.5):
        """
        Args:
            vocab_embeddings: [vocab_size, d_model] - frozen token embedding table
            k: Number of nearest neighbors to project onto
            alpha: Interpolation factor (0=pure projection, 1=original embedding)
        """
        super().__init__()
        self.register_buffer('vocab_embeddings', vocab_embeddings)
        self.k = k
        self.alpha = alpha

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, seq, d_model]
        Returns:
            Projected embeddings (convex combination of K nearest vocab embeddings)
        """
        B, S, D = embeddings.shape
        embeddings_flat = embeddings.view(-1, D)  # [B*S, D]

        # Compute cosine similarity to all vocab embeddings
        embeddings_norm = F.normalize(embeddings_flat, dim=-1)
        vocab_norm = F.normalize(self.vocab_embeddings, dim=-1)
        sim = embeddings_norm @ vocab_norm.T  # [B*S, vocab_size]

        # Get top-K nearest neighbors
        topk_sim, topk_idx = sim.topk(self.k, dim=-1)  # [B*S, K]
        topk_weights = F.softmax(topk_sim * 10.0, dim=-1)  # Temperature=0.1 for sharper

        # Weighted combination of top-K embeddings
        topk_embeddings = self.vocab_embeddings[topk_idx]  # [B*S, K, D]
        projected = (topk_weights.unsqueeze(-1) * topk_embeddings).sum(dim=1)  # [B*S, D]

        # Interpolate between original and projected
        result = self.alpha * embeddings_flat + (1 - self.alpha) * projected

        return result.view(B, S, D)


class SoftVocabInterpolator(nn.Module):
    """
    Experiment 1C: Soft Vocabulary Interpolation
    Forces embeddings to be explicit convex combinations of real token embeddings.
    """

    def __init__(self, vocab_embeddings: torch.Tensor, temperature: float = 1.0):
        """
        Args:
            vocab_embeddings: [vocab_size, d_model]
            temperature: Softmax temperature (lower = more discrete)
        """
        super().__init__()
        self.register_buffer('vocab_embeddings', vocab_embeddings)
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq, vocab_size] - raw logits over vocabulary
        Returns:
            Soft embeddings as weighted combination of vocab
        """
        weights = F.softmax(logits / self.temperature, dim=-1)  # [B, S, V]
        embeddings = weights @ self.vocab_embeddings  # [B, S, D]
        return embeddings


class AnchorPlusOffset(nn.Module):
    """
    Experiment 4A: Anchor Token + Small Offset
    Each latent is expressed as nearest token + small continuous residual.
    """

    def __init__(self, vocab_embeddings: torch.Tensor, epsilon: float = 0.1):
        """
        Args:
            vocab_embeddings: [vocab_size, d_model]
            epsilon: Maximum magnitude of offset (relative to embedding norm)
        """
        super().__init__()
        self.register_buffer('vocab_embeddings', vocab_embeddings)
        self.epsilon = epsilon

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [batch, seq, d_model] - raw continuous embeddings
        Returns:
            anchored_embeddings: [batch, seq, d_model] - anchor + clipped offset
            anchor_ids: [batch, seq] - nearest token IDs (for logging)
        """
        B, S, D = embeddings.shape
        embeddings_flat = embeddings.view(-1, D)

        # Find nearest anchor token (cosine similarity)
        embeddings_norm = F.normalize(embeddings_flat, dim=-1)
        vocab_norm = F.normalize(self.vocab_embeddings, dim=-1)
        sim = embeddings_norm @ vocab_norm.T  # [B*S, vocab_size]
        anchor_ids = sim.argmax(dim=-1)  # [B*S]
        anchor_embeddings = self.vocab_embeddings[anchor_ids]  # [B*S, D]

        # Compute offset and clip to epsilon
        offset = embeddings_flat - anchor_embeddings
        offset_norm = offset.norm(dim=-1, keepdim=True)
        anchor_norm = anchor_embeddings.norm(dim=-1, keepdim=True)
        max_offset = self.epsilon * anchor_norm

        # Clip offset magnitude
        clipped_offset = offset * torch.clamp(max_offset / (offset_norm + 1e-8), max=1.0)

        result = anchor_embeddings + clipped_offset
        return result.view(B, S, D), anchor_ids.view(B, S)


class SoftCodebook(nn.Module):
    """
    Experiment 3B: Soft Codebook (VQ-VAE style with Gumbel-Softmax)
    Discrete bottleneck but differentiable via Gumbel-Softmax trick.
    """

    def __init__(self, d_model: int, codebook_size: int = 1024, temperature: float = 1.0):
        """
        Args:
            d_model: Embedding dimension
            codebook_size: Number of discrete codes
            temperature: Gumbel-Softmax temperature (anneal during training)
        """
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model) * 0.02)
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [batch, seq, d_model] - input embeddings
            hard: If True, use hard (one-hot) sampling; else soft
        Returns:
            quantized: [batch, seq, d_model] - quantized embeddings
            logits: [batch, seq, codebook_size] - logits for regularization
        """
        B, S, D = embeddings.shape

        # Compute logits (negative distance to codebook entries)
        embeddings_flat = embeddings.view(-1, D)  # [B*S, D]
        distances = torch.cdist(embeddings_flat, self.codebook)  # [B*S, codebook_size]
        logits = -distances  # Higher logit = closer to codebook entry

        # Gumbel-Softmax sampling
        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
        else:
            # At inference, use hard argmax
            weights = F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()

        # Quantize
        quantized = weights @ self.codebook  # [B*S, D]

        return quantized.view(B, S, D), logits.view(B, S, -1)


class RandomInterpolationBaseline(nn.Module):
    """
    Experiment 7A: Random Token Interpolation Baseline (Diagnostic)
    Replace encoder with random convex combinations of real tokens.
    Tests if frozen LLM can handle ANY arbitrary valid embeddings.
    """

    def __init__(self, vocab_embeddings: torch.Tensor, k: int = 3):
        """
        Args:
            vocab_embeddings: [vocab_size, d_model]
            k: Number of tokens to interpolate per position
        """
        super().__init__()
        self.register_buffer('vocab_embeddings', vocab_embeddings)
        self.k = k

    def forward(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Generate random interpolations of real token embeddings.

        Args:
            batch_size: Number of examples
            seq_len: Sequence length
        Returns:
            embeddings: [batch_size, seq_len, d_model] - random interpolations
        """
        device = self.vocab_embeddings.device
        vocab_size = self.vocab_embeddings.size(0)

        # Sample K random tokens per position
        random_ids = torch.randint(0, vocab_size, (batch_size, seq_len, self.k), device=device)
        random_embeddings = self.vocab_embeddings[random_ids]  # [B, S, K, D]

        # Random convex combination weights
        weights = F.softmax(torch.randn(batch_size, seq_len, self.k, device=device), dim=-1)

        # Weighted sum
        interpolated = (weights.unsqueeze(-1) * random_embeddings).sum(dim=2)  # [B, S, D]

        return interpolated


class EmbeddingDiscriminator(nn.Module):
    """
    Experiment 5A: Adversarial Discriminator
    Discriminates between real token embeddings and learned latent embeddings.
    Used to train encoder adversarially to fool the discriminator.
    """

    def __init__(self, d_model: int, hidden_dim: int = 512):
        """
        Args:
            d_model: Embedding dimension
            hidden_dim: Hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, seq, d_model]
        Returns:
            logits: [batch, seq] - logit for "real" (vs fake/latent)
        """
        return self.net(embeddings).squeeze(-1)


class MixedPrefixAdapter(nn.Module):
    """
    Experiment 4B: Mixed Text + Latent Prefix
    Combines compressed latent embeddings with some real text token embeddings.
    """

    def __init__(self, latent_len: int, text_len: int, mode: str = "concat"):
        """
        Args:
            latent_len: Number of latent positions
            text_len: Number of real text token positions
            mode: "concat" or "interleave"
        """
        super().__init__()
        self.latent_len = latent_len
        self.text_len = text_len
        self.mode = mode

    def forward(self,
                latent_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_embeddings: [batch, latent_len, d_model]
            text_embeddings: [batch, text_len, d_model]
        Returns:
            mixed: [batch, latent_len + text_len, d_model]
        """
        if self.mode == "concat":
            # Latent first, then text
            return torch.cat([latent_embeddings, text_embeddings], dim=1)
        elif self.mode == "interleave":
            # Alternate between latent and text
            # This is more complex, for now just concat
            return torch.cat([latent_embeddings, text_embeddings], dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def get_vocab_embedding_stats(model: nn.Module) -> dict:
    """
    Extract statistical properties of a model's token embedding table.

    Args:
        model: HuggingFace model with get_input_embeddings()
    Returns:
        dict with 'mean', 'std', 'rms', 'embeddings' tensor
    """
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight  # [vocab_size, d_model]
        stats = {
            'embeddings': embeddings,
            'mean': embeddings.mean().item(),
            'std': embeddings.std().item(),
            'rms': embeddings.pow(2).mean(dim=-1).sqrt().mean().item(),
            'per_token_rms': embeddings.pow(2).mean(dim=-1, keepdim=True).sqrt(),  # [vocab_size, 1]
        }
    return stats


def apply_experiment(embeddings: torch.Tensor,
                     experiment: str,
                     vocab_stats: Optional[dict] = None,
                     modules: Optional[dict] = None) -> torch.Tensor:
    """
    Convenience function to apply an experiment transformation.

    Args:
        embeddings: [batch, seq, d_model] - learned embeddings
        experiment: Name of experiment (e.g., "rms_matching", "nearest_k", etc.)
        vocab_stats: Dict from get_vocab_embedding_stats()
        modules: Dict of experiment modules (pre-initialized)

    Returns:
        Transformed embeddings
    """
    if experiment == "none" or experiment == "baseline":
        return embeddings

    if modules is None:
        modules = {}

    # Statistical matching
    if experiment == "rms_matching":
        if 'rms_matcher' not in modules:
            modules['rms_matcher'] = StatisticalMatcher(mode="per_example_rms")
        target_stats = None
        if vocab_stats:
            # Compute per-position RMS targets
            B, S, D = embeddings.shape
            target_rms = torch.ones(B, S, 1, device=embeddings.device) * vocab_stats['rms']
            target_stats = {'rms': target_rms}
        return modules['rms_matcher'](embeddings, target_stats)

    if experiment == "batch_distribution":
        if 'dist_matcher' not in modules:
            modules['dist_matcher'] = StatisticalMatcher(mode="batch_distribution")
        return modules['dist_matcher'](embeddings, vocab_stats)

    # Projection methods
    if experiment == "nearest_k":
        if 'nearest_k' not in modules and vocab_stats:
            modules['nearest_k'] = NearestKProjector(vocab_stats['embeddings'], k=5, alpha=0.5)
            modules['nearest_k'].to(embeddings.device)
        if 'nearest_k' in modules:
            return modules['nearest_k'](embeddings)

    if experiment == "anchor_offset":
        if 'anchor_offset' not in modules and vocab_stats:
            modules['anchor_offset'] = AnchorPlusOffset(vocab_stats['embeddings'], epsilon=0.1)
            modules['anchor_offset'].to(embeddings.device)
        if 'anchor_offset' in modules:
            result, _ = modules['anchor_offset'](embeddings)
            return result

    # If experiment not recognized, return unchanged
    return embeddings
