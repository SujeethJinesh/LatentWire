#!/usr/bin/env python
# telepathy/latent_bridge_v15.py
"""
Phase 15: FSQ-Telepathy (Finite Scalar Quantization Bridge)

THE FIX FOR MANIFOLD MISMATCH:

Previous failures:
- Regression (V7): Blurry averages (semantic drift)
- Diffusion Global (V12): Converged but lost details
- Diffusion Cross-Attn (V13-14): Failed to converge
- VQ (V15 attempt 1): Codebook collapse (perplexity → 1)

FSQ Solution (Google Research, 2023):
1. No learned codebook = No collapse possible
2. Each dimension independently quantized to L levels
3. Effective codebook = product of levels (e.g., 8^8 = 16M codes)
4. Straight-through estimator for gradients

Architecture:
    Llama Hidden -> Normalizer -> Perceiver -> FSQ Bottleneck -> Scale -> Mistral

Reference: "Finite Scalar Quantization: VQ-VAE Made Simple"
https://arxiv.org/abs/2309.15505
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) - Google Research, 2023

    No learned codebook. Each dimension independently quantized to fixed levels.
    Effective codebook size = product of all levels.

    Key advantages over VQ:
    - No codebook collapse (no codebook to collapse!)
    - Simpler training (no commitment loss, no EMA updates)
    - Deterministic quantization
    """

    def __init__(self, levels, input_dim):
        """
        Args:
            levels: List of quantization levels per dimension, e.g., [8,8,8,8,8,8,8,8]
                    Effective codebook size = product(levels)
            input_dim: Dimension of input vectors (e.g., 4096)
        """
        super().__init__()
        self.levels = levels
        self.fsq_dim = len(levels)
        self.input_dim = input_dim

        # Register levels as buffer for device handling
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))

        # Compute effective codebook size
        self.codebook_size = math.prod(levels)

        # Project input_dim -> fsq_dim -> input_dim
        self.proj_down = nn.Linear(input_dim, self.fsq_dim)
        self.proj_up = nn.Linear(self.fsq_dim, input_dim)

        # Initialize projections for stability
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)
        nn.init.zeros_(self.proj_down.bias)
        nn.init.zeros_(self.proj_up.bias)

        print(f"[FSQ] {self.fsq_dim} dimensions, levels={levels}")
        print(f"[FSQ] Effective codebook size: {self.codebook_size:,}")

    def _bound(self, z):
        """Bound z to [-1, 1] using tanh."""
        return torch.tanh(z)

    def _quantize(self, z):
        """
        Quantize each dimension to its respective number of levels.

        For L levels, we quantize to: {-(L-1)/2, ..., -1, 0, 1, ..., (L-1)/2} / ((L-1)/2)
        which gives values in [-1, 1].
        """
        # z is bounded to [-1, 1]
        # Scale to [0, L-1], round, scale back to [-1, 1]
        half_levels = (self._levels - 1) / 2  # [D]

        # Scale from [-1, 1] to [-half, half]
        scaled = z * half_levels  # [B, K, D]

        # Round to nearest integer (STE: gradient flows through)
        quantized_int = torch.round(scaled)

        # Straight-through estimator
        quantized_int = scaled + (quantized_int - scaled).detach()

        # Scale back to [-1, 1]
        quantized = quantized_int / half_levels

        return quantized

    def forward(self, inputs):
        """
        Args:
            inputs: [B, K, input_dim] continuous vectors

        Returns:
            quantized: [B, K, input_dim] quantized vectors (projected back up)
            aux_loss: Always 0 (FSQ has no auxiliary loss)
            codebook_usage: Approximate measure of code diversity
        """
        input_shape = inputs.shape
        dtype = inputs.dtype

        # Project down to FSQ dimension (keep in same dtype as weights)
        z = self.proj_down(inputs)  # [B, K, fsq_dim]

        # Bound to [-1, 1] - use float32 for precision in quantization
        z_float = z.float()
        z_bounded = self._bound(z_float)

        # Quantize each dimension
        z_quantized = self._quantize(z_bounded)

        # Project back up to input dimension (convert back to weight dtype)
        out = self.proj_up(z_quantized.to(z.dtype))  # [B, K, input_dim]

        # Compute approximate codebook usage (how many unique codes in batch)
        # Convert quantized values to integer indices for diversity estimation
        with torch.no_grad():
            half_levels = (self._levels - 1) / 2
            indices = ((z_quantized * half_levels) + half_levels).long()  # [B, K, D]
            # Flatten to get unique code count (approximate)
            flat_indices = indices.view(-1, self.fsq_dim)
            # Simple diversity metric: unique rows / total rows
            unique_codes = torch.unique(flat_indices, dim=0).shape[0]
            total_codes = flat_indices.shape[0]
            diversity = unique_codes / total_codes  # 0 to 1

        # FSQ has no auxiliary loss (huge advantage over VQ!)
        aux_loss = torch.tensor(0.0, device=inputs.device, dtype=torch.float32)

        return out.to(dtype), aux_loss, diversity


class StatisticalNormalizer(nn.Module):
    """Normalizes Llama hidden states to Mistral's distribution."""
    def __init__(self, stats_path=None):
        super().__init__()
        if stats_path:
            try:
                stats = torch.load(stats_path, map_location="cpu", weights_only=True)
                self.register_buffer("l_mean", stats["l_mean"].float())
                self.register_buffer("l_std", stats["l_std"].float())
                self.register_buffer("m_mean", stats["m_mean"].float())
                self.register_buffer("m_std", stats["m_std"].float())
                self.has_stats = True
            except Exception as e:
                print(f"[StatisticalNormalizer] Could not load stats: {e}")
                self.has_stats = False
        else:
            self.has_stats = False

        if not self.has_stats:
            # Identity transform if no stats
            self.register_buffer("l_mean", torch.tensor(0.0))
            self.register_buffer("l_std", torch.tensor(1.0))
            self.register_buffer("m_mean", torch.tensor(0.0))
            self.register_buffer("m_std", torch.tensor(1.0))

    def forward(self, x):
        # Normalize from Llama distribution to Mistral distribution
        x_float = x.float()
        normalized = ((x_float - self.l_mean) / (self.l_std + 1e-8)) * self.m_std + self.m_mean
        return normalized.to(x.dtype)


class PerceiverResampler(nn.Module):
    """
    Perceiver-style cross-attention resampler.

    Compresses variable-length input to fixed-length latent sequence.
    """
    def __init__(self, src_dim, tgt_dim, num_latents=64, heads=8, depth=4):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)

        # Project source to target dimension
        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln1": nn.LayerNorm(tgt_dim),
                "self_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln2": nn.LayerNorm(tgt_dim),
                "ffn": nn.Sequential(
                    nn.Linear(tgt_dim, 4 * tgt_dim),
                    nn.GELU(),
                    nn.Linear(4 * tgt_dim, tgt_dim)
                ),
                "ln3": nn.LayerNorm(tgt_dim)
            }) for _ in range(depth)
        ])

        print(f"[PerceiverResampler] {num_latents} latents, {depth} layers, {heads} heads")

    def forward(self, src_hidden, src_mask=None):
        """
        Args:
            src_hidden: [B, T, src_dim] - Llama hidden states
            src_mask: [B, T] - attention mask (1=valid, 0=pad)

        Returns:
            [B, num_latents, tgt_dim] - compressed representation
        """
        B = src_hidden.shape[0]

        # Project source to target dimension
        keys = self.input_proj(src_hidden.to(self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype))

        # Expand latent queries for batch
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)

        # Key padding mask for attention (True = ignore)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            # Cross-attention to source
            x_norm = layer["ln1"](x)
            attn_out, _ = layer["cross_attn"](
                query=x_norm, key=keys, value=keys,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            x = x + attn_out

            # Self-attention among latents
            x_norm = layer["ln2"](x)
            attn_out, _ = layer["self_attn"](
                query=x_norm, key=x_norm, value=x_norm,
                need_weights=False
            )
            x = x + attn_out

            # FFN
            x = x + layer["ffn"](layer["ln3"](x))

        return x


class LatentBridgeV15(nn.Module):
    """
    Phase 15: FSQ-Telepathy Bridge (Finite Scalar Quantization)

    Architecture:
        Llama Hidden -> StatisticalNormalizer -> PerceiverResampler
                     -> FSQ Bottleneck -> Output Scale -> Mistral Embeddings

    Key Features:
    - FSQ: No codebook = No collapse possible
    - Discrete bottleneck prevents blurry/drifted outputs
    - 8^8 = 16M effective codes (vs 4096 VQ codes)
    - 1-step inference (no diffusion iteration)
    - No auxiliary VQ loss needed
    """
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

        # Statistical normalization (Llama -> Mistral distribution)
        stats_path = getattr(args, 'stats_path', None)
        self.normalizer = StatisticalNormalizer(stats_path)

        # Perceiver resampler (compress to fixed length)
        num_latents = getattr(args, 'soft_tokens', 128)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 4)
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # FSQ Bottleneck - 32 dimensions with 5 levels each
        # Previous 8-dim config collapsed to div=0 (too aggressive compression: 4096->8)
        # 32 dims preserves more diversity while still being discrete
        # Effective codes: 5^32 ≈ 2.3 × 10^22
        fsq_levels = getattr(args, 'fsq_levels', [5] * 32)  # 32 dims, 5 levels each
        self.fsq = FSQ(levels=fsq_levels, input_dim=tgt_dim)

        # Output scale to match Mistral embedding magnitude
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        print(f"[LatentBridgeV15] FSQ-Telepathy Bridge")
        print(f"  - src_dim: {src_dim}")
        print(f"  - tgt_dim: {tgt_dim}")
        print(f"  - num_latents: {num_latents}")
        print(f"  - fsq_levels: {fsq_levels}")
        print(f"  - effective_codebook: {self.fsq.codebook_size:,}")
        print(f"  - target_rms: {target_rms:.4f}")

    def forward(self, src_hidden, src_mask=None):
        """
        Args:
            src_hidden: [B, T, src_dim] - Llama hidden states
            src_mask: [B, T] - attention mask

        Returns:
            out: [B, K, tgt_dim] - quantized soft tokens for Mistral
            aux_loss: scalar - Always 0 for FSQ (no auxiliary loss needed!)
            diversity: scalar - Code diversity metric (0-1)
        """
        # 1. Normalize Llama -> Mistral distribution
        normed = self.normalizer(src_hidden)

        # 2. Compress to fixed-length representation
        compressed = self.resampler(normed, src_mask)  # [B, K, tgt_dim]

        # 3. Quantize through FSQ (no codebook = no collapse)
        quantized, aux_loss, diversity = self.fsq(compressed)

        # 4. Scale for Mistral embedding space
        out = quantized * self.output_scale

        return out, aux_loss, diversity
