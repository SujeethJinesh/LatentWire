#!/usr/bin/env python
# telepathy/latent_bridge_v15.py
"""
Phase 15: VQ-Telepathy (Vector Quantized Bridge)

THE FIX FOR MANIFOLD MISMATCH:

Previous failures:
- Regression (V7): Blurry averages (semantic drift)
- Diffusion Global (V12): Converged but lost details
- Diffusion Cross-Attn (V13-14): Failed to converge

VQ Solution:
1. Solves "Blur": Forces vectors to snap to discrete codebook entries
   - Every output is a valid "concept code"
   - Impossible to output off-manifold vectors

2. Solves "Drift": Entities map to specific codes
   - "Ducks" -> Code #42
   - "Chickens" -> Code #99
   - Cannot average to "Code #70.5" (Generic Bird)

3. Efficiency: 1-step inference (no diffusion iteration)

Architecture:
    Llama Hidden -> Normalizer -> Perceiver -> VQ Bottleneck -> Scale -> Mistral
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with Straight-Through Estimator (STE).

    Based on VQ-VAE (van den Oord et al., 2017).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # The Codebook - initialize with unit-normalized vectors for stable distances
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize with random unit vectors (better for normalized inputs)
        self.embedding.weight.data.normal_(0, 1)
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

    def forward(self, inputs):
        """
        Args:
            inputs: [B, K, D] continuous vectors

        Returns:
            quantized: [B, K, D] discrete vectors from codebook
            loss: VQ loss (codebook + commitment)
            perplexity: Codebook usage metric
        """
        input_shape = inputs.shape
        # Use contiguous().view() to handle non-contiguous tensors from attention
        flat_input = inputs.contiguous().view(-1, self.embedding_dim)

        # Ensure same dtype for distance calculation
        flat_input = flat_input.float()
        codebook = self.embedding.weight.float()

        # L2 normalize both for cosine similarity (prevents collapse)
        flat_input_norm = F.normalize(flat_input, dim=1)
        codebook_norm = F.normalize(codebook, dim=1)

        # Use negative cosine similarity as distance (higher similarity = lower distance)
        # This prevents collapse by making all codebook entries equally "reachable"
        similarity = torch.matmul(flat_input_norm, codebook_norm.t())
        distances = -similarity  # Convert to distance (minimize = maximize similarity)

        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize: lookup codebook vectors (use normalized codebook)
        quantized_norm = torch.matmul(encodings, codebook_norm).view(input_shape)

        # Scale quantized to match input magnitude (preserve scale information)
        input_scale = flat_input.norm(dim=1, keepdim=True).view(input_shape[0], input_shape[1], 1)
        quantized = quantized_norm * input_scale.mean()  # Use mean scale for stability

        # VQ Loss: use normalized vectors for loss computation
        flat_input_norm_shaped = flat_input_norm.view(input_shape)
        e_latent_loss = F.mse_loss(quantized_norm.detach(), flat_input_norm_shaped)
        q_latent_loss = F.mse_loss(quantized_norm, flat_input_norm_shaped.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator: gradient flows through quantized
        quantized = inputs + (quantized.to(inputs.dtype) - inputs).detach()

        # Perplexity: how many codes are being used
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


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
    Phase 15: Vector Quantized Telepathy Bridge

    Architecture:
        Llama Hidden -> StatisticalNormalizer -> PerceiverResampler
                     -> VectorQuantizer -> Output Scale -> Mistral Embeddings

    Key Features:
    - Discrete bottleneck prevents blurry/drifted outputs
    - 4096 codebook entries for rich concept vocabulary
    - 1-step inference (no diffusion iteration)
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

        # VQ Bottleneck (4096 codes, dimension = tgt_dim)
        # Note: Removed LayerNorm before VQ - it was causing all vectors to look identical
        # L2 normalization inside VQ (for cosine similarity) is sufficient
        self.vq = VectorQuantizer(
            num_embeddings=4096,
            embedding_dim=tgt_dim,
            commitment_cost=0.25
        )

        # Output scale to match Mistral embedding magnitude
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        print(f"[LatentBridgeV15] VQ-Telepathy Bridge")
        print(f"  - src_dim: {src_dim}")
        print(f"  - tgt_dim: {tgt_dim}")
        print(f"  - num_latents: {num_latents}")
        print(f"  - codebook_size: 4096")
        print(f"  - target_rms: {target_rms:.4f}")

    def forward(self, src_hidden, src_mask=None):
        """
        Args:
            src_hidden: [B, T, src_dim] - Llama hidden states
            src_mask: [B, T] - attention mask

        Returns:
            out: [B, K, tgt_dim] - quantized soft tokens for Mistral
            vq_loss: scalar - VQ training loss
            perplexity: scalar - codebook usage metric
        """
        # 1. Normalize Llama -> Mistral distribution
        normed = self.normalizer(src_hidden)

        # 2. Compress to fixed-length representation
        compressed = self.resampler(normed, src_mask)  # [B, K, tgt_dim]

        # 3. Quantize through codebook (L2 normalization happens inside VQ)
        quantized, vq_loss, perplexity = self.vq(compressed)

        # 4. Scale for Mistral embedding space
        out = quantized * self.output_scale

        return out, vq_loss, perplexity
