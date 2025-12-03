#!/usr/bin/env python
# telepathy/latent_bridge_vq.py
"""
Phase 16: VQ Bridge for SST-2 Signal Check

Uses Vector Quantization to force discrete decisions.
Perfect for binary classification (sentiment analysis).

If VQ works on SST-2 but not GSM8K, it confirms:
- The bridge architecture is fundamentally sound
- GSM8K requires more "bandwidth" than the bottleneck provides
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization with Straight-Through Estimator.

    Codebook of 4096 entries - forces discrete representation.
    """
    def __init__(self, num_embeddings=4096, embedding_dim=4096, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        """
        Args:
            inputs: [B, K, D] continuous vectors
        Returns:
            quantized: [B, K, D] quantized vectors
            loss: VQ loss (commitment + codebook)
            perplexity: measure of codebook usage
        """
        B, K, D = inputs.shape

        # Flatten for distance calculation
        flat_input = inputs.reshape(-1, self.embedding_dim).float()

        # Calculate distances: (x-y)^2 = x^2 + y^2 - 2xy
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight.float()**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.float().t())
        )

        # Encoding - find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Quantize
        quantized = self.embedding(encoding_indices.squeeze(1))
        quantized = quantized.view(B, K, D).to(inputs.dtype)

        # Losses
        e_latent_loss = F.mse_loss(quantized.detach().float(), inputs.float())
        q_latent_loss = F.mse_loss(quantized.float(), inputs.float().detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator - gradients pass through
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity (metric for codebook usage)
        # High perplexity = many codes used, low = collapse
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class PerceiverResampler(nn.Module):
    """
    Perceiver-style cross-attention resampler.
    Compresses variable-length input to fixed-length latent sequence.
    """
    def __init__(self, src_dim, tgt_dim, num_latents=32, heads=8, depth=2):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)
        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

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

    def forward(self, src_hidden, src_mask=None):
        B = src_hidden.shape[0]
        keys = self.input_proj(src_hidden)
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            x_norm = layer["ln1"](x)
            attn_out, _ = layer["cross_attn"](
                query=x_norm, key=keys, value=keys,
                key_padding_mask=key_padding_mask, need_weights=False
            )
            x = x + attn_out

            x_norm = layer["ln2"](x)
            attn_out, _ = layer["self_attn"](
                query=x_norm, key=x_norm, value=x_norm, need_weights=False
            )
            x = x + attn_out

            x = x + layer["ffn"](layer["ln3"](x))

        return x


class LatentBridgeVQ(nn.Module):
    """
    VQ-based Latent Bridge for SST-2 Signal Check.

    Architecture:
        Llama Hidden -> Perceiver (32 tokens) -> VQ (4096 codes) -> Scale -> Mistral

    Key difference from continuous V15:
        - Discrete bottleneck forces categorical decisions
        - Perfect for binary classification (sentiment)
    """
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

        # Perceiver resampler
        num_latents = getattr(args, 'soft_tokens', 32)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # VQ Bottleneck
        num_codes = getattr(args, 'num_codes', 4096)
        self.vq = VectorQuantizer(num_embeddings=num_codes, embedding_dim=tgt_dim)

        # Output scaling
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        print(f"[LatentBridgeVQ] VQ Bridge for Signal Check")
        print(f"  - src_dim: {src_dim}")
        print(f"  - tgt_dim: {tgt_dim}")
        print(f"  - num_latents: {num_latents}")
        print(f"  - num_codes: {num_codes}")
        print(f"  - target_rms: {target_rms:.4f}")

    def forward(self, src_hidden, src_mask=None):
        """
        Args:
            src_hidden: [B, T, src_dim] - Llama hidden states
            src_mask: [B, T] - attention mask
        Returns:
            out: [B, K, tgt_dim] - soft tokens for Mistral
            vq_loss: scalar - VQ commitment + codebook loss
            perplexity: scalar - codebook usage metric
        """
        # 1. Compress to fixed-length
        compressed = self.resampler(src_hidden, src_mask)

        # 2. Quantize through VQ
        quantized, vq_loss, perplexity = self.vq(compressed)

        # 3. Scale for Mistral
        out = quantized * self.output_scale

        return out, vq_loss, perplexity
