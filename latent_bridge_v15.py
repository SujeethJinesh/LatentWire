#!/usr/bin/env python
"""
Latent Bridge V15 - Telepathy Bridge Architecture

This module contains the core bridge architectures for cross-model communication:
- PerceiverResampler: Cross-attention resampler for compressing hidden states
- LatentBridgeV15: Main telepathy bridge for classification tasks
- LatentCoTBridge: Chain-of-thought bridge for reasoning tasks
"""

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Perceiver-style cross-attention resampler.

    Compresses variable-length source hidden states into fixed-length soft tokens
    using learned latent queries and cross-attention.
    """
    def __init__(self, src_dim, tgt_dim, num_latents=64, heads=8, depth=4):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim
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
        keys = self.input_proj(src_hidden.to(self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype))
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            x_norm = layer["ln1"](x)
            attn_out, _ = layer["cross_attn"](
                query=x_norm, key=keys, value=keys,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            x = x + attn_out
            x_norm = layer["ln2"](x)
            attn_out, _ = layer["self_attn"](
                query=x_norm, key=x_norm, value=x_norm,
                need_weights=False
            )
            x = x + attn_out
            x = x + layer["ffn"](layer["ln3"](x))
        return x


class LatentBridgeV15(nn.Module):
    """Telepathy Bridge (Continuous).

    Main bridge architecture for cross-model communication. Uses a Perceiver
    resampler to compress source hidden states into soft tokens that can be
    used as prefix embeddings for a target model.

    Args:
        args: Configuration object with soft_tokens, heads, depth attributes
        src_dim: Dimension of source model hidden states
        tgt_dim: Dimension of target model embeddings
        target_rms: Target RMS value for output normalization
    """
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        num_latents = getattr(args, 'soft_tokens', 128)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 4)
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward(self, src_hidden, src_mask=None):
        """Forward pass through bridge.

        Args:
            src_hidden: Source hidden states [B, seq_len, src_dim]
            src_mask: Attention mask [B, seq_len]

        Returns:
            tuple: (soft_tokens, aux_loss, compression_ratio, z_variance)
        """
        compressed = self.resampler(src_hidden, src_mask)
        rms = torch.sqrt((compressed ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (compressed / rms) * self.output_scale
        z_variance = compressed.var(dim=[0, 1]).mean()
        return out, torch.tensor(0.0, device=src_hidden.device), 1.0, z_variance


class RecurrentPerceiverBlock(nn.Module):
    """Perceiver block with recurrent capability for Chain-of-Thought."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_cross = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_self = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.ln_ffn = nn.LayerNorm(dim)

    def forward(self, latents, src_kv, src_mask=None, prev_latent=None):
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None
        x = self.ln_cross(latents)
        attn_out, _ = self.cross_attn(
            query=x, key=src_kv, value=src_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        latents = latents + attn_out

        x = self.ln_self(latents)
        if prev_latent is not None:
            prev_normed = self.ln_self(prev_latent)
            kv_context = torch.cat([x, prev_normed], dim=1)
        else:
            kv_context = x

        attn_out, _ = self.self_attn(
            query=x, key=kv_context, value=kv_context,
            need_weights=False
        )
        latents = latents + attn_out
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


class LatentCoTBridge(nn.Module):
    """Latent Chain-of-Thought Bridge for reasoning tasks.

    Multi-step bridge that generates soft tokens iteratively, with each step
    attending to both the source and previous step's latents.
    """
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.target_rms = target_rms
        self.num_latents = getattr(args, 'soft_tokens', 8)
        self.num_steps = getattr(args, 'cot_steps', 4)
        self.depth = getattr(args, 'depth', 2)
        self.heads = getattr(args, 'heads', 8)

        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()
        self.latent_queries = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_latents, tgt_dim) * 0.02)
            for _ in range(self.num_steps)
        ])
        self.step_embed = nn.Embedding(self.num_steps, tgt_dim)
        self.perceiver_blocks = nn.ModuleList([
            RecurrentPerceiverBlock(tgt_dim, self.heads)
            for _ in range(self.depth)
        ])
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward_step(self, src_kv, src_mask, step_idx, prev_latent=None):
        B = src_kv.shape[0]
        queries = self.latent_queries[step_idx].unsqueeze(0).expand(B, -1, -1)
        queries = queries.to(src_kv.dtype)
        step_emb = self.step_embed(torch.tensor([step_idx], device=src_kv.device))
        queries = queries + step_emb.unsqueeze(0)
        latents = queries
        for block in self.perceiver_blocks:
            latents = block(latents, src_kv, src_mask, prev_latent)
        return latents

    def forward(self, src_hidden, src_mask=None, return_all_steps=True):
        src_kv = self.input_proj(src_hidden.to(
            self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype
        ))
        all_latents = []
        prev_latent = None

        for step_idx in range(self.num_steps):
            step_latent = self.forward_step(src_kv, src_mask, step_idx, prev_latent)
            all_latents.append(step_latent)
            prev_latent = step_latent

        if return_all_steps:
            combined = torch.cat(all_latents, dim=1)
        else:
            combined = all_latents[-1]

        rms = torch.sqrt((combined ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (combined / rms) * self.output_scale
        z_variance = combined.var(dim=[0, 1]).mean()
        return out, torch.tensor(0.0, device=src_hidden.device), 1.0, z_variance


class Args:
    """Minimal args object for bridge instantiation."""
    def __init__(self, soft_tokens=8, heads=8, depth=2, cot_steps=4, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.cot_steps = cot_steps
        self.use_fsq = use_fsq
        self.stats_path = stats_path
