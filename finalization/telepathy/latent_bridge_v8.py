#!/usr/bin/env python
# telepathy/latent_bridge_v8.py
"""
Phase 8: Reconstruction Loss Bridge

KEY FIX: V7 suffered Mode Collapse - bridge memorized "students/clubs" templates
instead of encoding specific input content (ducks, pomegranates, etc.).

Solution: Add ReconstructionHead that projects soft tokens back to source space.
This forces the bridge to preserve input-specific information.
"""
import torch
import torch.nn as nn


class StatisticalNormalizer(nn.Module):
    def __init__(self, stats_path):
        super().__init__()
        try:
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            self.l_mean = nn.Parameter(stats["l_mean"].float(), requires_grad=True)
            self.l_std = nn.Parameter(stats["l_std"].float(), requires_grad=True)
            self.m_mean = nn.Parameter(stats["m_mean"].float(), requires_grad=True)
            self.m_std = nn.Parameter(stats["m_std"].float(), requires_grad=True)
            print(f"[StatisticalNormalizer] Loaded learnable stats from {stats_path}")
        except Exception as e:
            print(f"WARNING: Failed to load stats ({e}). Using identity.")
            self.l_mean = nn.Parameter(torch.tensor(0.0))
            self.l_std = nn.Parameter(torch.tensor(1.0))
            self.m_mean = nn.Parameter(torch.tensor(0.0))
            self.m_std = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return ((x.float() - self.l_mean) / self.l_std.clamp(min=1e-6)) * self.m_std + self.m_mean


class PerceiverResampler(nn.Module):
    def __init__(self, src_dim, tgt_dim, num_latents=64, heads=8, depth=4):
        super().__init__()
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
        x = self.latents.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](
                query=layer["ln1"](x), key=keys, value=keys,
                key_padding_mask=key_padding_mask
            )
            x = x + attn_out
            attn_out, _ = layer["self_attn"](
                query=layer["ln2"](x), key=layer["ln2"](x), value=layer["ln2"](x)
            )
            x = x + attn_out
            x = x + layer["ffn"](layer["ln3"](x))
        return x


class LatentBridgeV8(nn.Module):
    """
    Phase 8: Reconstruction Loss Bridge

    Key changes from V7:
    1. Added ReconstructionHead (recon_proj) to project back to source space
    2. Forward returns BOTH scaled output (for Mistral) AND raw compressed (for recon loss)
    3. Forces information preservation - can't cheat by memorizing templates
    """
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.normalizer = StatisticalNormalizer(args.stats_path)
        self.resampler = PerceiverResampler(
            src_dim, tgt_dim,
            args.soft_tokens, args.heads, args.depth
        )

        # V7: Output scaling
        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.target_rms = target_rms

        # PHASE 8: Reconstruction Head
        # Projects soft tokens back to source dimension to predict input mean
        self.recon_proj = nn.Linear(tgt_dim, src_dim)

        print(f"[LatentBridgeV8] Initialized with target_rms={target_rms:.4f}")
        print(f"[LatentBridgeV8] Added reconstruction head: {tgt_dim} -> {src_dim}")

    def forward(self, src_hidden, src_mask=None):
        # Preserve input dtype (bfloat16) after normalization
        input_dtype = src_hidden.dtype
        normed = self.normalizer(src_hidden)
        normed = normed.to(input_dtype)

        compressed = self.resampler(normed, src_mask)

        # Scale output for Mistral
        scaled_out = torch.tanh(compressed) * self.output_scale

        # Return BOTH: scaled (for Mistral) and raw compressed (for recon loss)
        return scaled_out, compressed
