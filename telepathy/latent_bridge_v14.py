#!/usr/bin/env python
# telepathy/latent_bridge_v14.py
"""
Phase 14: Hybrid Conditioning Diffusion Bridge

THE FIX FOR V13's CONDITIONING COLLAPSE:

V13 Problem: Pure cross-attention was too weak. The DiT collapsed to predicting
average velocity, producing repetitive outputs ("I I I I...", "What's not here...").

V14 Solution: HYBRID CONDITIONING
1. Global Pooling (V12 style) - Strong "gist" vector via attention pooling
   - Gives DiT a "guide rail" to prevent collapse
   - Sets the "topic" (this is a math problem)

2. Cross-Attention (V13 style) - Sequence-level details
   - Lets DiT fetch specific entities ("Janet", "ducks", "16")
   - Fills in the details the global vector can't capture

This mirrors Stable Diffusion XL architecture:
- Pooled text embeddings (global) + sequence embeddings (local)

Architecture:
- Attention-based pooling (learnable query attends to source)
- AdaLN conditioning on global_cond + timestep
- Cross-attention to full source sequence in each block
- Increased dim (1024 vs V13's 512) for more capacity
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Maps timestep to conditioning vector."""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class DiTBlockV14(nn.Module):
    """
    Hybrid DiT Block with:
    1. AdaLN conditioning on Global Gist + Timestep (prevents collapse)
    2. Cross-Attention to full source sequence (captures details)
    """
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()

        # Self Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)

        # Cross Attention to source sequence (for entity details)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)

        # MLP
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

        # AdaLN modulation from GLOBAL conditioning (the key fix!)
        # This gives DiT a strong signal to prevent collapse
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, global_cond, src_seq, src_mask=None):
        """
        Args:
            x: [B, K, D] - noisy latent tokens
            global_cond: [B, D] - GLOBAL conditioning (pooled source + timestep)
            src_seq: [B, T_src, D] - full source sequence for cross-attention
            src_mask: [B, T_src] - attention mask (1=valid, 0=pad)
        """
        # Global condition drives the gates (AdaLN) - PREVENTS COLLAPSE
        modulation = self.adaLN_modulation(global_cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        # 1. Self Attention with AdaLN (globally conditioned)
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross Attention to source sequence (fetches details)
        x_norm_cross = self.norm_cross(x)
        kpm = ~src_mask.bool() if src_mask is not None else None
        cross_out, _ = self.cross_attn(
            query=x_norm_cross,
            key=src_seq,
            value=src_seq,
            key_padding_mask=kpm,
            need_weights=False
        )
        x = x + cross_out  # Residual connection

        # 3. MLP with AdaLN (globally conditioned)
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class LatentBridgeV14(nn.Module):
    """
    Phase 14: Hybrid Conditioning Diffusion Bridge

    Key improvements over V13:
    1. GLOBAL conditioning via attention pooling (prevents collapse)
    2. LOCAL conditioning via cross-attention (captures entities)
    3. Larger internal dim (1024 vs 512)
    4. More heads (16 vs 8)

    Training: Rectified Flow
    Inference: Euler ODE integration
    """
    def __init__(self, args, src_dim, tgt_dim):
        super().__init__()
        self.K = args.soft_tokens
        self.dim = 1024  # Increased from V13's 512
        self.tgt_dim = tgt_dim

        # Project source to internal dimension
        self.src_proj = nn.Linear(src_dim, self.dim)

        # Project target embeddings to/from internal dimension
        self.tgt_proj_in = nn.Linear(tgt_dim, self.dim)
        self.tgt_proj_out = nn.Linear(self.dim, tgt_dim)

        # GLOBAL POOLING (V12 style, but with attention)
        # Learnable query attends to source sequence to get a single "gist" vector
        self.cond_query = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.cond_pool = nn.MultiheadAttention(self.dim, 8, batch_first=True)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(self.dim)

        # DiT blocks with hybrid conditioning
        self.blocks = nn.ModuleList([
            DiTBlockV14(self.dim, heads=16)
            for _ in range(args.depth)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(self.dim)

        # Learnable positional embeddings for latent positions
        self.latent_pos = nn.Parameter(torch.randn(self.K, self.dim) * 0.02)

        print(f"[LatentBridgeV14] Hybrid Conditioning Diffusion Bridge")
        print(f"  - num_latents: {self.K}")
        print(f"  - internal_dim: {self.dim} (increased from V13's 512)")
        print(f"  - depth: {args.depth}")
        print(f"  - heads: 16")
        print(f"  - KEY: Global pooling (guide rail) + Cross-attention (details)")

    def get_conditioning(self, src_hidden, src_mask=None):
        """
        Compute both global and local conditioning from source.

        Returns:
            src_seq: [B, T_src, dim] - projected source sequence (for cross-attention)
            global_cond: [B, dim] - pooled global conditioning (for AdaLN)
        """
        # Project source to internal dimension
        src_seq = self.src_proj(src_hidden.to(self.src_proj.weight.dtype))

        # Attention-based global pooling
        # Learnable query attends to source to extract "gist"
        B = src_seq.shape[0]
        query = self.cond_query.expand(B, -1, -1).to(src_seq.dtype)

        # Create key_padding_mask for pooling attention
        kpm = ~src_mask.bool() if src_mask is not None else None

        global_cond, _ = self.cond_pool(
            query, src_seq, src_seq,
            key_padding_mask=kpm,
            need_weights=False
        )
        global_cond = global_cond.squeeze(1)  # [B, dim]

        return src_seq, global_cond

    def forward_loss(self, src_hidden, src_mask, target_embeds):
        """
        Rectified Flow training loss with hybrid conditioning.

        Args:
            src_hidden: [B, T_src, src_dim] - Llama hidden states
            src_mask: [B, T_src] - attention mask
            target_embeds: [B, T_tgt, tgt_dim] - target embeddings

        Returns:
            loss: scalar - MSE loss on velocity prediction
        """
        B = src_hidden.shape[0]
        device = src_hidden.device

        # Get conditioning (both global and local)
        src_seq, global_cond = self.get_conditioning(src_hidden, src_mask)

        # Prepare target (x1): resample/pad to K tokens
        curr_len = target_embeds.shape[1]
        if curr_len >= self.K:
            x1 = target_embeds[:, :self.K, :]
        else:
            x1 = F.pad(target_embeds, (0, 0, 0, self.K - curr_len))
        x1 = self.tgt_proj_in(x1.to(self.tgt_proj_in.weight.dtype))

        # Sample noise (x0)
        x0 = torch.randn_like(x1)

        # Sample timestep t ~ U[0,1]
        t = torch.rand(B, device=device, dtype=x1.dtype)

        # Linear interpolation: x_t = t * x1 + (1-t) * x0
        t_broad = t.view(B, 1, 1)
        x_t = t_broad * x1 + (1 - t_broad) * x0

        # Add positional embeddings
        x_t = x_t + self.latent_pos.unsqueeze(0).to(x_t.dtype)

        # Combine global conditioning with timestep
        t_emb = self.time_embed(t)
        combined_cond = global_cond + t_emb  # [B, dim]

        # Process through DiT blocks with HYBRID conditioning
        v_pred = x_t
        for block in self.blocks:
            v_pred = block(v_pred, combined_cond, src_seq, src_mask)
        v_pred = self.final_norm(v_pred)

        # True velocity: v = x1 - x0
        v_target = x1 - x0

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def generate(self, src_hidden, src_mask, steps=10):
        """
        Generate soft tokens using Euler ODE integration.

        Args:
            src_hidden: [B, T_src, src_dim] - Llama hidden states
            src_mask: [B, T_src] - attention mask
            steps: int - number of integration steps

        Returns:
            soft_tokens: [B, K, tgt_dim] - generated soft tokens
        """
        B = src_hidden.shape[0]
        device = src_hidden.device

        # Get conditioning (compute once, reuse)
        src_seq, global_cond = self.get_conditioning(src_hidden, src_mask)

        # Start from noise
        x = torch.randn(B, self.K, self.dim, device=device, dtype=src_seq.dtype)

        # Euler integration
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B,), t_val, device=device, dtype=src_seq.dtype)
            t_emb = self.time_embed(t)
            combined_cond = global_cond + t_emb

            # Add positional embeddings
            x_input = x + self.latent_pos.unsqueeze(0).to(x.dtype)

            # Predict velocity
            v_pred = x_input
            for block in self.blocks:
                v_pred = block(v_pred, combined_cond, src_seq, src_mask)
            v_pred = self.final_norm(v_pred)

            # Euler step
            x = x + v_pred * dt

        # Project back to target dimension
        return self.tgt_proj_out(x)
