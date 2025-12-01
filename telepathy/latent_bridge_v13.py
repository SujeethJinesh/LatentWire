#!/usr/bin/env python
# telepathy/latent_bridge_v13.py
"""
Phase 13: High-Fidelity Cross-Attention Diffusion

THE TWO FIXES:

1. Remove Global Pooling Bottleneck
   - V12: Pooled Llama sequence to single vector -> entity info destroyed
   - V13: Full cross-attention to Llama sequence -> can read each token

2. Question Reconstruction Target
   - V12: Trained to generate Answer embeddings ("18")
   - V13: Train to generate Question embeddings ("Janet has 16 ducks...")
   - If bridge can reconstruct Q in Mistral's space, Mistral will solve it

Architecture: DiT with Cross-Attention + Rectified Flow
- AdaLN conditions ONLY on timestep (not pooled source)
- Cross-attention reads the FULL Llama sequence at every layer
- Each DiT block can attend to "Janet" (token 5) and "ducks" (token 12) individually
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
        # Sinusoidal embedding
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # Convert to model dtype
        embedding = embedding.to(self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class DiTBlockV13(nn.Module):
    """
    DiT Block with FULL cross-attention to source sequence.

    Key difference from V12:
    - NO global pooling of source
    - Cross-attention reads full Llama sequence
    - AdaLN conditions on timestep ONLY
    """
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()

        # Self Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)

        # Cross Attention to Llama Sequence (THE FIX: Full sequence, not pooled)
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

        # AdaLN modulation from Timestep ONLY (not pooled source)
        # Produces: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # Zero-init for residual learning
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, t_emb, src_seq, src_mask=None):
        """
        Args:
            x: [B, K, D] - noisy latent tokens
            t_emb: [B, D] - timestep embedding
            src_seq: [B, T_src, D] - FULL projected Llama sequence
            src_mask: [B, T_src] - attention mask (1=valid, 0=pad)
        """
        # AdaLN modulation from timestep
        modulation = self.adaLN_modulation(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        # 1. Self Attention with AdaLN
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross Attention to FULL Llama Sequence (THE KEY FIX)
        x_norm_cross = self.norm_cross(x)
        # key_padding_mask: True = ignore, False = attend
        kpm = ~src_mask.bool() if src_mask is not None else None
        cross_out, _ = self.cross_attn(
            query=x_norm_cross,
            key=src_seq,
            value=src_seq,
            key_padding_mask=kpm,
            need_weights=False
        )
        x = x + cross_out  # Residual connection

        # 3. MLP with AdaLN
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class LatentBridgeV13(nn.Module):
    """
    Phase 13: High-Fidelity Cross-Attention Diffusion Bridge

    Key changes from V12:
    1. NO global pooling - cross-attention reads full Llama sequence
    2. Smaller internal dimension (512) to reduce memory
    3. Projects to/from target dimension at boundaries

    Training (Rectified Flow):
        x_t = t * target + (1-t) * noise
        v_pred = model(x_t, t, src_seq)
        loss = MSE(v_pred, target - noise)

    Inference:
        x = noise
        for step in range(num_steps):
            v = model(x, t, src_seq)
            x = x + v * dt
        return project_out(x)
    """
    def __init__(self, args, src_dim, tgt_dim):
        super().__init__()
        self.K = args.soft_tokens  # Number of output latent tokens
        self.dim = 512  # Internal DiT dimension (reduced for memory)
        self.tgt_dim = tgt_dim

        # Project Llama hidden states to DiT dimension
        self.src_proj = nn.Linear(src_dim, self.dim)

        # Project target embeddings to/from DiT dimension
        self.tgt_proj_in = nn.Linear(tgt_dim, self.dim)
        self.tgt_proj_out = nn.Linear(self.dim, tgt_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(self.dim)

        # DiT blocks with cross-attention
        self.blocks = nn.ModuleList([
            DiTBlockV13(self.dim, heads=8)
            for _ in range(args.depth)
        ])

        # Final norm and projection
        self.final_norm = nn.LayerNorm(self.dim)

        # Learnable positional embeddings for latent positions
        self.latent_pos = nn.Parameter(torch.randn(self.K, self.dim) * 0.02)

        print(f"[LatentBridgeV13] High-Fidelity Cross-Attention Diffusion Bridge")
        print(f"  - num_latents: {self.K}")
        print(f"  - internal_dim: {self.dim}")
        print(f"  - depth: {args.depth}")
        print(f"  - src_dim: {src_dim} -> internal -> tgt_dim: {tgt_dim}")
        print(f"  - KEY: Full cross-attention to Llama sequence (no pooling)")

    def forward_loss(self, src_hidden, src_mask, target_embeds):
        """
        Rectified Flow training loss.

        Args:
            src_hidden: [B, T_src, src_dim] - Llama hidden states
            src_mask: [B, T_src] - attention mask (1=valid, 0=pad)
            target_embeds: [B, T_tgt, tgt_dim] - Mistral question embeddings

        Returns:
            loss: scalar - MSE loss on velocity prediction
        """
        B = src_hidden.shape[0]
        device = src_hidden.device
        dtype = src_hidden.dtype

        # Project Llama sequence to DiT dimension
        src_seq = self.src_proj(src_hidden.to(self.src_proj.weight.dtype))

        # Prepare target (x1): resample/pad to K tokens, project to DiT dim
        curr_len = target_embeds.shape[1]
        if curr_len >= self.K:
            x1 = target_embeds[:, :self.K, :]
        else:
            x1 = F.pad(target_embeds, (0, 0, 0, self.K - curr_len))
        x1 = self.tgt_proj_in(x1.to(self.tgt_proj_in.weight.dtype))  # [B, K, dim]

        # Sample noise (x0)
        x0 = torch.randn_like(x1)

        # Sample timestep t ~ U[0,1]
        t = torch.rand(B, device=device, dtype=self.src_proj.weight.dtype)

        # Linear interpolation: x_t = t * x1 + (1-t) * x0
        t_broad = t.view(B, 1, 1)
        x_t = t_broad * x1 + (1 - t_broad) * x0

        # Add positional embeddings
        x_t = x_t + self.latent_pos.unsqueeze(0)

        # Get timestep embedding
        t_emb = self.time_embed(t)

        # Process through DiT blocks with cross-attention
        v_pred = x_t
        for block in self.blocks:
            v_pred = block(v_pred, t_emb, src_seq, src_mask)

        v_pred = self.final_norm(v_pred)

        # True velocity: v = x1 - x0 (constant in Rectified Flow)
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

        # Project source sequence once
        src_seq = self.src_proj(src_hidden.to(self.src_proj.weight.dtype))

        # Start from noise
        x = torch.randn(B, self.K, self.dim, device=device, dtype=src_seq.dtype)

        # Euler integration
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B,), t_val, device=device, dtype=src_seq.dtype)
            t_emb = self.time_embed(t)

            # Add positional embeddings
            x_input = x + self.latent_pos.unsqueeze(0)

            # Predict velocity through DiT blocks
            v_pred = x_input
            for block in self.blocks:
                v_pred = block(v_pred, t_emb, src_seq, src_mask)
            v_pred = self.final_norm(v_pred)

            # Euler step
            x = x + v_pred * dt

        # Project back to target dimension
        return self.tgt_proj_out(x)
