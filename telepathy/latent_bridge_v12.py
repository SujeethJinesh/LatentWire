#!/usr/bin/env python
# telepathy/latent_bridge_v12.py
"""
Phase 12: Diffusion Bridge with DiT + Rectified Flow

THE FIX: Stop predicting (regression) → start generating (diffusion).

Why regression fails (The Blur Problem):
- Given input "Janet's ducks...", many valid Mistral vectors could decode it
- Regression outputs the AVERAGE of all valid outputs → lies OFF the manifold
- Off-manifold vectors decode as garbage

Why diffusion works:
- Learns to move TOWARD the data manifold from any starting point
- Output is guaranteed to lie ON the Mistral embedding manifold
- Sharp, valid vectors instead of blurry averages

Architecture: DiT (Diffusion Transformer) with Rectified Flow
- Rectified Flow: Linear interpolation x_t = t*target + (1-t)*noise
- Predict velocity v = target - noise (constant for RF)
- Source conditioning via AdaLN (Adaptive Layer Norm)
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embeddings for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TimestepEmbedding(nn.Module):
    """Maps timestep to conditioning vector."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.sinusoidal = SinusoidalPositionEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t):
        # t: [B] in [0, 1]
        t_emb = self.sinusoidal(t)
        return self.mlp(t_emb)


class AdaLN(nn.Module):
    """Adaptive Layer Norm - conditions on timestep and source context."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Predict scale and shift from conditioning
        self.to_gamma_beta = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )

    def forward(self, x, cond):
        # x: [B, L, D], cond: [B, D]
        gamma_beta = self.to_gamma_beta(cond)  # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, D] each
        x = self.norm(x)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with AdaLN conditioning."""
    def __init__(self, dim, heads=8, ff_mult=4, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or dim

        # Self-attention with AdaLN
        self.adaln1 = AdaLN(dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # Cross-attention to source (AdaLN conditioned)
        self.adaln2 = AdaLN(dim, cond_dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # FFN with AdaLN
        self.adaln3 = AdaLN(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x, cond, src_kv, src_mask=None):
        """
        x: [B, L, D] - noisy latents
        cond: [B, D] - timestep + global source conditioning
        src_kv: [B, S, D] - source hidden states for cross-attention
        src_mask: [B, S] - source attention mask
        """
        # Self-attention
        x_norm = self.adaln1(x, cond)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention to source
        x_norm = self.adaln2(x, cond)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None
        cross_out, _ = self.cross_attn(
            x_norm, src_kv, src_kv,
            key_padding_mask=key_padding_mask
        )
        x = x + cross_out

        # FFN
        x_norm = self.adaln3(x, cond)
        x = x + self.ffn(x_norm)

        return x


class LatentBridgeV12(nn.Module):
    """
    Phase 12: Diffusion Bridge using DiT + Rectified Flow

    Training (Rectified Flow):
        1. Sample t ~ U[0,1]
        2. Interpolate: x_t = t * target + (1-t) * noise
        3. Predict velocity: v_pred = model(x_t, t, source)
        4. Loss: ||v_pred - (target - noise)||^2

    Inference:
        1. Start from noise x_0 ~ N(0, 1)
        2. Integrate: x_1 = x_0 + v_pred (single step for RF)
        3. Multi-step for better quality: x_{t+dt} = x_t + v_pred * dt
    """
    def __init__(self, args, src_dim, tgt_dim, num_latents=128, depth=6, heads=8):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim

        # Project source to target dimension
        self.src_proj = nn.Linear(src_dim, tgt_dim)

        # Timestep conditioning
        self.time_embed = TimestepEmbedding(tgt_dim)

        # Global source conditioning (pool + project)
        self.src_pool = nn.Sequential(
            nn.Linear(tgt_dim, tgt_dim),
            nn.SiLU(),
            nn.Linear(tgt_dim, tgt_dim),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(tgt_dim, heads=heads, cond_dim=tgt_dim)
            for _ in range(depth)
        ])

        # Final output projection
        self.final_norm = nn.LayerNorm(tgt_dim)
        self.final_proj = nn.Linear(tgt_dim, tgt_dim)

        # Learnable positional embeddings for latent positions
        self.latent_pos = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)

        print(f"[LatentBridgeV12] DiT Diffusion Bridge initialized")
        print(f"  - num_latents: {num_latents}")
        print(f"  - depth: {depth}")
        print(f"  - heads: {heads}")
        print(f"  - src_dim: {src_dim} -> tgt_dim: {tgt_dim}")

    def forward(self, x_t, t, src_hidden, src_mask=None):
        """
        Forward pass: predict velocity from noisy latents.

        Args:
            x_t: [B, L, D] - noisy latent tokens at time t
            t: [B] - timesteps in [0, 1]
            src_hidden: [B, S, D_src] - source model hidden states
            src_mask: [B, S] - source attention mask

        Returns:
            v_pred: [B, L, D] - predicted velocity
        """
        B = x_t.shape[0]

        # Project source to target dimension
        src_kv = self.src_proj(src_hidden.float()).to(x_t.dtype)

        # Timestep embedding
        t_emb = self.time_embed(t)  # [B, D]

        # Global source conditioning (mean pool + transform)
        if src_mask is not None:
            mask = src_mask.unsqueeze(-1).float()
            src_pooled = (src_kv * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            src_pooled = src_kv.mean(dim=1)
        src_cond = self.src_pool(src_pooled)  # [B, D]

        # Combined conditioning
        cond = t_emb + src_cond  # [B, D]

        # Add positional embeddings to noisy latents
        x = x_t + self.latent_pos.unsqueeze(0)

        # Process through DiT blocks
        for block in self.blocks:
            x = block(x, cond, src_kv, src_mask)

        # Final projection
        x = self.final_norm(x)
        v_pred = self.final_proj(x)

        return v_pred

    def _resample_to_latents(self, embeds):
        """
        Resample variable-length embeddings to fixed num_latents.

        Args:
            embeds: [B, S, D] - variable length embeddings

        Returns:
            resampled: [B, num_latents, D] - fixed length
        """
        B, S, D = embeds.shape
        if S == self.num_latents:
            return embeds
        # Use interpolation to resize
        # [B, S, D] -> [B, D, S] -> interpolate -> [B, D, num_latents] -> [B, num_latents, D]
        embeds_t = embeds.permute(0, 2, 1)  # [B, D, S]
        resampled = torch.nn.functional.interpolate(
            embeds_t.float(),
            size=self.num_latents,
            mode='linear',
            align_corners=True
        ).to(embeds.dtype)
        return resampled.permute(0, 2, 1)  # [B, num_latents, D]

    def forward_loss(self, src_hidden, tgt_embeds, src_mask=None):
        """
        Compute Rectified Flow training loss.

        Args:
            src_hidden: [B, S, D_src] - source model hidden states
            tgt_embeds: [B, T, D_tgt] - target embeddings (ground truth)
            src_mask: [B, S] - source attention mask

        Returns:
            loss: scalar - MSE loss on velocity prediction
        """
        B = src_hidden.shape[0]
        device = src_hidden.device
        dtype = src_hidden.dtype

        # Resample target to fixed num_latents
        target = self._resample_to_latents(tgt_embeds)  # [B, num_latents, D]

        # Sample noise and timestep
        noise = torch.randn_like(target)
        t = torch.rand(B, device=device, dtype=dtype)

        # Linear interpolation: x_t = t * target + (1-t) * noise
        t_expand = t.view(B, 1, 1)
        x_t = t_expand * target + (1 - t_expand) * noise

        # Predict velocity
        v_pred = self.forward(x_t, t, src_hidden, src_mask)

        # True velocity (constant in Rectified Flow)
        v_true = target - noise

        # MSE loss
        loss = torch.nn.functional.mse_loss(v_pred, v_true)

        return loss

    @torch.no_grad()
    def sample(self, src_hidden, src_mask=None, num_steps=10):
        """
        Generate soft tokens from source hidden states.

        Uses Euler integration of the Rectified Flow ODE:
            dx/dt = v(x, t)
            x(0) = noise, x(1) = target

        Args:
            src_hidden: [B, S, D_src] - source model hidden states
            src_mask: [B, S] - source attention mask
            num_steps: int - number of integration steps

        Returns:
            soft_tokens: [B, L, D] - generated soft tokens
        """
        B = src_hidden.shape[0]
        device = src_hidden.device
        dtype = src_hidden.dtype

        # Start from Gaussian noise
        x = torch.randn(B, self.num_latents, self.tgt_dim, device=device, dtype=dtype)

        # Euler integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device, dtype=dtype)
            v = self.forward(x, t, src_hidden, src_mask)
            x = x + v * dt

        return x

    @torch.no_grad()
    def generate(self, src_hidden, src_mask=None, steps=10):
        """
        Alias for sample() - matches eval script API.
        """
        return self.sample(src_hidden, src_mask, num_steps=steps)
