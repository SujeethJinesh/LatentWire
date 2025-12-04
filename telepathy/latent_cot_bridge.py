#!/usr/bin/env python
# telepathy/latent_cot_bridge.py
"""
Phase 18: Latent Chain-of-Thought Bridge for GSM8K

MOTIVATION:
Classification (SST-2, AG News) succeeded with single-shot latent transfer.
Reasoning (GSM8K) requires multi-step processing - can't squeeze CoT into one burst.

SOLUTION: Latent CoT
Instead of: Llama -> Bridge -> [8 tokens] -> Mistral -> Answer
We use:     Llama -> Bridge -> [Thought 1] -> Bridge -> [Thought 2] -> ... -> Mistral

The bridge learns to produce a SEQUENCE of latent reasoning tokens that
accumulate information across steps, allowing Mistral to "think" before answering.

ARCHITECTURE:
- RecurrentPerceiverBridge: Takes (src_hidden, prev_latent) and produces next_latent
- Latent tokens from all steps are concatenated for Mistral
- Training: Supervise final answer, let intermediate reasoning emerge

REFERENCE: COCONUT (Chain of Continuous Thought)
"Training Large Language Models to Reason in a Continuous Latent Space" (2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentPerceiverBlock(nn.Module):
    """
    A single perceiver block that can incorporate previous latent state.

    Unlike standard Perceiver which only attends to source,
    this block can also attend to previous reasoning latents.
    """
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim

        # Cross-attention to source (question context)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_cross = nn.LayerNorm(dim)

        # Self-attention among current latents + previous latents
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_self = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.ln_ffn = nn.LayerNorm(dim)

    def forward(self, latents, src_kv, src_mask=None, prev_latent=None):
        """
        Args:
            latents: [B, K, D] - current latent queries
            src_kv: [B, T, D] - source hidden states (question)
            src_mask: [B, T] - attention mask for source
            prev_latent: [B, K, D] or None - previous reasoning step's output

        Returns:
            [B, K, D] - updated latents
        """
        # 1. Cross-attend to source (question)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None
        x = self.ln_cross(latents)
        attn_out, _ = self.cross_attn(
            query=x, key=src_kv, value=src_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        latents = latents + attn_out

        # 2. Self-attend among latents (incorporate prev_latent if available)
        x = self.ln_self(latents)
        if prev_latent is not None:
            # Concat previous latent for self-attention context
            # This allows current reasoning to attend to previous reasoning
            prev_normed = self.ln_self(prev_latent)
            kv_context = torch.cat([x, prev_normed], dim=1)  # [B, 2K, D]
        else:
            kv_context = x

        attn_out, _ = self.self_attn(
            query=x, key=kv_context, value=kv_context,
            need_weights=False
        )
        latents = latents + attn_out

        # 3. FFN
        latents = latents + self.ffn(self.ln_ffn(latents))

        return latents


class LatentCoTBridge(nn.Module):
    """
    Latent Chain-of-Thought Bridge

    Extends LatentBridgeV15 with recurrent reasoning capability.
    Can produce multiple reasoning steps, each building on the previous.

    Key differences from V15:
    1. Takes optional prev_latent for recurrent processing
    2. Step-aware positional encoding
    3. Designed for multi-step reasoning tasks (GSM8K)
    """
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.target_rms = target_rms

        # Number of latent tokens per reasoning step
        self.num_latents = getattr(args, 'soft_tokens', 8)
        # Number of reasoning steps (like CoT steps)
        self.num_steps = getattr(args, 'cot_steps', 4)
        # Perceiver depth
        self.depth = getattr(args, 'depth', 2)
        self.heads = getattr(args, 'heads', 8)

        # Input projection (Llama dim -> Mistral dim)
        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

        # Learnable latent queries (one set per step, or shared)
        # Using step-specific queries allows specialization
        self.latent_queries = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_latents, tgt_dim) * 0.02)
            for _ in range(self.num_steps)
        ])

        # Step embedding (to differentiate reasoning positions)
        self.step_embed = nn.Embedding(self.num_steps, tgt_dim)

        # Recurrent perceiver blocks (shared across steps for parameter efficiency)
        self.perceiver_blocks = nn.ModuleList([
            RecurrentPerceiverBlock(tgt_dim, self.heads)
            for _ in range(self.depth)
        ])

        # Output scale
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        print(f"[LatentCoTBridge] Latent Chain-of-Thought Bridge")
        print(f"  - src_dim: {src_dim}")
        print(f"  - tgt_dim: {tgt_dim}")
        print(f"  - num_latents: {self.num_latents} per step")
        print(f"  - num_steps: {self.num_steps} reasoning steps")
        print(f"  - total_latents: {self.num_latents * self.num_steps}")
        print(f"  - depth: {self.depth}")
        print(f"  - target_rms: {target_rms:.4f}")

    def forward_step(self, src_kv, src_mask, step_idx, prev_latent=None):
        """
        Run one reasoning step.

        Args:
            src_kv: [B, T, tgt_dim] - projected source hidden states
            src_mask: [B, T] - attention mask
            step_idx: int - which reasoning step (0-indexed)
            prev_latent: [B, K, tgt_dim] or None - previous step's output

        Returns:
            [B, K, tgt_dim] - latent tokens for this step
        """
        B = src_kv.shape[0]

        # Get step-specific queries
        queries = self.latent_queries[step_idx].unsqueeze(0).expand(B, -1, -1)
        queries = queries.to(src_kv.dtype)

        # Add step embedding
        step_emb = self.step_embed(torch.tensor([step_idx], device=src_kv.device))
        queries = queries + step_emb.unsqueeze(0)  # [B, K, D]

        # Run through perceiver blocks
        latents = queries
        for block in self.perceiver_blocks:
            latents = block(latents, src_kv, src_mask, prev_latent)

        return latents

    def forward(self, src_hidden, src_mask=None, return_all_steps=True):
        """
        Run full chain-of-thought reasoning.

        Args:
            src_hidden: [B, T, src_dim] - Llama hidden states
            src_mask: [B, T] - attention mask
            return_all_steps: if True, concatenate all steps; else return last only

        Returns:
            out: [B, K*num_steps, tgt_dim] or [B, K, tgt_dim] - latent tokens
            aux_loss: 0 (no auxiliary loss)
            diversity: 1.0 (continuous)
            z_variance: variance of outputs
        """
        # Project source to target dimension
        src_kv = self.input_proj(src_hidden.to(
            self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype
        ))

        # Run reasoning steps
        all_latents = []
        prev_latent = None

        for step_idx in range(self.num_steps):
            step_latent = self.forward_step(src_kv, src_mask, step_idx, prev_latent)
            all_latents.append(step_latent)
            prev_latent = step_latent

        # Combine latents from all steps
        if return_all_steps:
            combined = torch.cat(all_latents, dim=1)  # [B, K*num_steps, D]
        else:
            combined = all_latents[-1]  # [B, K, D]

        # RMS normalize and scale
        rms = torch.sqrt((combined ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (combined / rms) * self.output_scale

        # Compute variance for monitoring
        z_variance = combined.var(dim=[0, 1]).mean()

        return out, torch.tensor(0.0, device=src_hidden.device), 1.0, z_variance

    def forward_streaming(self, src_hidden, src_mask=None):
        """
        Generator that yields latent tokens step by step.
        Useful for visualization or streaming decoding.

        Yields:
            step_idx, step_latent for each reasoning step
        """
        src_kv = self.input_proj(src_hidden.to(
            self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype
        ))

        prev_latent = None
        for step_idx in range(self.num_steps):
            step_latent = self.forward_step(src_kv, src_mask, step_idx, prev_latent)

            # Normalize this step's output
            rms = torch.sqrt((step_latent ** 2).mean(dim=-1, keepdim=True) + 1e-8)
            step_out = (step_latent / rms) * self.output_scale

            yield step_idx, step_out
            prev_latent = step_latent
