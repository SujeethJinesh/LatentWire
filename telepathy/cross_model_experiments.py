#!/usr/bin/env python3
"""
Cross-Model Communication Experiments for Telepathy

This module implements NOVEL experiments identified by 20+ opus subagents
across diverse fields (neuroscience, physics, information theory, etc.).

NOVEL EXPERIMENTS (high novelty + feasibility):
1. Predictive Coding Bridge - transmit prediction errors only (neuroscience)
2. Optimal Transport Bridge - Sinkhorn alignment (topology/geometry)
3. Contrastive InfoNCE - rate-distortion bound estimation (info theory)
4. Sparse Distributed Reps - k-WTA sparsity (neuroscience)
5. Residual/Innovation Coding - transmit residuals (signal processing)
6. Lock-and-Key Binding - sparse attention binding (chemistry)
7. Mixture-of-Experts Bridge - heterogeneous semantic routing (Mixtral/DeepSeek)

LEGACY EXPERIMENTS (not novel, kept for reference):
- Ridge Regression Baseline (LatentMAS - exact copy)
- Multi-Layer Extraction (standard technique)
- VIB Regularization (2017 technique)
- Curriculum Training (COCONUT - exact copy)

Run with: python telepathy/cross_model_experiments.py --experiment <name>
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.bridge import LatentBridge, PerceiverResampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# NOVEL EXPERIMENT 1: Predictive Coding Bridge (Neuroscience)
# =============================================================================
# Key insight: Instead of transmitting the full hidden state, transmit only
# the prediction error (innovation). The receiver maintains a predictive
# model and uses errors to update its representation.
#
# This is fundamentally different from existing approaches because:
# - Standard bridges transmit compressed versions of the full state
# - This transmits only the DELTA needed to correct the receiver's prediction
# - Information efficiency: Only send what the receiver doesn't already know
# =============================================================================

class PredictiveCodingBridge(nn.Module):
    """
    Predictive Coding Bridge - transmit prediction errors only.

    Architecture:
    1. Receiver maintains a predictive model of sender state
    2. Sender computes prediction error = actual - predicted
    3. Bridge compresses and transmits the error signal
    4. Receiver updates representation using error

    This is novel because it implements predictive coding theory from
    neuroscience for cross-model communication, reducing bits transmitted.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

        num_latents = getattr(args, 'soft_tokens', 8)
        self.num_latents = num_latents

        # Predictive model: estimates sender state from context
        # (in practice, from primer/task embedding)
        self.prior_mean = nn.Parameter(torch.zeros(1, src_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(1, src_dim))

        # Error encoder: compress prediction error
        self.error_encoder = nn.Sequential(
            nn.Linear(src_dim, src_dim // 2),
            nn.GELU(),
            nn.Linear(src_dim // 2, tgt_dim * num_latents),
        )

        # Error precision: learn which dimensions matter most
        self.precision_logits = nn.Parameter(torch.zeros(src_dim))

        # Output scaling
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        logger.info(f"PredictiveCodingBridge: {num_latents} soft tokens via error coding")

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, hidden_dim] from sender
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
            error_magnitude: scalar for monitoring
        """
        B = src_hidden.shape[0]

        # Pool sender hidden state
        if src_mask is not None:
            mask_expanded = src_mask.unsqueeze(-1).float()
            pooled = (src_hidden * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            pooled = src_hidden[:, -1, :]  # Last token

        # Compute prediction error
        prior = self.prior_mean.expand(B, -1)
        error = pooled - prior

        # Weight error by learned precision (which dimensions matter)
        precision = torch.sigmoid(self.precision_logits)
        weighted_error = error * precision

        # Encode error into soft tokens
        encoded = self.error_encoder(weighted_error.float())
        soft_tokens = encoded.view(B, self.num_latents, self.tgt_dim)

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Metrics for monitoring
        error_magnitude = error.pow(2).mean()
        diversity = precision.std()  # How selective is precision?

        return soft_tokens.to(src_hidden.dtype), error_magnitude, diversity, weighted_error.var()


# =============================================================================
# NOVEL EXPERIMENT 2: Optimal Transport Bridge (Sinkhorn)
# =============================================================================
# Key insight: Use optimal transport to align sender and receiver distributions.
# Instead of learning a direct mapping, learn a transport plan that minimizes
# the cost of moving probability mass from sender to receiver space.
#
# This is novel because:
# - OT provides theoretical guarantees on alignment quality
# - Sinkhorn is differentiable and efficient
# - Can handle multi-modal distributions in both spaces
# =============================================================================

class OptimalTransportBridge(nn.Module):
    """
    Optimal Transport Bridge using Sinkhorn algorithm.

    Architecture:
    1. Project sender states to cost matrix space
    2. Compute Sinkhorn transport plan
    3. Apply transport to get receiver-space representation

    Key parameters:
    - epsilon: entropy regularization (lower = sharper transport)
    - n_iters: Sinkhorn iterations (more = more accurate)
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 epsilon: float = 0.1, n_iters: int = 20):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.epsilon = epsilon
        self.n_iters = n_iters

        num_latents = getattr(args, 'soft_tokens', 8)
        self.num_latents = num_latents

        # Source projector
        self.src_proj = nn.Linear(src_dim, tgt_dim)

        # Learnable anchor points in target space (support of target distribution)
        self.anchors = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)

        # Cost function parameters (Mahalanobis-like)
        self.cost_scale = nn.Parameter(torch.ones(tgt_dim))

        # Output scaling
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        logger.info(f"OptimalTransportBridge: {num_latents} anchors, eps={epsilon}")

    def sinkhorn(self, C: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm for entropy-regularized OT.

        Args:
            C: [B, n, m] cost matrix
            a: [B, n] source marginal
            b: [B, m] target marginal
        Returns:
            P: [B, n, m] transport plan
        """
        # Gibbs kernel
        K = torch.exp(-C / self.epsilon)

        # Initialize scaling vectors
        u = torch.ones_like(a)

        for _ in range(self.n_iters):
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # Transport plan
        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        return P

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
        """
        B, S, _ = src_hidden.shape

        # Project source to target space
        src_proj = self.src_proj(src_hidden.float())  # [B, S, tgt_dim]

        # Compute cost matrix (squared scaled Euclidean distance)
        anchors = self.anchors.unsqueeze(0).expand(B, -1, -1)  # [B, K, tgt_dim]

        # Cost: ||src - anchor||^2 weighted by cost_scale
        scale = self.cost_scale.abs() + 0.1
        src_scaled = src_proj * scale  # [B, S, tgt_dim]
        anchor_scaled = anchors * scale  # [B, K, tgt_dim]

        # Efficient squared distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        src_sq = (src_scaled ** 2).sum(-1)  # [B, S]
        anchor_sq = (anchor_scaled ** 2).sum(-1)  # [B, K]
        cross = torch.bmm(src_scaled, anchor_scaled.transpose(1, 2))  # [B, S, K]

        C = src_sq.unsqueeze(-1) + anchor_sq.unsqueeze(-2) - 2 * cross  # [B, S, K]

        # Marginals
        if src_mask is not None:
            a = src_mask.float()
            a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
        else:
            a = torch.ones(B, S, device=src_hidden.device) / S

        b = torch.ones(B, self.num_latents, device=src_hidden.device) / self.num_latents

        # Compute transport plan
        P = self.sinkhorn(C, a, b)  # [B, S, K]

        # Transport: soft_tokens = P^T @ src_proj (barycentric projection)
        soft_tokens = torch.bmm(P.transpose(1, 2), src_proj)  # [B, K, tgt_dim]

        # Normalize by column sums (how much mass each anchor received)
        col_sums = P.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8  # [B, K, 1]
        soft_tokens = soft_tokens / col_sums

        # RMS scaling
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # OT cost for monitoring
        ot_cost = (P * C).sum(dim=(1, 2)).mean()

        return soft_tokens.to(src_hidden.dtype), ot_cost, P.std(), soft_tokens.var()


# =============================================================================
# NOVEL EXPERIMENT 3: Contrastive InfoNCE Bridge
# =============================================================================
# Key insight: Train bridge to maximize mutual information between input and
# soft tokens via InfoNCE bound. This provides a principled information-
# theoretic objective for compression.
#
# This is novel because:
# - Provides rate-distortion interpretation of bridge training
# - InfoNCE gives lower bound on mutual information
# - Natural regularization against mode collapse
# =============================================================================

class ContrastiveInfoNCEBridge(nn.Module):
    """
    Contrastive InfoNCE Bridge - maximize I(input; soft_tokens).

    Architecture:
    1. Encode input to soft tokens
    2. Encode soft tokens back to critic space
    3. InfoNCE: positive = (input, its soft_tokens), negative = (input, other soft_tokens)

    The InfoNCE loss provides a lower bound on mutual information:
    I(X; Z) >= log(N) - L_InfoNCE
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 temperature: float = 0.07):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.temperature = temperature

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Main encoder (Perceiver-style)
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Critic networks for InfoNCE
        critic_dim = 256
        self.src_critic = nn.Sequential(
            nn.Linear(src_dim, critic_dim),
            nn.GELU(),
            nn.Linear(critic_dim, critic_dim),
        )
        self.tgt_critic = nn.Sequential(
            nn.Linear(tgt_dim * num_latents, critic_dim),
            nn.GELU(),
            nn.Linear(critic_dim, critic_dim),
        )

        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        logger.info(f"ContrastiveInfoNCEBridge: {num_latents} tokens, temp={temperature}")

    def compute_infonce_loss(self, src_hidden: torch.Tensor, soft_tokens: torch.Tensor,
                             src_mask=None) -> torch.Tensor:
        """
        Compute InfoNCE loss between source and soft tokens.
        """
        B = src_hidden.shape[0]

        # Pool source representation
        if src_mask is not None:
            mask_expanded = src_mask.unsqueeze(-1).float()
            src_pooled = (src_hidden * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            src_pooled = src_hidden.mean(dim=1)

        # Project to critic space
        src_z = F.normalize(self.src_critic(src_pooled.float()), dim=-1)  # [B, critic_dim]
        tgt_z = F.normalize(self.tgt_critic(soft_tokens.float().view(B, -1)), dim=-1)  # [B, critic_dim]

        # Compute similarity matrix
        sim = torch.mm(src_z, tgt_z.t()) / self.temperature  # [B, B]

        # InfoNCE: diagonal entries are positives
        labels = torch.arange(B, device=sim.device)
        loss = F.cross_entropy(sim, labels)

        return loss

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
            infonce_loss: scalar (auxiliary loss)
        """
        # Encode
        soft_tokens = self.resampler(src_hidden, src_mask)

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Compute InfoNCE loss (only during training)
        if self.training:
            infonce_loss = self.compute_infonce_loss(src_hidden, soft_tokens, src_mask)
        else:
            infonce_loss = torch.tensor(0.0, device=soft_tokens.device)

        return soft_tokens, infonce_loss, 1.0, soft_tokens.var()


# =============================================================================
# NOVEL EXPERIMENT 4: Sparse Distributed Representation Bridge (k-WTA)
# =============================================================================
# Key insight: Use k-Winner-Take-All sparsity to create sparse distributed
# representations. Only the top-k dimensions are active in each soft token.
#
# This is novel because:
# - Sparse codes are more robust and interpretable
# - Natural capacity control via sparsity level k
# - Biological plausibility (sparse coding in cortex)
# =============================================================================

class SparseKWTABridge(nn.Module):
    """
    Sparse Distributed Representation Bridge using k-WTA.

    Architecture:
    1. Encode to dense representation
    2. Apply k-WTA: keep only top-k values per token
    3. Sparsity provides natural regularization

    Key insight: Sparsity level k controls information capacity.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 sparsity_k: int = 128):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.sparsity_k = sparsity_k

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Base encoder
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Pre-sparsity projection (expand dimensions for sparsity)
        self.pre_sparse = nn.Linear(tgt_dim, tgt_dim)

        # Post-sparsity normalization
        self.post_norm = nn.LayerNorm(tgt_dim)

        # Learnable sparsity threshold (for soft k-WTA during training)
        self.threshold_temp = nn.Parameter(torch.tensor(1.0))

        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        logger.info(f"SparseKWTABridge: {num_latents} tokens, k={sparsity_k}/{tgt_dim}")

    def k_wta(self, x: torch.Tensor, k: int, hard: bool = False) -> torch.Tensor:
        """
        k-Winner-Take-All activation.

        During training: soft top-k via temperature-scaled sigmoid
        During inference: hard top-k
        """
        B, N, D = x.shape

        if hard or not self.training:
            # Hard k-WTA
            _, indices = torch.topk(x.abs(), k, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, indices, 1.0)
            return x * mask
        else:
            # Soft k-WTA via temperature-controlled threshold
            # Find k-th largest value per token
            topk_vals, _ = torch.topk(x.abs(), k, dim=-1)
            threshold = topk_vals[:, :, -1:].detach()  # [B, N, 1]

            # Soft threshold
            temp = self.threshold_temp.abs() + 0.1
            soft_mask = torch.sigmoid((x.abs() - threshold) / temp)

            return x * soft_mask

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim] (sparse)
        """
        # Encode
        dense = self.resampler(src_hidden, src_mask)

        # Pre-sparsity transform
        pre = self.pre_sparse(dense)

        # Apply k-WTA sparsity
        sparse = self.k_wta(pre, self.sparsity_k)

        # Post-sparsity normalization
        soft_tokens = self.post_norm(sparse)

        # RMS scaling
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Sparsity metrics
        actual_sparsity = (sparse.abs() < 1e-6).float().mean()

        return soft_tokens, torch.tensor(0.0, device=soft_tokens.device), actual_sparsity, soft_tokens.var()


# =============================================================================
# NOVEL EXPERIMENT 5: Residual/Innovation Coding Bridge
# =============================================================================
# Key insight: Instead of transmitting the full representation, transmit
# the residual between the encoded representation and a learned prior.
# Similar to predictive coding but with a static learned prior.
#
# This is novel because:
# - Residuals have lower entropy than full states (better compression)
# - Prior captures dataset-level statistics
# - Progressive refinement is possible
# =============================================================================

class ResidualCodingBridge(nn.Module):
    """
    Residual/Innovation Coding Bridge.

    Architecture:
    1. Learn a prior (mean representation for the task)
    2. Encode input and compute residual from prior
    3. Transmit compressed residual
    4. Reconstruct at receiver as prior + residual

    Key insight: Residuals are easier to compress than full states.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 num_refinement_steps: int = 2):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.num_steps = num_refinement_steps

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Learned prior in target space
        self.prior = nn.Parameter(torch.randn(1, num_latents, tgt_dim) * 0.02)

        # Initial encoder (full state)
        self.initial_encoder = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Residual encoders (for progressive refinement)
        self.residual_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tgt_dim, tgt_dim),
                nn.GELU(),
                nn.Linear(tgt_dim, tgt_dim),
            ) for _ in range(num_refinement_steps)
        ])

        # Residual scaling (how much to weight each refinement)
        self.residual_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_refinement_steps)
        ])

        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        logger.info(f"ResidualCodingBridge: {num_latents} tokens, {num_refinement_steps} refinement steps")

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
        """
        B = src_hidden.shape[0]

        # Initial encoding
        initial = self.initial_encoder(src_hidden, src_mask)

        # Start from prior
        prior = self.prior.expand(B, -1, -1)

        # Compute initial residual
        residual = initial - prior

        # Progressive refinement
        current = prior.clone()
        total_residual_magnitude = residual.pow(2).mean()

        for i, (encoder, scale) in enumerate(zip(self.residual_encoders, self.residual_scales)):
            # Encode and compress residual
            compressed_residual = encoder(residual)

            # Add scaled residual
            current = current + scale.abs() * compressed_residual

            # Update residual for next step
            if i < self.num_steps - 1:
                residual = initial - current

        soft_tokens = current

        # RMS scaling
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        return soft_tokens, total_residual_magnitude, 1.0, soft_tokens.var()


# =============================================================================
# NOVEL EXPERIMENT 6: Lock-and-Key Binding Bridge
# =============================================================================
# Key insight: Inspired by molecular binding, soft tokens should "bind" to
# specific receiver positions via sparse attention. Each token acts as a
# "key" that activates specific "locks" in the receiver.
#
# This is novel because:
# - Explicit binding mechanism instead of dense attention
# - Sparsity in binding provides interpretability
# - Chemistry-inspired loss encourages selectivity
# =============================================================================

class LockAndKeyBridge(nn.Module):
    """
    Lock-and-Key Binding Bridge.

    Architecture:
    1. Generate soft tokens (keys)
    2. Compute binding affinity to learned receptor sites (locks)
    3. Sparse binding: each key binds to only a few locks
    4. Output is binding-weighted combination

    Key insight: Sparse binding is more robust and interpretable.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 num_receptors: int = 32, binding_sparsity: float = 0.1):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.binding_sparsity = binding_sparsity

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Key generator
        self.key_encoder = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Learnable receptor sites (locks)
        self.receptors = nn.Parameter(torch.randn(num_receptors, tgt_dim) * 0.02)

        # Binding affinity network
        self.binding_net = nn.Sequential(
            nn.Linear(tgt_dim * 2, tgt_dim),
            nn.GELU(),
            nn.Linear(tgt_dim, 1),
        )

        # Output projection
        self.output_proj = nn.Linear(num_receptors, tgt_dim)

        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents
        self.num_receptors = num_receptors

        logger.info(f"LockAndKeyBridge: {num_latents} keys, {num_receptors} receptors")

    def compute_binding(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute binding affinities between keys and receptors.

        Args:
            keys: [B, K, D] key representations
        Returns:
            binding: [B, K, R] binding affinities
            sparsity_loss: scalar encouraging sparse binding
        """
        B, K, D = keys.shape
        R = self.num_receptors

        # Expand for pairwise comparison
        keys_exp = keys.unsqueeze(2).expand(-1, -1, R, -1)  # [B, K, R, D]
        receptors_exp = self.receptors.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)  # [B, K, R, D]

        # Concatenate and compute affinity
        combined = torch.cat([keys_exp, receptors_exp], dim=-1)  # [B, K, R, 2D]
        affinity = self.binding_net(combined).squeeze(-1)  # [B, K, R]

        # Sparse binding via top-k or threshold
        k = max(1, int(R * self.binding_sparsity))
        topk_vals, topk_idx = torch.topk(affinity, k, dim=-1)

        # Soft sparsity mask
        threshold = topk_vals[:, :, -1:].detach()
        sparse_mask = torch.sigmoid((affinity - threshold) * 10)

        binding = F.softmax(affinity, dim=-1) * sparse_mask

        # Sparsity loss: encourage each key to bind to few receptors
        sparsity_loss = binding.pow(2).sum(dim=-1).mean()  # L2 promotes sparsity

        return binding, sparsity_loss

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
        """
        B = src_hidden.shape[0]

        # Generate keys
        keys = self.key_encoder(src_hidden, src_mask)  # [B, K, D]

        # Compute binding
        binding, sparsity_loss = self.compute_binding(keys)  # [B, K, R]

        # Bind to receptors
        bound = torch.bmm(binding, self.receptors.unsqueeze(0).expand(B, -1, -1))  # [B, K, D]

        # Combine with original keys
        soft_tokens = keys + bound

        # RMS scaling
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Binding selectivity metric
        binding_entropy = -(binding * (binding + 1e-8).log()).sum(dim=-1).mean()

        return soft_tokens, sparsity_loss, binding_entropy, soft_tokens.var()


# =============================================================================
# NOVEL EXPERIMENT 7: Mixture-of-Experts Bridge (MoE)
# =============================================================================
# Key insight: Use Mixture-of-Experts to handle heterogeneous cross-model
# transfer. Different experts specialize in different types of semantic content.
#
# This is novel because:
# - MoE has NEVER been applied to cross-model LLM communication
# - Experts can specialize: one for sentiment, one for entities, one for syntax
# - Interpretable: which experts fire tells us what information transfers
# - Efficient: only top-k experts compute per input (sparse activation)
#
# Based on:
# - Mixtral 8x7B: top-2 routing, 8 experts
# - DeepSeek-MoE: shared expert, auxiliary-loss-free with learnable bias
# - Soft MoE: fully differentiable routing for smooth gradients
# =============================================================================

class MoEFeedForward(nn.Module):
    """
    Expert feedforward network (same architecture as transformer FFN).
    """

    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoEBridge(nn.Module):
    """
    Mixture-of-Experts Bridge for cross-model LLM communication.

    Architecture:
    1. Perceiver encodes source to latent tokens
    2. MoE layer routes each token to top-k experts
    3. Shared expert (DeepSeek-style) handles common patterns
    4. Load balancing loss ensures all experts are utilized

    Key innovation: MoE for cross-model transfer, not just within-model scaling.
    Different experts can specialize in different semantic content types.

    Parameters:
    - num_experts: Number of expert FFNs (default 8, like Mixtral)
    - top_k: Number of experts to route to (default 2, like Mixtral)
    - use_shared_expert: Whether to include a shared expert (DeepSeek-style)
    - aux_loss_weight: Weight for load balancing loss (0.01 default)
    - use_aux_loss_free: Use learnable bias instead of aux loss (DeepSeek-V3)
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 num_experts: int = 8, top_k: int = 2, use_shared_expert: bool = True,
                 aux_loss_weight: float = 0.01, use_aux_loss_free: bool = False):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.aux_loss_weight = aux_loss_weight
        self.use_aux_loss_free = use_aux_loss_free

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Perceiver encoder (same as other bridges)
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Router: linear projection to expert logits
        self.router = nn.Linear(tgt_dim, num_experts, bias=False)

        # Learnable bias for auxiliary-loss-free routing (DeepSeek-V3 style)
        if use_aux_loss_free:
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.expert_bias = None

        # Expert feedforward networks
        self.experts = nn.ModuleList([
            MoEFeedForward(tgt_dim, hidden_mult=4) for _ in range(num_experts)
        ])

        # Shared expert (DeepSeek-style) - always activated
        if use_shared_expert:
            self.shared_expert = MoEFeedForward(tgt_dim, hidden_mult=2)
            self.shared_expert_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.shared_expert = None
            self.shared_expert_weight = None

        # Output scaling
        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        # Expert usage tracking for interpretability
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

        logger.info(f"MoEBridge: {num_latents} tokens, {num_experts} experts, top-{top_k} routing, "
                   f"shared_expert={use_shared_expert}, aux_loss_free={use_aux_loss_free}")

    def compute_routing(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute expert routing probabilities.

        Args:
            x: [B, K, D] latent tokens
        Returns:
            router_probs: [B, K, num_experts] softmax routing probabilities
            top_k_indices: [B, K, top_k] indices of selected experts
            top_k_weights: [B, K, top_k] normalized weights for selected experts
        """
        B, K, D = x.shape

        # Compute router logits
        router_logits = self.router(x)  # [B, K, num_experts]

        # Add learnable bias for auxiliary-loss-free routing
        if self.expert_bias is not None:
            router_logits = router_logits + self.expert_bias

        # Softmax over experts
        router_probs = F.softmax(router_logits, dim=-1)  # [B, K, num_experts]

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [B, K, top_k]

        # Renormalize weights for selected experts
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return router_probs, top_k_indices, top_k_weights

    def compute_load_balancing_loss(self, router_probs: torch.Tensor,
                                      top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss (Mixtral/GShard style).

        L_aux = alpha * N * sum(f_i * P_i)

        where:
        - f_i = fraction of tokens where expert i is selected (hard, from top-k)
        - P_i = mean routing probability assigned to expert i (soft)
        - N = number of experts
        - alpha = aux_loss_weight

        This loss encourages uniform expert utilization by penalizing correlation
        between hard assignments and soft probabilities.
        """
        if self.use_aux_loss_free:
            # No auxiliary loss - rely on learnable bias for balancing
            return torch.tensor(0.0, device=router_probs.device)

        # f_i: fraction of tokens where expert i is in top-k (hard selection)
        # Convert top_k_indices to one-hot and compute fraction
        # top_k_indices: [B, K, top_k]
        B, K, top_k = top_k_indices.shape
        top_k_one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts)  # [B, K, top_k, num_experts]
        # Sum over top_k positions (each token can select multiple experts)
        # Then average over batch and tokens
        f = top_k_one_hot.float().sum(dim=2).mean(dim=(0, 1))  # [num_experts]

        # P_i: mean routing probability assigned to expert i (soft probabilities)
        # router_probs: [B, K, num_experts]
        P = router_probs.mean(dim=(0, 1))  # [num_experts]

        # Load balancing loss
        # Penalizes when hard assignments (f) correlate with soft probabilities (P)
        # This encourages the router to spread load across experts
        aux_loss = self.aux_loss_weight * self.num_experts * (f * P).sum()

        return aux_loss

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim] from sender
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
            aux_loss: load balancing loss
            expert_entropy: entropy of routing distribution (for monitoring)
            z_var: variance of soft tokens
        """
        B = src_hidden.shape[0]

        # Encode source to latent tokens
        latents = self.resampler(src_hidden, src_mask)  # [B, K, D]

        # Compute routing
        router_probs, top_k_indices, top_k_weights = self.compute_routing(latents)

        # Track expert usage for interpretability
        if self.training:
            with torch.no_grad():
                for i in range(self.num_experts):
                    mask = (top_k_indices == i).any(dim=-1).float()  # [B, K]
                    self.expert_counts[i] += mask.sum()
                self.total_tokens += B * self.num_latents

        # Apply experts with sparse routing
        # Initialize output
        expert_output = torch.zeros_like(latents)

        # For each expert, compute output for tokens routed to it
        for i in range(self.num_experts):
            # Mask for tokens where this expert is in top-k
            # [B, K, top_k] -> [B, K]
            expert_mask = (top_k_indices == i).any(dim=-1)

            if not expert_mask.any():
                continue

            # Get weight for this expert (sum over top_k position where this expert appears)
            expert_weight = torch.zeros(B, self.num_latents, device=latents.device)
            for k in range(self.top_k):
                expert_weight += top_k_weights[:, :, k] * (top_k_indices[:, :, k] == i).float()

            # Compute expert output
            expert_out = self.experts[i](latents)  # [B, K, D]

            # Add weighted expert output
            expert_output = expert_output + expert_weight.unsqueeze(-1) * expert_out

        # Add shared expert output (always activated, with learnable weight)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(latents)
            expert_output = expert_output + self.shared_expert_weight * shared_out

        # Residual connection: output = residual + MoE(residual)
        # Standard pattern in MoE architectures (Mixtral, DeepSeek)
        soft_tokens = latents + expert_output

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Compute auxiliary loss
        aux_loss = self.compute_load_balancing_loss(router_probs, top_k_indices)

        # Compute routing entropy (higher = more balanced)
        routing_entropy = -(router_probs * (router_probs + 1e-8).log()).sum(dim=-1).mean()

        return soft_tokens.to(src_hidden.dtype), aux_loss, routing_entropy, soft_tokens.var()

    def get_expert_utilization(self) -> Dict[str, float]:
        """Get expert utilization statistics for interpretability."""
        if self.total_tokens.item() == 0:
            return {}

        utilization = {}
        for i in range(self.num_experts):
            utilization[f'expert_{i}'] = (self.expert_counts[i] / self.total_tokens).item()

        return utilization

    def reset_expert_counts(self):
        """Reset expert usage counters."""
        self.expert_counts.zero_()
        self.total_tokens.zero_()


# =============================================================================
# MATH EXPERIMENT 8: Spectral CCA Bridge (Canonical Correlation Analysis)
# =============================================================================
# Key insight: Use CCA to find a shared subspace where sender and receiver
# representations are maximally correlated. Project through this subspace.
#
# This is novel because:
# - CCA provides closed-form solution for alignment
# - Finds optimal linear projection without iterative training
# - Theoretical guarantees on correlation maximization
# =============================================================================

class SpectralCCABridge(nn.Module):
    """
    Spectral CCA Bridge - align via Canonical Correlation Analysis.

    Architecture:
    1. Collect sender-receiver embedding pairs
    2. Compute CCA to find shared subspace
    3. Project sender through CCA and expand to receiver space

    This is a hybrid: CCA computed once, then neural refinement.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 cca_dim: int = 256):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.cca_dim = cca_dim

        num_latents = getattr(args, 'soft_tokens', 8)
        self.num_latents = num_latents

        # Learnable projections (initialized randomly, could be set via CCA)
        self.src_proj = nn.Linear(src_dim, cca_dim)
        self.tgt_proj = nn.Linear(cca_dim, tgt_dim * num_latents)

        # Correlation-preserving regularization
        self.register_buffer('src_mean', torch.zeros(src_dim))
        self.register_buffer('src_std', torch.ones(src_dim))

        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        logger.info(f"SpectralCCABridge: {num_latents} tokens via {cca_dim}-dim CCA space")

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
        """
        B = src_hidden.shape[0]

        # Pool source
        if src_mask is not None:
            mask_expanded = src_mask.unsqueeze(-1).float()
            pooled = (src_hidden * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            pooled = src_hidden[:, -1, :]

        # Normalize (for correlation preservation)
        pooled_norm = (pooled - self.src_mean) / (self.src_std + 1e-8)

        # Project through CCA space
        cca_repr = self.src_proj(pooled_norm.float())
        cca_repr = F.gelu(cca_repr)  # Non-linearity for expressiveness

        # Expand to target space
        expanded = self.tgt_proj(cca_repr)
        soft_tokens = expanded.view(B, self.num_latents, self.tgt_dim)

        # RMS scaling
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Correlation metric
        correlation = F.cosine_similarity(pooled_norm, cca_repr[:, :self.src_dim] if self.cca_dim >= self.src_dim else F.pad(cca_repr, (0, self.src_dim - self.cca_dim)), dim=-1).mean()

        return soft_tokens.to(src_hidden.dtype), torch.tensor(0.0, device=soft_tokens.device), correlation, soft_tokens.var()


# =============================================================================
# MATH EXPERIMENT 8: Flow Matching Bridge
# =============================================================================
# Key insight: Use continuous normalizing flows to learn the transformation
# from sender space to receiver space. Flow matching provides a simulation-free
# training objective.
#
# This is novel because:
# - Flow matching is state-of-the-art for generative modeling
# - Never applied to cross-model LLM communication
# - Provides smooth, invertible transformations
# =============================================================================

class FlowMatchingBridge(nn.Module):
    """
    Flow Matching Bridge - learn transformation via conditional flow matching.

    Architecture:
    1. Define source distribution (sender hidden states)
    2. Define target distribution (receiver embedding space)
    3. Learn velocity field that transports source to target
    4. Integrate ODE to get soft tokens

    Based on: Lipman et al. "Flow Matching for Generative Modeling" (2023)
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 num_flow_steps: int = 4):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.num_steps = num_flow_steps

        num_latents = getattr(args, 'soft_tokens', 8)
        self.num_latents = num_latents

        # Velocity network: v(x, t) predicts flow direction
        self.velocity_net = nn.Sequential(
            nn.Linear(tgt_dim + 1, tgt_dim * 2),  # +1 for time embedding
            nn.GELU(),
            nn.Linear(tgt_dim * 2, tgt_dim * 2),
            nn.GELU(),
            nn.Linear(tgt_dim * 2, tgt_dim),
        )

        # Initial projection from source to target space
        self.initial_proj = nn.Linear(src_dim, tgt_dim * num_latents)

        # Target prior (learned Gaussian)
        self.target_mean = nn.Parameter(torch.zeros(1, num_latents, tgt_dim))
        self.target_logstd = nn.Parameter(torch.zeros(1, num_latents, tgt_dim) - 1)

        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        logger.info(f"FlowMatchingBridge: {num_latents} tokens, {num_flow_steps} flow steps")

    def compute_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field at point x and time t.

        Args:
            x: [B, K, D] current position
            t: [B, 1] or scalar time in [0, 1]
        Returns:
            v: [B, K, D] velocity
        """
        B, K, D = x.shape

        # Expand time to match shape
        if t.dim() == 0:
            t = t.expand(B, 1)
        t_expanded = t.unsqueeze(-1).expand(B, K, 1)

        # Concatenate position and time
        xt = torch.cat([x, t_expanded], dim=-1)  # [B, K, D+1]

        # Compute velocity
        v = self.velocity_net(xt)
        return v

    def integrate_flow(self, x0: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """
        Integrate flow from x0 to x1 using Euler method.

        Args:
            x0: [B, K, D] initial position
        Returns:
            x1: [B, K, D] final position
        """
        if num_steps is None:
            num_steps = self.num_steps

        x = x0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=x.device)
            v = self.compute_velocity(x, t)
            x = x + dt * v

        return x

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
        """
        B = src_hidden.shape[0]

        # Pool source
        if src_mask is not None:
            mask_expanded = src_mask.unsqueeze(-1).float()
            pooled = (src_hidden * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            pooled = src_hidden[:, -1, :]

        # Initial projection
        x0 = self.initial_proj(pooled.float())
        x0 = x0.view(B, self.num_latents, self.tgt_dim)

        # Integrate flow
        soft_tokens = self.integrate_flow(x0)

        # RMS scaling
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Flow matching loss (for training): MSE between predicted and optimal velocity
        # Optimal velocity for linear interpolation: v*(x,t) = x1 - x0
        if self.training:
            # Sample random time
            t = torch.rand(B, 1, device=soft_tokens.device)
            # Interpolate
            target = self.target_mean.expand(B, -1, -1) + torch.exp(self.target_logstd) * torch.randn_like(x0)
            xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * target
            # Optimal velocity
            v_opt = target - x0
            # Predicted velocity
            v_pred = self.compute_velocity(xt, t)
            flow_loss = F.mse_loss(v_pred, v_opt)
        else:
            flow_loss = torch.tensor(0.0, device=soft_tokens.device)

        return soft_tokens.to(src_hidden.dtype), flow_loss, 1.0, soft_tokens.var()


# =============================================================================
# HAIL MARY EXPERIMENT 1: Cross-Modal Distillation Bridge
# =============================================================================
# Key insight: Train the bridge so that receiver with soft tokens matches
# the sender's full probability distribution via KL divergence.
# This provides richer supervision than just task labels.
# =============================================================================

class CrossModalDistillationBridge(nn.Module):
    """
    Cross-Modal Distillation Bridge - match sender's output distribution.

    Architecture:
    1. Standard Perceiver encoding to soft tokens
    2. Additional KL divergence loss between sender logits and receiver logits
    3. Temperature scaling for softer probability distributions

    Key insight: The sender's full distribution contains more information
    than just the correct label. KD transfers this dark knowledge.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 temperature: float = 2.0, kd_weight: float = 0.5):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.temperature = temperature
        self.kd_weight = kd_weight

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Standard Perceiver encoder
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Output scaling
        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        # Store sender logits for KD loss computation (set during training step)
        self.cached_sender_logits = None

        logger.info(f"CrossModalDistillationBridge: {num_latents} tokens, T={temperature}, kd_weight={kd_weight}")

    def set_sender_logits(self, logits: torch.Tensor):
        """Cache sender logits for KD loss computation."""
        self.cached_sender_logits = logits.detach()

    def compute_kd_loss(self, receiver_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between sender and receiver distributions.

        Args:
            receiver_logits: [B, vocab_size] from receiver model
        Returns:
            kd_loss: scalar KL divergence loss
        """
        if self.cached_sender_logits is None:
            return torch.tensor(0.0, device=receiver_logits.device)

        # Align vocab sizes (use minimum)
        min_vocab = min(self.cached_sender_logits.shape[-1], receiver_logits.shape[-1])
        sender_logits = self.cached_sender_logits[..., :min_vocab]
        receiver_logits = receiver_logits[..., :min_vocab]

        # Soft targets with temperature
        sender_probs = F.softmax(sender_logits / self.temperature, dim=-1)
        receiver_log_probs = F.log_softmax(receiver_logits / self.temperature, dim=-1)

        # KL divergence (scaled by T^2 as per Hinton et al.)
        kd_loss = F.kl_div(receiver_log_probs, sender_probs, reduction='batchmean')
        kd_loss = kd_loss * (self.temperature ** 2)

        return kd_loss * self.kd_weight

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
            aux_loss: 0 (KD loss computed separately via compute_kd_loss)
        """
        # Encode
        soft_tokens = self.resampler(src_hidden, src_mask)

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        return soft_tokens.to(src_hidden.dtype), torch.tensor(0.0, device=soft_tokens.device), 1.0, soft_tokens.var()


# =============================================================================
# HAIL MARY EXPERIMENT 2: MINE Bridge (Mutual Information Neural Estimation)
# =============================================================================
# Key insight: Use Donsker-Varadhan representation for tighter MI bounds
# than InfoNCE. This provides asymptotically exact MI estimation.
# =============================================================================

class MINEBridge(nn.Module):
    """
    MINE Bridge - maximize I(input; soft_tokens) via Donsker-Varadhan.

    Architecture:
    1. Perceiver encodes to soft tokens
    2. Statistics network T(x, z) estimates MI
    3. MINE objective: I(X;Z) >= E[T(x,z)] - log(E[exp(T(x',z))])

    Key insight: MINE provides tighter MI bounds than InfoNCE, especially
    when batch sizes are limited.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 mine_weight: float = 0.1):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.mine_weight = mine_weight

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Main encoder
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # MINE statistics network T(x, z)
        stat_dim = 256
        self.statistics_net = nn.Sequential(
            nn.Linear(src_dim + tgt_dim * num_latents, stat_dim),
            nn.GELU(),
            nn.Linear(stat_dim, stat_dim),
            nn.GELU(),
            nn.Linear(stat_dim, 1),
        )

        # Moving average for MINE stability (exponential moving average of exp(T))
        self.register_buffer('ema_exp_t', torch.tensor(1.0))
        self.ema_decay = 0.99

        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        logger.info(f"MINEBridge: {num_latents} tokens, mine_weight={mine_weight}")

    def compute_mine_loss(self, src_hidden: torch.Tensor, soft_tokens: torch.Tensor,
                          src_mask=None) -> torch.Tensor:
        """
        Compute MINE loss (negative MI lower bound).

        Donsker-Varadhan: I(X;Z) >= E[T(x,z)] - log(E[exp(T(x',z))])
        """
        B = src_hidden.shape[0]

        # Pool source representation
        if src_mask is not None:
            mask_expanded = src_mask.unsqueeze(-1).float()
            src_pooled = (src_hidden * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            src_pooled = src_hidden.mean(dim=1)

        # Flatten soft tokens
        soft_flat = soft_tokens.view(B, -1)  # [B, K*D]

        # Positive samples: (x_i, z_i) pairs
        positive_input = torch.cat([src_pooled.float(), soft_flat.float()], dim=-1)  # [B, src_dim + K*D]
        t_positive = self.statistics_net(positive_input).squeeze(-1)  # [B]

        # Negative samples: (x_i, z_j) with j shuffled
        perm = torch.randperm(B, device=src_hidden.device)
        soft_shuffled = soft_flat[perm]
        negative_input = torch.cat([src_pooled.float(), soft_shuffled.float()], dim=-1)
        t_negative = self.statistics_net(negative_input).squeeze(-1)  # [B]

        # MINE objective with EMA for stability
        positive_term = t_positive.mean()

        # Use log-sum-exp trick for numerical stability
        exp_t_neg = torch.exp(t_negative)

        # Update EMA
        if self.training:
            with torch.no_grad():
                self.ema_exp_t = self.ema_decay * self.ema_exp_t + (1 - self.ema_decay) * exp_t_neg.mean()

        # Use EMA in gradient (biased but lower variance)
        negative_term = torch.log(self.ema_exp_t + 1e-8) + (exp_t_neg.mean() / (self.ema_exp_t + 1e-8)) - 1

        # Negative because we want to maximize MI
        mine_loss = -(positive_term - negative_term)

        return mine_loss * self.mine_weight

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
            mine_loss: scalar auxiliary loss
        """
        # Encode
        soft_tokens = self.resampler(src_hidden, src_mask)

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Compute MINE loss
        if self.training:
            mine_loss = self.compute_mine_loss(src_hidden, soft_tokens, src_mask)
        else:
            mine_loss = torch.tensor(0.0, device=soft_tokens.device)

        return soft_tokens.to(src_hidden.dtype), mine_loss, 1.0, soft_tokens.var()


# =============================================================================
# HAIL MARY EXPERIMENT 3: Mixture-of-Depths Bridge (Early Exit)
# =============================================================================
# Key insight: Not all tokens need the same amount of computation.
# Easy tokens can exit early, hard tokens traverse full depth.
# =============================================================================

class MixtureOfDepthsBridge(nn.Module):
    """
    Mixture-of-Depths Bridge - adaptive compute via early exit.

    Architecture:
    1. Each Perceiver layer has a router deciding which tokens continue
    2. Tokens that exit early accumulate to final output
    3. Hard tokens get full depth processing

    Based on: Raposo et al. "Mixture-of-Depths" (2024)
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 capacity_factor: float = 0.5):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.capacity_factor = capacity_factor  # Fraction of tokens that continue at each layer

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 4)  # More depth for MoD to be effective

        self.num_latents = num_latents
        self.depth = depth

        # Input projection
        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)

        # Per-layer blocks with routers
        self.layers = nn.ModuleList()
        self.routers = nn.ModuleList()

        for _ in range(depth):
            # Transformer-style block
            self.layers.append(nn.ModuleDict({
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
            }))

            # Router: predicts which tokens should continue
            self.routers.append(nn.Linear(tgt_dim, 1))

        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        logger.info(f"MixtureOfDepthsBridge: {num_latents} tokens, {depth} layers, capacity={capacity_factor}")

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
        """
        B = src_hidden.shape[0]

        # Project source
        keys = self.input_proj(src_hidden.to(self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype))
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        # Initialize latents
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)

        # Track which tokens have exited and their values
        accumulated_output = torch.zeros_like(x)
        active_mask = torch.ones(B, self.num_latents, device=x.device, dtype=torch.bool)

        # Capacity loss for load balancing
        capacity_loss = torch.tensor(0.0, device=x.device)

        for i, (layer, router) in enumerate(zip(self.layers, self.routers)):
            # Only process active tokens
            if not active_mask.any():
                break

            # Compute router scores for active tokens
            router_scores = router(x).squeeze(-1)  # [B, K]

            # Determine how many tokens continue (based on capacity_factor)
            k = max(1, int(self.num_latents * self.capacity_factor))

            # Select top-k tokens to continue (per batch)
            _, continue_idx = torch.topk(router_scores, k, dim=-1)  # [B, k]

            # Create continue mask
            continue_mask = torch.zeros_like(active_mask)
            continue_mask.scatter_(1, continue_idx, True)

            # Tokens that exit at this layer
            exit_mask = active_mask & ~continue_mask

            # Accumulate exited tokens
            accumulated_output = accumulated_output + x * exit_mask.unsqueeze(-1).float()

            # Process continuing tokens through the layer
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

            # Update active mask
            active_mask = continue_mask

            # Capacity loss: encourage uniform routing
            # Penalize deviation from expected capacity
            actual_continue = continue_mask.float().mean()
            capacity_loss = capacity_loss + (actual_continue - self.capacity_factor) ** 2

        # Add remaining active tokens to output
        accumulated_output = accumulated_output + x * active_mask.unsqueeze(-1).float()

        soft_tokens = accumulated_output

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        return soft_tokens.to(src_hidden.dtype), capacity_loss * 0.01, 1.0, soft_tokens.var()


# =============================================================================
# HAIL MARY EXPERIMENT 4: Thalamic Relay Bridge
# =============================================================================
# Key insight: The thalamus acts as the brain's relay station, using
# inhibitory gating to selectively transmit task-relevant information.
# =============================================================================

class ThalamicRelayBridge(nn.Module):
    """
    Thalamic Relay Bridge - inhibitory gating between channels.

    Architecture:
    1. Perceiver encodes to soft tokens
    2. Thalamic gating module applies channel-wise inhibition
    3. Reticular nucleus (TRN) creates winner-take-all competition

    Key insight: Not all dimensions are equally important. Thalamic
    gating learns which dimensions are task-relevant.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 gate_temperature: float = 1.0):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.gate_temperature = gate_temperature

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Main encoder
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Thalamic gating network
        # Maps each token to gate values across channels
        self.gate_proj = nn.Sequential(
            nn.Linear(tgt_dim, tgt_dim),
            nn.GELU(),
            nn.Linear(tgt_dim, tgt_dim),
        )

        # Reticular nucleus: lateral inhibition across tokens
        # Each token inhibits others based on activity
        self.reticular = nn.Linear(tgt_dim, num_latents)

        # Learnable baseline activity
        self.baseline = nn.Parameter(torch.zeros(1, 1, tgt_dim))

        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        logger.info(f"ThalamicRelayBridge: {num_latents} tokens with inhibitory gating")

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
        """
        B = src_hidden.shape[0]

        # Encode via Perceiver
        latents = self.resampler(src_hidden, src_mask)  # [B, K, D]

        # Compute gate values (per-dimension gating)
        gate_logits = self.gate_proj(latents)  # [B, K, D]

        # Sigmoid gating with temperature
        gates = torch.sigmoid(gate_logits / self.gate_temperature)  # [B, K, D]

        # Reticular inhibition: competition between tokens
        # Pool each token's activity
        token_activity = latents.abs().mean(dim=-1)  # [B, K]

        # Reticular output: how much each token inhibits others
        inhibition = self.reticular(latents.mean(dim=1))  # [B, K]
        inhibition = F.softmax(inhibition / self.gate_temperature, dim=-1)  # [B, K]

        # Apply inhibition (winner-take-all competition)
        inhibited_activity = token_activity * inhibition
        token_gates = inhibited_activity.unsqueeze(-1)  # [B, K, 1]

        # Combined gating: dimension gates * token gates
        combined_gates = gates * token_gates

        # Apply gating
        gated = latents * combined_gates + self.baseline * (1 - combined_gates)

        soft_tokens = gated

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Gating entropy (for monitoring)
        gate_entropy = -(gates * (gates + 1e-8).log()).mean()

        return soft_tokens.to(src_hidden.dtype), torch.tensor(0.0, device=soft_tokens.device), gate_entropy, soft_tokens.var()


# =============================================================================
# HAIL MARY EXPERIMENT 5: Domain Adversarial Bridge (DANN-style)
# =============================================================================
# Key insight: Use adversarial training to align soft token distribution
# with the target model's embedding distribution.
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainAdversarialBridge(nn.Module):
    """
    Domain Adversarial Bridge - align distributions via adversarial training.

    Architecture:
    1. Perceiver encodes to soft tokens
    2. Discriminator tries to distinguish soft tokens from real embeddings
    3. Gradient reversal makes encoder fool the discriminator

    Based on: Ganin et al. "Domain-Adversarial Training" (2016)
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 adv_weight: float = 0.1):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.adv_weight = adv_weight

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Main encoder
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Domain discriminator: distinguishes soft tokens from real embeddings
        self.discriminator = nn.Sequential(
            nn.Linear(tgt_dim, tgt_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(tgt_dim // 2, tgt_dim // 4),
            nn.GELU(),
            nn.Linear(tgt_dim // 4, 1),
        )

        # Learnable GRL alpha (annealed during training)
        self.register_buffer('grl_alpha', torch.tensor(1.0))

        # Reference embeddings from target model (set externally)
        self.register_buffer('reference_embeddings', None)

        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_latents = num_latents

        logger.info(f"DomainAdversarialBridge: {num_latents} tokens, adv_weight={adv_weight}")

    def set_reference_embeddings(self, embeddings: torch.Tensor):
        """Set reference embeddings from target model for discriminator training."""
        self.reference_embeddings = embeddings.detach()

    def compute_adversarial_loss(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial loss for domain alignment.

        soft_tokens should be classified as "fake" (from bridge)
        reference_embeddings should be classified as "real" (from target model)
        """
        B, K, D = soft_tokens.shape

        # Apply gradient reversal to soft tokens
        soft_reversed = GradientReversalFunction.apply(soft_tokens, self.grl_alpha)

        # Discriminator predictions for soft tokens (should be 0 = fake)
        soft_preds = self.discriminator(soft_reversed.view(-1, D)).view(B, K)  # [B, K]

        # If we have reference embeddings, use them
        if self.reference_embeddings is not None and self.reference_embeddings.numel() > 0:
            # Sample reference embeddings
            num_ref = min(B * K, self.reference_embeddings.shape[0])
            ref_idx = torch.randperm(self.reference_embeddings.shape[0])[:num_ref]
            ref_emb = self.reference_embeddings[ref_idx]

            # Discriminator predictions for reference (should be 1 = real)
            ref_preds = self.discriminator(ref_emb).squeeze(-1)  # [num_ref]

            # Binary cross entropy loss
            fake_labels = torch.zeros_like(soft_preds)
            real_labels = torch.ones_like(ref_preds)

            fake_loss = F.binary_cross_entropy_with_logits(soft_preds, fake_labels)
            real_loss = F.binary_cross_entropy_with_logits(ref_preds, real_labels)

            adv_loss = (fake_loss + real_loss) / 2
        else:
            # Without reference, just use soft tokens
            fake_labels = torch.zeros_like(soft_preds)
            adv_loss = F.binary_cross_entropy_with_logits(soft_preds, fake_labels)

        return adv_loss * self.adv_weight

    def anneal_grl_alpha(self, progress: float):
        """Anneal GRL alpha from 0 to 1 based on training progress."""
        # Gradual schedule from DANN paper
        self.grl_alpha.fill_(2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0)

    def forward(self, src_hidden: torch.Tensor, src_mask=None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
        Returns:
            soft_tokens: [B, num_latents, tgt_dim]
            adv_loss: adversarial alignment loss
        """
        # Encode
        soft_tokens = self.resampler(src_hidden, src_mask)

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Compute adversarial loss
        if self.training:
            adv_loss = self.compute_adversarial_loss(soft_tokens)
        else:
            adv_loss = torch.tensor(0.0, device=soft_tokens.device)

        return soft_tokens.to(src_hidden.dtype), adv_loss, self.grl_alpha.item(), soft_tokens.var()


# =============================================================================
# HAIL MARY EXPERIMENT 6: Successive Refinement Bridge
# =============================================================================
# Key insight: Layered encoding where each token refines the representation.
# Can use 2-16 tokens at inference time for bandwidth adaptation.
# =============================================================================

class SuccessiveRefinementBridge(nn.Module):
    """
    Successive Refinement Bridge - progressive token generation.

    Architecture:
    1. Base encoder produces first token
    2. Each subsequent token refines/corrects the representation
    3. Can use variable number of tokens at inference

    Key insight: This allows runtime bandwidth-accuracy tradeoff without
    retraining - use fewer tokens for fast inference, more for accuracy.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03,
                 max_tokens: int = 16):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.max_tokens = max_tokens

        num_latents = getattr(args, 'soft_tokens', 8)
        self.num_latents = min(num_latents, max_tokens)

        # Input projection
        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

        # Base encoder (produces first token/coarse representation)
        self.base_encoder = nn.Sequential(
            nn.Linear(tgt_dim, tgt_dim),
            nn.GELU(),
            nn.Linear(tgt_dim, tgt_dim),
        )

        # Refinement layers (each produces one refinement token)
        self.refinement_layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(tgt_dim, 8, batch_first=True),
                "ln1": nn.LayerNorm(tgt_dim),
                "refine": nn.Sequential(
                    nn.Linear(tgt_dim * 2, tgt_dim),  # Input: current state + source
                    nn.GELU(),
                    nn.Linear(tgt_dim, tgt_dim),
                ),
                "ln2": nn.LayerNorm(tgt_dim),
            }) for _ in range(max_tokens - 1)
        ])

        # Per-layer scale (how much each refinement contributes)
        self.refinement_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(max_tokens - 1)
        ])

        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        logger.info(f"SuccessiveRefinementBridge: max {max_tokens} tokens, using {self.num_latents}")

    def forward(self, src_hidden: torch.Tensor, src_mask=None, num_tokens: int = None):
        """
        Args:
            src_hidden: [B, seq_len, src_dim]
            num_tokens: How many tokens to generate (default: self.num_latents)
        Returns:
            soft_tokens: [B, num_tokens, tgt_dim]
        """
        if num_tokens is None:
            num_tokens = self.num_latents

        num_tokens = min(num_tokens, self.max_tokens)
        B = src_hidden.shape[0]

        # Project source
        src_proj = self.input_proj(src_hidden.float())
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        # Pool source for base encoding
        if src_mask is not None:
            mask_expanded = src_mask.unsqueeze(-1).float()
            pooled = (src_proj * mask_expanded).sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            pooled = src_proj.mean(dim=1)

        # Base token (coarse representation)
        base_token = self.base_encoder(pooled).unsqueeze(1)  # [B, 1, D]

        tokens = [base_token]
        current_repr = base_token

        # Progressive refinement
        for i in range(min(num_tokens - 1, len(self.refinement_layers))):
            layer = self.refinement_layers[i]
            scale = self.refinement_scales[i].abs()

            # Cross-attend to source
            x_norm = layer["ln1"](current_repr)
            attn_out, _ = layer["cross_attn"](
                query=x_norm, key=src_proj, value=src_proj,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )

            # Compute refinement
            combined = torch.cat([current_repr.squeeze(1), attn_out.squeeze(1)], dim=-1)
            refinement = layer["refine"](combined).unsqueeze(1)
            refinement = layer["ln2"](refinement)

            # Scale and add refinement token
            scaled_refinement = scale * refinement
            tokens.append(scaled_refinement)

            # Update current representation for next refinement
            current_repr = current_repr + scaled_refinement

        # Stack all tokens
        soft_tokens = torch.cat(tokens, dim=1)  # [B, num_tokens, D]

        # RMS normalization
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * self.output_scale

        # Refinement magnitude (for monitoring)
        if len(tokens) > 1:
            refinement_mag = sum(t.pow(2).mean() for t in tokens[1:]) / len(tokens[1:])
        else:
            refinement_mag = torch.tensor(0.0, device=soft_tokens.device)

        return soft_tokens.to(src_hidden.dtype), refinement_mag, float(num_tokens), soft_tokens.var()


# =============================================================================
# LEGACY: Ridge Regression Baseline (from LatentMAS - NOT NOVEL)
# =============================================================================

class RidgeRegressionBridge(nn.Module):
    """
    Training-free alignment via ridge regression (LatentMAS approach).

    Computes: W_a = (W_out^T @ W_out + λI)^{-1} @ W_out^T @ W_in

    This maps sender hidden states to receiver space without neural network training.
    """

    def __init__(self, sender_model, receiver_model, lambda_reg: float = 1e-4,
                 pooling: str = 'last'):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.pooling = pooling

        logger.info(f"Computing ridge regression alignment (lambda={lambda_reg})...")

        with torch.no_grad():
            # Get output projection from sender (lm_head)
            W_out = sender_model.lm_head.weight.detach().float()  # [vocab, hidden]

            # Get input embeddings from receiver
            W_in = receiver_model.model.embed_tokens.weight.detach().float()  # [vocab, hidden]

            # Align vocabulary sizes (use minimum)
            min_vocab = min(W_out.shape[0], W_in.shape[0])
            W_out = W_out[:min_vocab]
            W_in = W_in[:min_vocab]

            # Compute alignment matrix
            # W_a @ h_sender ≈ h_receiver
            hidden_dim = W_out.shape[1]
            WtW = W_out.T @ W_out  # [hidden, hidden]
            WtW_reg = WtW + lambda_reg * torch.eye(hidden_dim, device=WtW.device)

            # Solve: W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in
            W_a = torch.linalg.solve(WtW_reg, W_out.T @ W_in)

            self.register_buffer('alignment_matrix', W_a.half())

        logger.info(f"Ridge alignment matrix computed: {W_a.shape}")
        self.trainable_params = 0

    def forward(self, sender_hidden_states: torch.Tensor, src_mask=None) -> torch.Tensor:
        """
        Args:
            sender_hidden_states: [B, seq_len, hidden_dim] from sender
        Returns:
            aligned_states: [B, 1, hidden_dim] for receiver
        """
        if self.pooling == 'last':
            pooled = sender_hidden_states[:, -1, :]
        elif self.pooling == 'mean':
            if src_mask is not None:
                mask_expanded = src_mask.unsqueeze(-1).float()
                pooled = (sender_hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1)
            else:
                pooled = sender_hidden_states.mean(dim=1)
        elif self.pooling == 'max':
            pooled = sender_hidden_states.max(dim=1)[0]
        else:
            pooled = sender_hidden_states[:, -1, :]

        # Apply alignment
        aligned = pooled.float() @ self.alignment_matrix.float()
        return aligned.unsqueeze(1).half(), torch.tensor(0.0), 1.0, aligned.var()


# =============================================================================
# Experiment 2: Multi-Layer Extraction with Learned Weights
# =============================================================================

class MultiLayerExtractor(nn.Module):
    """
    Extract from multiple sender layers with learned combination weights.

    Based on Cache2Cache insight that different layers contain different information.
    """

    def __init__(self, extract_layers: List[int], hidden_dim: int = 4096):
        super().__init__()
        self.extract_layers = extract_layers
        self.num_layers = len(extract_layers)

        # Learnable layer weights (softmax-normalized)
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))

        # Per-layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in extract_layers
        ])

        logger.info(f"MultiLayerExtractor: extracting from layers {extract_layers}")

    def forward(self, hidden_states_dict: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states_dict: {layer_idx: [B, seq, hidden]}
        Returns:
            combined: [B, seq, hidden] weighted combination
            weights: [num_layers] learned weights
        """
        weights = F.softmax(self.layer_logits, dim=0)

        combined = 0
        for i, layer_idx in enumerate(self.extract_layers):
            h = hidden_states_dict[layer_idx]
            h = self.layer_norms[i](h)
            combined = combined + weights[i] * h

        return combined, weights


class MultiLayerBridge(nn.Module):
    """Bridge with multi-layer extraction."""

    def __init__(self, args, src_dim: int, tgt_dim: int,
                 extract_layers: List[int] = [16, 24, 31]):
        super().__init__()
        self.extractor = MultiLayerExtractor(extract_layers, src_dim)
        self.base_bridge = LatentBridge(args, src_dim, tgt_dim)

    def forward(self, hidden_states_dict: Dict[int, torch.Tensor], src_mask=None):
        combined, layer_weights = self.extractor(hidden_states_dict)
        soft_tokens, aux_loss, diversity, z_var = self.base_bridge(combined, src_mask)
        return soft_tokens, aux_loss, diversity, z_var, layer_weights


# =============================================================================
# Experiment 3: VIB (Variational Information Bottleneck) Bridge
# =============================================================================

class VIBLatentBridge(nn.Module):
    """
    Variational Information Bottleneck Bridge.

    Forces compression by adding stochasticity: z = mu + sigma * epsilon
    KL regularization encourages minimal sufficient statistics.
    """

    def __init__(self, args, src_dim: int, tgt_dim: int, target_rms: float = 0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

        num_latents = getattr(args, 'soft_tokens', 8)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 2)

        # Shared encoder
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)

        # Separate heads for mu and log_var
        self.mu_head = nn.Linear(tgt_dim, tgt_dim)
        self.logvar_head = nn.Linear(tgt_dim, tgt_dim)

        self.output_scale = nn.Parameter(torch.tensor(target_rms))

        logger.info(f"VIBLatentBridge: {num_latents} soft tokens with stochastic bottleneck")

    def forward(self, src_hidden: torch.Tensor, src_mask=None, deterministic: bool = False):
        # Encode
        h = self.resampler(src_hidden, src_mask)

        # Compute distribution parameters
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        log_var = torch.clamp(log_var, -10, 2)  # Numerical stability

        # Reparameterization trick
        if self.training and not deterministic:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # KL divergence: KL(N(mu, sigma) || N(0, 1))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # RMS normalization
        rms = torch.sqrt((z ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (z / rms) * self.output_scale

        return out, kl_loss, 1.0, z.var()


# =============================================================================
# Experiment 4: Curriculum Training (COCONUT-style)
# =============================================================================

@dataclass
class CurriculumStage:
    """Configuration for one curriculum stage."""
    name: str
    text_ratio: float  # 1.0 = all text, 0.0 = all latent
    soft_tokens: int
    steps: int


CLASSIFICATION_CURRICULUM = [
    CurriculumStage("text_full", 1.0, 0, 200),
    CurriculumStage("text_75", 0.75, 2, 300),
    CurriculumStage("text_50", 0.50, 4, 300),
    CurriculumStage("text_25", 0.25, 6, 400),
    CurriculumStage("latent_only", 0.0, 8, 500),
]


class CurriculumTrainer:
    """COCONUT-style curriculum from text to latent communication."""

    def __init__(self, curriculum: List[CurriculumStage]):
        self.curriculum = curriculum
        self.current_stage_idx = 0
        self.steps_in_stage = 0
        self.total_steps = sum(s.steps for s in curriculum)

    def get_current_stage(self) -> CurriculumStage:
        return self.curriculum[self.current_stage_idx]

    def step(self) -> bool:
        """Advance curriculum. Returns True if stage changed."""
        self.steps_in_stage += 1
        stage = self.get_current_stage()

        if self.steps_in_stage >= stage.steps:
            if self.current_stage_idx < len(self.curriculum) - 1:
                self.current_stage_idx += 1
                self.steps_in_stage = 0
                logger.info(f"Curriculum: advancing to stage '{self.get_current_stage().name}'")
                return True
        return False

    def get_text_ratio(self) -> float:
        return self.get_current_stage().text_ratio

    def get_num_soft_tokens(self) -> int:
        return self.get_current_stage().soft_tokens


# =============================================================================
# Experiment 5: Gumbel-Sigmoid Layer Gating (Cache2Cache style)
# =============================================================================

class GumbelSigmoidGate(nn.Module):
    """Learnable binary gate using Gumbel-sigmoid for differentiable selection."""

    def __init__(self, num_gates: int, init_temp: float = 2.0):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(num_gates))
        self.register_buffer('temperature', torch.tensor(init_temp))

    def forward(self, hard: bool = False) -> torch.Tensor:
        if self.training:
            # Gumbel-sigmoid
            u = torch.rand_like(self.gate_logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            gate = torch.sigmoid((self.gate_logits + gumbel) / self.temperature)
        else:
            # Hard gate at inference
            gate = (self.gate_logits > 0).float()

        return gate

    def anneal_temperature(self, step: int, total_steps: int,
                          start_temp: float = 2.0, end_temp: float = 0.1):
        """Exponential temperature annealing."""
        progress = step / max(total_steps, 1)
        new_temp = start_temp * (end_temp / start_temp) ** progress
        self.temperature.fill_(new_temp)


class GatedMultiLayerInjector(nn.Module):
    """Inject soft tokens at multiple layers with learned gates."""

    def __init__(self, inject_layers: List[int], hidden_dim: int = 4096):
        super().__init__()
        self.inject_layers = inject_layers
        self.gates = GumbelSigmoidGate(len(inject_layers))

        # Per-layer adapters
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in inject_layers
        ])

    def forward(self, soft_tokens: torch.Tensor, layer_idx: int) -> Optional[torch.Tensor]:
        if layer_idx not in self.inject_layers:
            return None

        i = self.inject_layers.index(layer_idx)
        gate = self.gates()[i]
        adapted = self.adapters[i](soft_tokens)
        return gate * adapted


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_ridge_regression_experiment(args, output_dir: Path):
    """Run ridge regression baseline comparison."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Ridge Regression Baseline (LatentMAS)")
    logger.info("=" * 60)

    results = {
        'experiment': 'ridge_regression',
        'lambda_values': [],
        'datasets': {}
    }

    # Test different lambda values
    lambda_values = [1e-6, 1e-4, 1e-2, 1.0]
    datasets = ['sst2', 'agnews']

    for dataset in datasets:
        results['datasets'][dataset] = {}

        for lambda_reg in lambda_values:
            logger.info(f"\nTesting lambda={lambda_reg} on {dataset}")

            # This would load models and run evaluation
            # For now, log the configuration
            config = {
                'lambda': lambda_reg,
                'pooling': 'last',
                'status': 'configured'
            }
            results['datasets'][dataset][f'lambda_{lambda_reg}'] = config

    results['lambda_values'] = lambda_values

    # Save results
    with open(output_dir / 'ridge_regression_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Ridge regression results saved to {output_dir}")
    return results


def run_multi_layer_experiment(args, output_dir: Path):
    """Run multi-layer extraction experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Multi-Layer Extraction with Learned Weights")
    logger.info("=" * 60)

    results = {
        'experiment': 'multi_layer_extraction',
        'layer_configs': [],
        'datasets': {}
    }

    # Test different layer combinations
    layer_configs = [
        [31],           # Single layer (baseline)
        [16, 31],       # Two layers
        [16, 24, 31],   # Three layers (recommended)
        [8, 16, 24, 31] # Four layers
    ]

    datasets = ['sst2', 'agnews']

    for dataset in datasets:
        results['datasets'][dataset] = {}

        for layers in layer_configs:
            layer_key = '_'.join(map(str, layers))
            logger.info(f"\nTesting layers {layers} on {dataset}")

            config = {
                'layers': layers,
                'status': 'configured'
            }
            results['datasets'][dataset][f'layers_{layer_key}'] = config

    results['layer_configs'] = layer_configs

    with open(output_dir / 'multi_layer_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Multi-layer results saved to {output_dir}")
    return results


def run_vib_experiment(args, output_dir: Path):
    """Run VIB regularization experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Variational Information Bottleneck")
    logger.info("=" * 60)

    results = {
        'experiment': 'vib_regularization',
        'beta_values': [],
        'datasets': {}
    }

    # Test different beta (KL weight) values
    beta_values = [0.0001, 0.001, 0.01, 0.1]
    datasets = ['sst2']

    for dataset in datasets:
        results['datasets'][dataset] = {}

        for beta in beta_values:
            logger.info(f"\nTesting beta={beta} on {dataset}")

            config = {
                'beta': beta,
                'beta_anneal': True,
                'status': 'configured'
            }
            results['datasets'][dataset][f'beta_{beta}'] = config

    results['beta_values'] = beta_values

    with open(output_dir / 'vib_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"VIB results saved to {output_dir}")
    return results


def run_curriculum_experiment(args, output_dir: Path):
    """Run curriculum training experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Curriculum Training (COCONUT-style)")
    logger.info("=" * 60)

    results = {
        'experiment': 'curriculum_training',
        'stages': [
            {'name': s.name, 'text_ratio': s.text_ratio,
             'soft_tokens': s.soft_tokens, 'steps': s.steps}
            for s in CLASSIFICATION_CURRICULUM
        ],
        'datasets': {}
    }

    datasets = ['sst2']

    for dataset in datasets:
        logger.info(f"\nRunning curriculum on {dataset}")

        trainer = CurriculumTrainer(CLASSIFICATION_CURRICULUM)
        stage_results = []

        for stage in CLASSIFICATION_CURRICULUM:
            stage_results.append({
                'stage': stage.name,
                'text_ratio': stage.text_ratio,
                'soft_tokens': stage.soft_tokens,
                'status': 'configured'
            })

        results['datasets'][dataset] = {'stages': stage_results}

    with open(output_dir / 'curriculum_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Curriculum results saved to {output_dir}")
    return results


def run_layer_gating_experiment(args, output_dir: Path):
    """Run Gumbel-sigmoid layer gating experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Gumbel-Sigmoid Layer Gating (C2C-style)")
    logger.info("=" * 60)

    results = {
        'experiment': 'layer_gating',
        'inject_layers': [0, 8, 16, 24],
        'temp_schedule': {
            'start': 2.0,
            'end': 0.1
        },
        'datasets': {}
    }

    datasets = ['sst2', 'agnews']
    inject_configs = [
        [0],           # Embedding only (baseline)
        [0, 16],       # Embedding + middle
        [0, 8, 16, 24] # Multiple layers
    ]

    for dataset in datasets:
        results['datasets'][dataset] = {}

        for layers in inject_configs:
            layer_key = '_'.join(map(str, layers))
            logger.info(f"\nTesting injection at layers {layers} on {dataset}")

            config = {
                'inject_layers': layers,
                'gumbel_sigmoid': True,
                'status': 'configured'
            }
            results['datasets'][dataset][f'inject_{layer_key}'] = config

    with open(output_dir / 'layer_gating_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Layer gating results saved to {output_dir}")
    return results


def run_moe_experiment(args, output_dir: Path):
    """Run Mixture-of-Experts Bridge experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Mixture-of-Experts Bridge (MoE)")
    logger.info("=" * 60)

    results = {
        'experiment': 'moe_bridge',
        'moe_configs': [],
        'datasets': {}
    }

    # Test different MoE configurations
    moe_configs = [
        {'num_experts': 4, 'top_k': 1, 'use_shared_expert': False, 'use_aux_loss_free': False},  # Minimal
        {'num_experts': 8, 'top_k': 2, 'use_shared_expert': False, 'use_aux_loss_free': False},  # Mixtral-style
        {'num_experts': 8, 'top_k': 2, 'use_shared_expert': True, 'use_aux_loss_free': False},   # + shared expert
        {'num_experts': 8, 'top_k': 2, 'use_shared_expert': True, 'use_aux_loss_free': True},    # DeepSeek-V3 style
    ]

    datasets = ['arc_easy', 'sst2']

    for dataset in datasets:
        results['datasets'][dataset] = {}

        for i, config in enumerate(moe_configs):
            config_name = f"moe_{config['num_experts']}x{config['top_k']}"
            if config['use_shared_expert']:
                config_name += "_shared"
            if config['use_aux_loss_free']:
                config_name += "_auxfree"

            logger.info(f"\nTesting {config_name} on {dataset}")

            result_config = {
                **config,
                'status': 'configured'
            }
            results['datasets'][dataset][config_name] = result_config

    results['moe_configs'] = moe_configs

    with open(output_dir / 'moe_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"MoE results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-Model Communication Experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['ridge', 'multi_layer', 'vib', 'curriculum',
                               'layer_gating', 'moe', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--output_dir', type=str, default='runs/cross_model_experiments',
                       help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Running experiment: {args.experiment}")

    # Run experiments
    all_results = {}

    if args.experiment in ['ridge', 'all']:
        all_results['ridge'] = run_ridge_regression_experiment(args, output_dir)

    if args.experiment in ['multi_layer', 'all']:
        all_results['multi_layer'] = run_multi_layer_experiment(args, output_dir)

    if args.experiment in ['vib', 'all']:
        all_results['vib'] = run_vib_experiment(args, output_dir)

    if args.experiment in ['curriculum', 'all']:
        all_results['curriculum'] = run_curriculum_experiment(args, output_dir)

    if args.experiment in ['layer_gating', 'all']:
        all_results['layer_gating'] = run_layer_gating_experiment(args, output_dir)

    if args.experiment in ['moe', 'all']:
        all_results['moe'] = run_moe_experiment(args, output_dir)

    # Save combined results
    with open(output_dir / 'all_experiments_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
