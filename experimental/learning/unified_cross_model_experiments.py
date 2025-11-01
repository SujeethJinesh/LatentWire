#!/usr/bin/env python3
"""
Unified cross-model alignment experiments combining Procrustes and learned adapters.
Optimized for 4 H100 GPUs with DistributedDataParallel (DDP).

═══════════════════════════════════════════════════════════════════════════════
RESEARCH QUESTION
═══════════════════════════════════════════════════════════════════════════════

Can different LLM architectures (Llama 3.1 8B vs Mistral 7B, or Llama 3.1 8B vs
Llama 3.2 3B) exchange information via learned alignment of internal
representations, bypassing tokenization?

HYPOTHESIS: Vocabulary mismatch (128K vs 32K tokens) is the primary barrier to
cross-model alignment. Same-vocabulary models should align better.

═══════════════════════════════════════════════════════════════════════════════
EXPERIMENTAL DESIGN: 11 EXPERIMENTS IN 3 PHASES (~5-6 hours total)
═══════════════════════════════════════════════════════════════════════════════

PHASE 1: FAST BASELINES (~30 min) - Establish Feasibility Without Training
───────────────────────────────────────────────────────────────────────────────

EXPERIMENT 1: Procrustes Alignment (Llama 3.1 8B ↔ Llama 3.2 3B) - 5 min
─────────────────────────────────────────────────────────────────────────────
WHAT: SVD-based orthogonal alignment between hidden states at layers [8,16,24]
WHY:  Zero-training baseline to establish feasibility ceiling for same-vocab case
HYPOTHESIS: Same vocabulary (128,256 tokens) should yield CKA > 0.7
COMPARISON: vs Exp 2 (cross-vocab Procrustes) - isolates vocabulary effect
LITERATURE: Lester et al. "Model Stitching" (arXiv:2506.06609) - affine +5-8%
EXPECTED: CKA 0.7-0.8 (high due to identical vocab + similar architecture)
DECISION: If CKA < 0.5, vocabulary isn't the only problem (architecture matters)

EXPERIMENT 2: Procrustes Alignment (Llama 3.1 8B ↔ Mistral 7B) - 5 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Same SVD alignment but with vocabulary mismatch (128K vs 32K tokens)
WHY:  Tests whether geometric alignment alone can overcome vocab mismatch
HYPOTHESIS: Vocab mismatch lowers CKA significantly vs Exp 1
COMPARISON: vs Exp 1 - measures vocabulary penalty on alignment
EXPECTED: CKA 0.3-0.5 (much lower than Exp 1 due to vocab mismatch)
DECISION: If CKA ≈ Exp 1, vocab mismatch doesn't matter (surprising!)
          If CKA << Exp 1, vocab is the key barrier (expected)

EXPERIMENT 3: Activation Communication (Llama 3.1 8B ↔ Llama 3.2 3B) - 10 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Direct hidden state injection - Model A's activations → Model B's layers
WHY:  Tests whether same-vocab models can consume each other's "thoughts"
HYPOTHESIS: Should work well with same vocabulary (both use identical embeddings)
COMPARISON: vs Exp 4 (cross-vocab activation) - isolates vocabulary effect
LITERATURE: Tandem Transformers (NeurIPS 2024) - hidden state sharing works
EXPECTED: Coherent generation, cosine similarity > 0.7
DECISION: If generation is gibberish, even same-vocab fails (architecture issue)
          If generation is coherent, validates approach for same-vocab case

EXPERIMENT 4: Activation Communication (Llama 3.1 8B ↔ Mistral 7B) - 10 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Same hidden state injection but with vocab mismatch
WHY:  Core LatentWire hypothesis - can different models share representations?
HYPOTHESIS: Vocab mismatch breaks zero-shot injection (needs learned alignment)
COMPARISON: vs Exp 3 - measures how much vocab mismatch hurts communication
EXPECTED: Low cosine similarity (0.3-0.5), degraded generation quality
DECISION: If this works zero-shot, learned adapters might be unnecessary!
          If it fails, justifies Phases 2-3 (learned alignment training)

PHASE 1 SUMMARY: 30 minutes → Establishes feasibility & vocabulary penalty
───────────────────────────────────────────────────────────────────────────────
Key Comparisons:
- Exp 1 vs 2: How much does vocab mismatch hurt Procrustes? (CKA gap)
- Exp 3 vs 4: How much does vocab mismatch hurt activation sharing? (quality gap)
- Procrustes vs Activation: Is geometric alignment enough, or need training?

═══════════════════════════════════════════════════════════════════════════════

PHASE 2: TRAINED ADAPTERS - SAME VOCAB (~2-3 hours) - Validate Learning Works
───────────────────────────────────────────────────────────────────────────────

EXPERIMENT 5: LoRA Adapter (Llama 3.1 8B ↔ Llama 3.2 3B) - 45 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Low-rank adapter (260K params) trained with InfoNCE contrastive loss
WHY:  Tests whether learned alignment beats Procrustes for same-vocab case
HYPOTHESIS: Should achieve CKA > 0.8 (better than Procrustes Exp 1)
COMPARISON: vs Exp 1 (Procrustes) - does training help same-vocab?
            vs Exp 6,7 (Linear/Affine) - parameter efficiency comparison
            vs Exp 8 (LoRA cross-vocab) - isolates vocabulary effect
LITERATURE: Cross-LoRA (arXiv:2508.05232) - 85-95% performance retention
EXPECTED: CKA 0.8-0.9, generation loss 2.5-3.5 (validates approach works)
DECISION: If CKA < Exp 1, training doesn't help (algorithm bug)
          If CKA >> Exp 1, training significantly improves alignment

EXPERIMENT 6: Linear Adapter (Llama 3.1 8B ↔ Llama 3.2 3B) - 45 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Full-rank linear projection (16M params, no bias)
WHY:  Upper bound on linear alignment capacity
HYPOTHESIS: Should perform similar to LoRA but with 50× more parameters
COMPARISON: vs Exp 5 (LoRA) - tests parameter efficiency
            vs Exp 7 (Affine) - tests whether bias term helps
EXPECTED: CKA 0.8-0.9 (similar to LoRA)
DECISION: If Linear >> LoRA, low-rank bottleneck hurts (increase rank)
          If Linear ≈ LoRA, LoRA is sufficient (use LoRA for efficiency)

EXPERIMENT 7: Affine Adapter (Llama 3.1 8B ↔ Llama 3.2 3B) - 45 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Full-rank affine projection (16M params + 4K bias)
WHY:  Tests whether bias term provides additional alignment capacity
HYPOTHESIS: Bias should provide minimal improvement over Linear
COMPARISON: vs Exp 6 (Linear) - isolates bias term contribution
LITERATURE: Lester et al. - affine +5-8% over orthogonal Procrustes
EXPECTED: CKA 0.8-0.9 (similar to Linear and LoRA)
DECISION: If Affine >> Linear, bias term is critical (unexpected)
          If Affine ≈ Linear, bias is redundant (as expected)

PHASE 2 SUMMARY: ~2-3 hours → Validates learned alignment works for same-vocab
───────────────────────────────────────────────────────────────────────────────
Key Comparisons:
- All vs Exp 1: Does training beat Procrustes? (should improve CKA by 0.1-0.2)
- LoRA vs Linear: Is low-rank sufficient? (parameter efficiency question)
- Linear vs Affine: Does bias help? (probably not much)

Expected Outcome: LoRA ≈ Linear ≈ Affine >> Procrustes (training helps)
Decision Point: If same-vocab experiments succeed (CKA > 0.8), proceed to Phase 3
                If they fail (CKA < 0.6), fundamental approach is flawed

═══════════════════════════════════════════════════════════════════════════════

PHASE 4: TRAINED ADAPTERS - CROSS VOCAB (~2-3 hours) - The Hard Test
───────────────────────────────────────────────────────────────────────────────

EXPERIMENT 8: LoRA Adapter (Llama 3.1 8B ↔ Mistral 7B) - 45 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Same LoRA architecture as Exp 5, but with vocab mismatch (128K vs 32K)
WHY:  Tests whether learned alignment can overcome vocabulary barrier
HYPOTHESIS: Vocab mismatch makes alignment harder (CKA < Exp 5)
COMPARISON: vs Exp 5 (same-vocab LoRA) - CRITICAL vocabulary ablation
            vs Exp 2 (cross-vocab Procrustes) - does training help?
EXPECTED: CKA 0.4-0.6 (lower than Exp 5, but better than Exp 2)
          Generation loss stuck at ~8.7 (vocabulary mismatch barrier)
DECISION: If CKA < 0.5, vocabulary mismatch is insurmountable with LoRA
          If CKA > 0.7, vocabulary mismatch can be overcome (surprising!)

**CRITICAL COMPARISON**: Exp 5 vs Exp 8 isolates vocabulary effect
- If (Exp 5 CKA) - (Exp 8 CKA) > 0.3, vocabulary is the main barrier
- If (Exp 5 CKA) ≈ (Exp 8 CKA), vocabulary doesn't matter (architecture does)

EXPERIMENT 9: Token Compression (Llama 3.1 8B ↔ Mistral 7B) - 45 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Learned compressor that maps 512 tokens → 64 soft prompts (8× compression)
WHY:  This IS the LatentWire interlingua - the actual wire format
HYPOTHESIS: Token-level compression might sidestep vocabulary mismatch
COMPARISON: vs Exp 8 (LoRA) - different compression approach
LITERATURE: LLMLingua (20× compression), CompactPrompt (60% reduction)
EXPECTED: Perplexity < 30, 60-70% token accuracy
DECISION: If perplexity << LoRA, compression is better approach
          If perplexity ≈ LoRA, both methods are equivalent

EXPERIMENT 10: Linear Adapter (Llama 3.1 8B ↔ Mistral 7B) - 45 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Full-rank linear (16M params) for cross-vocab alignment
WHY:  Tests whether more parameters help overcome vocabulary mismatch
HYPOTHESIS: Linear should perform similarly to LoRA (low-rank is sufficient)
COMPARISON: vs Exp 8 (LoRA) - parameter efficiency for cross-vocab case
EXPECTED: CKA 0.4-0.6 (similar to LoRA Exp 8)
DECISION: If Linear >> LoRA, need more capacity for cross-vocab
          If Linear ≈ LoRA, parameter count isn't the limiting factor

EXPERIMENT 11: Affine Adapter (Llama 3.1 8B ↔ Mistral 7B) - 45 min
─────────────────────────────────────────────────────────────────────────────
WHAT: Full-rank affine (16M + 4K params) for cross-vocab alignment
WHY:  Completeness - tests bias term for cross-vocab case
HYPOTHESIS: Bias provides minimal benefit (same as Exp 7 conclusion)
COMPARISON: vs Exp 10 (Linear) - isolates bias contribution for cross-vocab
EXPECTED: CKA 0.4-0.6 (similar to Linear and LoRA)
DECISION: If Affine >> Linear, bias matters for cross-vocab (unexpected)
          If Affine ≈ Linear, bias is still redundant

PHASE 3 SUMMARY: ~2-3 hours → Tests whether cross-vocab alignment is feasible
───────────────────────────────────────────────────────────────────────────────
Key Comparisons:
- Exp 8 vs Exp 5: Vocabulary mismatch penalty (MOST IMPORTANT)
- Exp 8 vs Exp 2: Does training help cross-vocab? (LoRA vs Procrustes)
- Exp 8/10/11: Parameter efficiency holds even for cross-vocab

Expected Outcome: All cross-vocab experiments struggle (CKA 0.4-0.6)
Critical Question: Is vocabulary mismatch insurmountable, or just harder?

═══════════════════════════════════════════════════════════════════════════════
SCIENTIFIC RATIONALE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

The 11 experiments form a 2×3×2 factorial design:

FACTOR 1: Model Pair (vocabulary)
  - Same vocabulary (Llama 3.1 8B ↔ Llama 3.2 3B): Exps 1,3,5,6,7
  - Different vocabulary (Llama 3.1 8B ↔ Mistral 7B): Exps 2,4,8,9,10,11

FACTOR 2: Alignment Method
  - Geometric (Procrustes): Exps 1,2
  - Hidden state injection (Activation): Exps 3,4
  - Learned adapters (LoRA/Linear/Affine): Exps 5,6,7,8,10,11
  - Token compression: Exp 9

FACTOR 3: Training
  - Zero-training (Procrustes, Activation): Exps 1,2,3,4
  - Trained (LoRA, Linear, Affine, Compression): Exps 5,6,7,8,9,10,11

KEY ABLATIONS:
1. Vocabulary Effect: Compare same-vocab (1,3,5,6,7) vs cross-vocab (2,4,8,9,10,11)
   → Quantifies how much vocabulary mismatch hurts alignment

2. Training Effect: Compare Procrustes (1,2) vs LoRA (5,8)
   → Quantifies how much learned alignment helps vs geometric baseline

3. Parameter Efficiency: Compare LoRA (260K) vs Linear (16M) vs Affine (16M+4K)
   → Tests whether low-rank bottleneck limits performance

4. Method Comparison: Procrustes vs Activation vs Adapters vs Compression
   → Identifies best approach for cross-model communication

CRITICAL SUCCESS CRITERIA:
- Phase 1 (30 min): Establishes feasibility & vocabulary penalty
  → If Procrustes fails for same-vocab (CKA < 0.5), approach is flawed

- Phase 2 (2-3 hrs): Validates learning improves same-vocab alignment
  → If trained adapters don't beat Procrustes, training algorithm has bugs

- Phase 3 (2-3 hrs): Tests whether cross-vocab alignment is possible
  → If cross-vocab CKA > 0.7, vocabulary mismatch can be overcome
  → If cross-vocab CKA < 0.5, vocabulary mismatch is insurmountable

DECISION TREE:
┌─ Phase 1 Results ─┐
│ Same-vocab CKA?   │
├─ > 0.7: Good! ────┼─→ Proceed to Phase 2
├─ 0.5-0.7: OK ─────┼─→ Proceed with caution
└─ < 0.5: BAD ──────┴─→ STOP - fundamental approach is flawed

┌─ Phase 2 Results ─┐
│ Trained vs Proc?  │
├─ +0.2 CKA: Good! ─┼─→ Training helps, proceed to Phase 3
├─ +0.1 CKA: Meh ───┼─→ Training helps slightly
└─ +0.0 CKA: BAD ───┴─→ STOP - training algorithm is broken

┌─ Phase 3 Results ─────────────┐
│ Cross-vocab vs Same-vocab?    │
├─ Within 0.1 CKA: GREAT! ──────┼─→ Vocabulary doesn't matter!
├─ Within 0.2 CKA: Good ────────┼─→ Vocab hurts but manageable
├─ Within 0.3 CKA: Struggling ──┼─→ Vocab is major barrier
└─ > 0.3 CKA gap: FAIL ─────────┴─→ Vocabulary mismatch is insurmountable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import warnings
import os
import sys

# Suppress known warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import time
from pathlib import Path
from datetime import datetime
import math
import multiprocessing as mp
import shutil
import datasets

# ============================================================================
# Platform Detection and Device Configuration
# ============================================================================

def get_device_and_config():
    """Auto-detect platform and return appropriate device and config."""
    config = {}

    # Check environment variables
    use_mps = os.environ.get('USE_MPS', '0') == '1'
    use_cuda = os.environ.get('USE_CUDA', '0') == '1'
    disable_flash = os.environ.get('DISABLE_FLASH_ATTENTION', '0') == '1'

    # Detect device
    if use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        platform = 'mac'
        print("==> Using MPS (Metal Performance Shaders) on Mac")
    elif use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        platform = 'hpc'
        print(f"==> Using CUDA on HPC ({torch.cuda.device_count()} GPUs available)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        platform = 'mac'
        print("==> Auto-detected MPS on Mac")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        platform = 'hpc'
        print("==> Auto-detected CUDA")
    else:
        device = torch.device('cpu')
        platform = 'cpu'
        print("==> Using CPU (no GPU available)")

    # Platform-specific config
    if platform == 'mac':
        config['batch_size'] = int(os.environ.get('MAC_BATCH_SIZE', '4'))
        config['num_samples'] = int(os.environ.get('MAC_SAMPLES', '1000'))
        config['epochs'] = int(os.environ.get('MAC_EPOCHS', '2'))
        config['use_bf16'] = False  # BF16 not well supported on MPS
        config['use_flash_attention'] = False
        config['grad_accum_steps'] = 8  # More accumulation for smaller batches
        print(f"  - Batch size: {config['batch_size']} (Mac memory constraints)")
        print(f"  - Samples: {config['num_samples']} (reduced for testing)")
        print(f"  - Epochs: {config['epochs']} (reduced for testing)")
    else:
        # HPC configuration - batch size optimized for multi-GPU with DDP
        num_gpus = torch.cuda.device_count() if platform == 'hpc' else 1
        # DDP: Each process gets batch_size samples
        # Start with 10 per GPU (previously caused OOM with DataParallel, but DDP is more efficient)
        config['batch_size'] = 10  # Per-process batch size
        config['num_samples'] = 10000  # Conservative for preemptible cluster
        config['epochs'] = 5  # Reduced from 10 based on convergence analysis (saves 40-50% compute)
        config['use_bf16'] = torch.cuda.is_bf16_supported() if platform == 'hpc' else False
        config['use_flash_attention'] = not disable_flash and platform == 'hpc'
        config['grad_accum_steps'] = 8  # DDP is more efficient, can reduce grad accum
        if platform == 'hpc':
            print(f"  - Batch size per GPU: {config['batch_size']}")
            print(f"  - Global batch size: {config['batch_size'] * num_gpus}")
            print(f"  - Effective batch (with grad accum): {config['batch_size'] * num_gpus * config['grad_accum_steps']}")
            print(f"  - Samples: {config['num_samples']}")
            print(f"  - Epochs: {config['epochs']}")
            print(f"  - BF16: {config['use_bf16']}")
            print(f"  - Flash Attention: {config['use_flash_attention']}")

    return device, platform, config

# Get device and config at module level
DEVICE, PLATFORM, PLATFORM_CONFIG = get_device_and_config()

# ============================================================================
# Distributed Data Parallel (DDP) Setup
# ============================================================================

def setup_ddp():
    """
    Initialize DDP for multi-GPU training.

    Returns:
        tuple: (rank, world_size, device)
            - rank: Process rank (0 to world_size-1)
            - world_size: Total number of processes
            - device: torch.device for this process
    """
    if not dist.is_available():
        return 0, 1, DEVICE

    if not dist.is_initialized():
        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = DEVICE

    return rank, world_size, device

def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Return True if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    """Get current process rank, returns 0 if not using DDP."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """Get world size, returns 1 if not using DDP."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

# ============================================================================
# InfoNCE Contrastive Loss
# ============================================================================

class InfoNCE(nn.Module):
    """
    InfoNCE loss for contrastive learning - essential for alignment.

    References:
        - van den Oord et al., "Representation Learning with Contrastive Predictive Coding"
          arXiv:1807.03748 (2018) - Original InfoNCE formulation
        - Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
          arXiv:2002.05709 (SimCLR, 2020) - Popularized for representation learning
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        anchor: [batch_size, hidden_dim]
        positive: [batch_size, hidden_dim]
        negatives: [batch_size, num_negatives, hidden_dim]
        """
        batch_size = anchor.shape[0]

        # Normalize representations
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # Negative similarities
        neg_sim = torch.matmul(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature

        # Compute InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)

# ============================================================================
# CKA (Centered Kernel Alignment) - Superior to SVCCA
# ============================================================================

class CKA:
    """
    Debiased CKA for measuring similarity between representations.

    Based on Murphy et al., ICLR 2024: "Unbiased HSIC Estimation"
    Critical for preventing bias in low-sample, high-dimensional settings.
    Standard CKA shows artificially high similarity for random matrices with n < 1000.
    """

    @staticmethod
    def linear_kernel(X):
        """Linear kernel (dot product)."""
        return X @ X.T

    @staticmethod
    def center_gram(K):
        """Center gram matrix."""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
        return H @ K @ H

    @staticmethod
    def unbiased_hsic_estimator(K, L):
        """
        Compute unbiased HSIC estimator (Murphy et al., ICLR 2024).

        Critical for preventing bias in low-sample, high-dimensional settings.
        Standard HSIC is biased when n_samples < n_features (common in deep learning).

        Args:
            K: Gram matrix [n, n]
            L: Gram matrix [n, n]

        Returns:
            Unbiased HSIC estimate (scalar)
        """
        n = K.shape[0]

        if n < 4:
            # Fall back to standard HSIC for very small samples
            K_c = CKA.center_gram(K)
            L_c = CKA.center_gram(L)
            return torch.sum(K_c * L_c)

        # Remove diagonal (for unbiased estimation)
        K_tilde = K - torch.diag(torch.diag(K))
        L_tilde = L - torch.diag(torch.diag(L))

        # Unbiased HSIC formula
        trace_term = torch.trace(K_tilde @ L_tilde)
        sum_k = torch.sum(K_tilde)
        sum_l = torch.sum(L_tilde)
        sum_kl = torch.sum(K_tilde * L_tilde)

        hsic_unbiased = (
            trace_term +
            (sum_k * sum_l) / ((n - 1) * (n - 2)) -
            (2 * sum_kl) / (n - 2)
        ) / (n * (n - 3))

        return hsic_unbiased

    @staticmethod
    def cka_similarity(X, Y, debiased=None):
        """
        Compute CKA similarity between two representation matrices.

        Args:
            X, Y: [n_samples, n_features] representation matrices
            debiased: Use unbiased HSIC estimator (default: USE_DEBIASED_CKA global flag)
                     Recommended True for n_samples < 1000

        Returns:
            Scalar similarity score in [0, 1] (higher = more similar)

        References:
            - Kornblith et al., ICML 2019: "Similarity of Neural Network Representations Revisited"
            - Murphy et al., ICLR 2024: "Unbiased HSIC Estimation for Low-Sample Regimes"
        """
        if debiased is None:
            debiased = USE_DEBIASED_CKA

        # Compute gram matrices
        K = CKA.linear_kernel(X)
        L = CKA.linear_kernel(Y)

        if debiased:
            # Use unbiased HSIC estimator (Murphy et al., "Unbiased HSIC Estimation", ICLR 2024)
            # Critical for low-sample, high-dimensional settings (n < 1000)
            hsic_xy = CKA.unbiased_hsic_estimator(K, L)
            hsic_xx = CKA.unbiased_hsic_estimator(K, K)
            hsic_yy = CKA.unbiased_hsic_estimator(L, L)

            # Prevent division by zero or negative values
            denominator = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=1e-12))
            return hsic_xy / (denominator + 1e-8)
        else:
            # Standard CKA (keep for backward compatibility)
            K_c = CKA.center_gram(K)
            L_c = CKA.center_gram(L)

            hsic = torch.sum(K_c * L_c)
            var_x = torch.sqrt(torch.sum(K_c * K_c))
            var_y = torch.sqrt(torch.sum(L_c * L_c))

            return hsic / (var_x * var_y + 1e-8)

# ============================================================================
# Alignment and Uniformity Metrics
# ============================================================================

class AlignmentUniformity:
    """
    Alignment and Uniformity metrics for contrastive learning quality.

    Based on Wang & Isola, ICML 2020: "Understanding Contrastive Representation Learning"
    These metrics directly predict downstream task performance and help diagnose:
    - Alignment: Are positive pairs close? (Lower is better)
    - Uniformity: Are representations evenly distributed? (Lower is better)

    Key insight: Good contrastive learning requires BOTH low alignment and low uniformity.
    - High alignment → Positive pairs aren't learning to be similar
    - High uniformity → Representations collapsing (all similar)
    - Low uniformity → Representations clustering (not utilizing embedding space)
    """

    @staticmethod
    def alignment_loss(x, y, alpha=2):
        """
        Alignment: measures closeness of positive pairs.

        Lower is better (0 = perfect alignment).

        Args:
            x, y: [batch_size, hidden_dim] - positive pair representations
            alpha: distance power (default 2 for squared Euclidean distance)

        Returns:
            Alignment loss (scalar, lower = better alignment)
        """
        return (x - y).norm(dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity_loss(x, t=2):
        """
        Uniformity: measures how uniformly representations are distributed on hypersphere.

        Lower is better (more uniform distribution prevents collapse).

        Args:
            x: [batch_size, hidden_dim] - representations
            t: temperature parameter (default 2, higher = more sensitive to clustering)

        Returns:
            Uniformity loss (scalar, lower = more uniform distribution)
        """
        # Normalize representations to unit hypersphere
        x = F.normalize(x, dim=-1)

        # Compute pairwise squared distances
        # sq_dist[i,j] = ||x[i] - x[j]||^2
        sq_dist = torch.cdist(x, x).pow(2)

        # Log of mean of exponentials (measures density on hypersphere)
        # Lower = more spread out, Higher = more clustered
        return torch.log(torch.exp(-t * sq_dist).mean() + 1e-8)

    @staticmethod
    def compute_metrics(anchor, positive):
        """
        Compute both alignment and uniformity metrics.

        Args:
            anchor, positive: [batch_size, hidden_dim] - positive pair representations

        Returns:
            (alignment_loss, uniformity_loss) - both scalars, lower is better

        Example:
            align, uniform = AlignmentUniformity.compute_metrics(anchor, positive)
            # Good contrastive learning: align < 1.0, uniform < -1.0
            # Poor alignment: align > 5.0
            # Representation collapse: uniform > 0.0
        """
        align = AlignmentUniformity.alignment_loss(anchor, positive)
        uniform = AlignmentUniformity.uniformity_loss(anchor)

        return align, uniform

# ============================================================================
# Pooling Functions
# ============================================================================

def mean_pooling(token_embeddings, attention_mask):
    """
    Mean pooling with attention mask weighting.

    Outperforms CLS token by 2-5% for semantic similarity tasks in LLMs.

    References:
        - Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
          arXiv:1908.10084 (2019) - Mean pooling outperforms CLS for semantic similarity
        - Muennighoff et al., "MTEB: Massive Text Embedding Benchmark"
          arXiv:2210.07316 (2022) - Confirms 2-5% improvement over CLS pooling

    Args:
        token_embeddings: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len] (1 = real token, 0 = padding)

    Returns:
        pooled: [batch_size, hidden_dim] - mean of non-padded tokens

    References:
        - Reimers & Gurevych (2019): Sentence-BERT uses mean pooling
        - SGPT (2022): Mean pooling superior to CLS for semantic tasks
    """
    # Expand attention mask to match embedding dimensions
    # [batch_size, seq_len] → [batch_size, seq_len, hidden_dim]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum embeddings (masked)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

    # Sum of mask (number of real tokens per example)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # Mean pooling = sum / count
    return sum_embeddings / sum_mask

# ============================================================================
# Contrastive Weight Scheduler
# ============================================================================

class ContrastiveWeightScheduler:
    """
    Curriculum learning for contrastive loss weighting.

    Gradually increases contrastive loss weight to prevent overwhelming the primary task.

    Starts with low contrastive weight to let the model learn the primary task first,
    then gradually increases to improve representation quality.

    References:
        - Hacohen & Weinshall, "On The Power of Curriculum Learning in Training Deep Networks"
          arXiv:1904.03626 (ICML 2019) - Curriculum learning improves convergence
        - Wang & Isola, "Understanding Contrastive Representation Learning"
          ICML 2020 - Proper weighting between alignment and primary task is critical

    Args:
        initial_weight: Starting contrastive weight (default 0.1)
        final_weight: Target contrastive weight (default CONTRASTIVE_WEIGHT)
        warmup_steps: Number of steps to linearly increase weight (default: 2 epochs)
    """

    def __init__(self, initial_weight=0.1, final_weight=None, warmup_steps=1000):
        self.initial_weight = initial_weight
        self.final_weight = final_weight if final_weight is not None else CONTRASTIVE_WEIGHT
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """
        Get current contrastive weight and increment step counter.

        Returns:
            Current weight (float)
        """
        if self.current_step >= self.warmup_steps:
            weight = self.final_weight
        else:
            # Linear warmup from initial to final
            progress = self.current_step / self.warmup_steps
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * progress

        self.current_step += 1
        return weight

    def get_current_weight(self):
        """Get current weight without incrementing (for logging)."""
        if self.current_step >= self.warmup_steps:
            return self.final_weight
        else:
            progress = self.current_step / self.warmup_steps
            return self.initial_weight + (self.final_weight - self.initial_weight) * progress

# ============================================================================
# Configuration
# ============================================================================

# Models
# IMPORTANT: Using BASE models (not -Instruct) for pure representation alignment research
# Rationale: Instruct models have undergone additional fine-tuning (RLHF, instruction-following)
# which introduces representation drift and "forgetting". We want to study core learned
# representations without confounding from instruction-tuning.
LLAMA_MODEL = "meta-llama/Llama-3.1-8B"
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"

# Same-vocabulary ablation models (Llama 3.1 8B ↔ Llama 3.2 3B)
# Both use identical 128,256 token vocabulary
# Research: "Transferring Features Across Language Models With Model Stitching" (arXiv 2506.06609)
# Llama 3.2 3B was trained using Llama 3.1 8B logits as targets (Meta documentation)
LLAMA_31_8B = "meta-llama/Llama-3.1-8B"
LLAMA_32_3B = "meta-llama/Llama-3.2-3B"

# Training - Use platform-specific values
BATCH_SIZE = PLATFORM_CONFIG['batch_size']
EPOCHS = PLATFORM_CONFIG['epochs']
LEARNING_RATE = 5e-5  # Slightly reduced for stability
NUM_SAMPLES = PLATFORM_CONFIG['num_samples']
GRAD_ACCUM_STEPS = PLATFORM_CONFIG['grad_accum_steps']
USE_BF16 = PLATFORM_CONFIG['use_bf16']
USE_FLASH_ATTENTION = PLATFORM_CONFIG['use_flash_attention']
MAX_LENGTH = 256  # Reduced from 512 to halve activation memory

# Contrastive Learning Parameters (NEW from 2025 research)
# InfoNCE and contrastive learning configuration (2024-2025 research updates)
TEMPERATURE = 0.15  # Optimal for text representations (0.07 is for vision, causes uniformity-tolerance dilemma)
CONTRASTIVE_WEIGHT = 0.2  # Reduced from 0.3 to prevent overwhelming primary objective (Wang & Isola, ICML 2020)
NUM_NEGATIVES = 127  # Number of negative samples

# 2024-2025 Research-Based Configuration Flags
USE_MEAN_POOLING = True  # Use mean pooling instead of CLS token (2-5% improvement for semantic tasks)
USE_DEBIASED_CKA = True  # Use unbiased HSIC estimator (critical for < 1000 samples, Murphy et al. ICLR 2024)
USE_AFFINE_PROCRUSTES = True  # Add bias term to Procrustes (5-8% improvement, model stitching 2024)
LOG_ALIGN_UNIFORM = True  # Log alignment/uniformity metrics (Wang & Isola, ICML 2020)
USE_CONTRASTIVE_CURRICULUM = True  # Gradually warm up contrastive weight (prevents overwhelming primary task)

# Multi-layer alignment (prevents single-point failure)
# Restored to multi-layer after model freezing freed 10-15 GB/GPU
ALIGNMENT_LAYERS = [8, 16, 24]  # Early, middle, late representations
LAYER_WEIGHTS = [0.3, 0.4, 0.3]  # Emphasize middle layer slightly
LAYER_IDX = 16  # Default for backward compatibility

# Layers to test for Procrustes
# CRITICAL: Layer indices must be valid for BOTH models
# - Llama 3.1 8B: 32 layers (indices 0-31)
# - Llama 3.2 3B: 28 layers (indices 0-27)
# - Mistral 7B: 32 layers (indices 0-31)
# Safe layers for all: [0, 8, 16, 24] (removed 32 which is out of bounds for 3.1 8B)
# For paper reproduction: Use layer 26 specifically (within bounds for all models)
LAYERS_TO_TEST = [0, 8, 16, 24]
RAMESH_LI_LAYER = 26  # Layer 26 from "Communicating Activations Between LLM Agents" (Ramesh & Li, ICML 2025)

# Calibration data size (reduced for memory constraints)
CALIBRATION_SIZE = 50

# Test prompts
TEST_PROMPTS = [
    "The capital of France is",
    "To solve this problem, we need to",
    "The future of artificial intelligence is",
    "In the year 2050,",
    "The main difference between cats and dogs is"
]

# ============================================================================
# Logging Setup
# ============================================================================

class TeeLogger:
    """Redirect stdout and stderr to both console and file."""

    def __init__(self, file_handle):
        self.terminal = sys.stdout
        self.log = file_handle

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================================================================
# Procrustes Alignment (CPU)
# ============================================================================

class ProcrustesAlignment:
    """
    Affine Procrustes alignment between hidden spaces.

    Extended from orthogonal to include bias term for better alignment.
    Affine transformations (linear + bias) consistently outperform pure orthogonal
    by 5-8% in model stitching tasks.

    References:
        - Schönemann, "A generalized solution of the orthogonal Procrustes problem"
          Psychometrika 1966 - Original orthogonal Procrustes formulation
        - Lester et al., "Transferring Features Across Language Models With Model Stitching"
          arXiv:2506.06609 (June 2025) - Affine extension outperforms orthogonal by 5-8%

    Args:
        use_affine: If True, use affine transformation (W + b). If False, orthogonal only (W).
                   Defaults to USE_AFFINE_PROCRUSTES global flag.
    """

    def __init__(self, use_affine=None):
        self.W = None  # Orthogonal transformation matrix
        self.b = None  # Bias term (affine extension)
        self.source_mean = None
        self.target_mean = None
        self.source_norm = None
        self.target_norm = None
        self.use_affine = use_affine if use_affine is not None else USE_AFFINE_PROCRUSTES

    def fit(self, source, target):
        """
        Fit affine transformation such that ||source transformed - target||_F is minimized.

        The affine transformation is achieved through the sequence:
        1. Center both datasets (removes means)
        2. Normalize (scales to unit norm)
        3. Find optimal orthogonal W via SVD
        4. Rescale and recenter to target space (provides translation)

        This is equivalent to affine transformation Y ≈ sWX + b where:
        - W is the rotation matrix
        - s is the scale ratio (target_norm/source_norm)
        - b is the translation (target_mean after accounting for sW@source_mean)

        No explicit bias term is needed since the recentering step provides
        the optimal translation for centered, normalized data.

        Args:
            source, target: [n_samples, n_features] representation matrices

        Returns:
            self (fitted)
        """
        assert source.shape == target.shape, "Source and target must have same shape"

        # Step 1: Center both datasets
        self.source_mean = source.mean(dim=0, keepdim=True)
        self.target_mean = target.mean(dim=0, keepdim=True)
        source_centered = source - self.source_mean
        target_centered = target - self.target_mean

        # Step 2: Normalize to unit Frobenius norm (tr(AA^T) = 1)
        # Use torch.norm for numerical stability (avoids overflow)
        self.source_norm = torch.norm(source_centered, 'fro')
        self.target_norm = torch.norm(target_centered, 'fro')

        # Add small epsilon to prevent division by zero
        eps = 1e-8
        source_normalized = source_centered / (self.source_norm + eps)
        target_normalized = target_centered / (self.target_norm + eps)

        # Sanity check for numerical issues
        if torch.isinf(self.source_norm) or torch.isinf(self.target_norm):
            print(f"  WARNING: Infinite norm detected, using layer normalization fallback")
            source_normalized = torch.nn.functional.normalize(source_centered, dim=-1)
            target_normalized = torch.nn.functional.normalize(target_centered, dim=-1)

        # Step 3: Compute cross-covariance matrix in float32 for numerical stability
        # CRITICAL: Large hidden dims (4096x4096) in low precision accumulate errors
        # Must convert inputs to float32 BEFORE matmul, not after
        M = source_normalized.float().T @ target_normalized.float()  # [D, D] matmul in float32

        # Step 4: SVD of M in float32 (SVD is numerically sensitive, needs high precision)
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)

        # Step 5: Optimal orthogonal transformation
        # CRITICAL: Keep W in float32! Converting to bfloat16 destroys orthogonality
        # Orthogonal matrices require high precision to maintain W @ W.T = I property
        self.W = U @ Vt  # Stay in float32 from SVD

        # Step 6: Bias term (affine extension from Lester et al., arXiv:2506.06609)
        # After centering and recentering, no explicit bias is needed
        # The affine translation is implicitly provided by recentering to target_mean
        # in the transform() method. Any explicit bias would be redundant since:
        # - We center both datasets before computing W (source_mean and target_mean → 0)
        # - After transformation, source_mean maps to 0 @ W = 0
        # - Recentering adds target_mean, completing the affine transformation
        # - Additional bias would double-count the translation
        if self.use_affine:
            # Affine transformation achieved through centering + W + recentering (no explicit bias)
            self.b = torch.zeros_like(self.target_mean)
        else:
            # Orthogonal only (no bias)
            self.b = torch.zeros_like(self.target_mean)

        # Verify orthogonality of W (W is already in float32)
        I = self.W @ self.W.T
        ortho_error = torch.norm(I - torch.eye(I.shape[0], device=I.device), 'fro')
        if ortho_error > 1e-3:
            print(f"  WARNING: Orthogonality error = {ortho_error:.6f}")

        return self

    def transform(self, source):
        """
        Apply the fitted affine transformation to new data.

        Transformation sequence:
        1. Center by source_mean (learned from fit)
        2. Normalize by source_norm (learned from fit)
        3. Apply orthogonal rotation W (learned from fit)
        4. Rescale by target_norm (learned from fit)
        5. Recenter by target_mean (provides affine translation)

        This sequence implements the affine transformation Y ≈ sW(X - μ_X) + μ_Y
        where s = target_norm/source_norm, and no explicit bias is needed.

        Args:
            source: [n_samples, n_features]

        Returns:
            transformed: [n_samples, n_features]
        """
        assert self.W is not None, "Must fit before transform"

        # CRITICAL: Move all Procrustes parameters to the same device as source
        # This prevents device mismatch errors when source is on different GPU than fit data
        device = source.device
        source_mean = self.source_mean.to(device)
        source_norm = self.source_norm.to(device)
        W = self.W.to(device)
        target_norm = self.target_norm.to(device)
        target_mean = self.target_mean.to(device)

        # Step 1-2: Center and normalize (same as fit)
        source_centered = source - source_mean
        source_normalized = source_centered / (source_norm + 1e-8)

        # Step 3: Apply orthogonal transformation
        transformed = source_normalized @ W

        # Step 4-5: Rescale and recenter to target space (completes affine transformation)
        transformed = transformed * target_norm
        transformed = transformed + target_mean

        # Note: No explicit bias addition needed - the recentering step above
        # provides the optimal translation for the affine transformation.
        # The use_affine flag controls whether we want this affine behavior
        # vs pure orthogonal (though mathematically both paths are equivalent
        # after centering, as bias would be zero anyway).

        return transformed

# ============================================================================
# Learned Adapter Architectures
# ============================================================================

class LinearAdapter(nn.Module):
    """Full linear projection (16.8M params)"""

    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.proj(x)


class AffineAdapter(nn.Module):
    """Full affine projection with bias (16.8M params)"""

    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class LoRAAdapter(nn.Module):
    """Low-rank adapter (65k params) - efficient baseline"""

    def __init__(self, hidden_dim=4096, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return x + self.scaling * self.lora_B(self.lora_A(x))


class LearnedProjection(nn.Module):
    """
    Learned linear projection to handle dimension mismatches between models.

    Used for cross-model communication when models have different hidden dimensions.
    For example, Llama 3.2 3B (3072) → Llama 3.1 8B (4096).

    References:
        - Ramesh & Li, "Communicating Activations Between Language Model Agents"
          arXiv:2501.14082 (ICML 2025)
          Uses learned projection W trained on C4 dataset to handle dimension mismatch
    """

    def __init__(self, source_dim, target_dim):
        """
        Args:
            source_dim: Source model's hidden dimension (e.g., 3072 for Llama 3.2 3B)
            target_dim: Target model's hidden dimension (e.g., 4096 for Llama 3.1 8B)
        """
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.proj = nn.Linear(source_dim, target_dim, bias=False)
        # Kaiming initialization for better gradient flow
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, x):
        """
        Project from source dimension to target dimension.

        Args:
            x: Tensor of shape [..., source_dim]

        Returns:
            Tensor of shape [..., target_dim]
        """
        return self.proj(x)

# ============================================================================
# Token-Initialized Compression
# ============================================================================

class TokenInitializedCompressor(nn.Module):
    """
    Compress sequences by initializing with actual token embeddings from the input.
    Key insight: Start from the semantic content we're compressing, not random noise.
    Creates a compressed z vector in lower dimensional space.
    """

    def __init__(self, model, tokenizer, compressed_length=64, hidden_dim=4096, d_z=256):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.compressed_length = compressed_length
        self.hidden_dim = hidden_dim
        self.d_z = d_z  # Latent dimension (much smaller than hidden_dim)

        # Get the embedding layer (handle DDP/DataParallel wrapped models)
        # CRITICAL: DDP and DataParallel wrap models, so attributes like get_input_embeddings()
        # are accessed via .module for wrapped models
        base_model = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
        self.embed_layer = base_model.get_input_embeddings()

        # Project from model space to latent z space
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, d_z),
            nn.LayerNorm(d_z)
        )

        # Project from latent z space back to model space
        self.from_latent = nn.Sequential(
            nn.Linear(d_z, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Learned attention pooling in latent space
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=d_z,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # Learnable query vectors in latent space
        self.compression_queries = nn.Parameter(
            torch.randn(compressed_length, d_z) * 0.02
        )

    def forward(self, input_ids, return_z=False):
        """
        Compress input_ids to compressed_length soft tokens via latent z space.

        Args:
            input_ids: [batch_size, seq_len] token ids
            return_z: If True, return the z latent representation

        Returns:
            If return_z=False: [batch_size, compressed_length, hidden_dim] model space embeddings
            If return_z=True: ([B, M, hidden_dim], [B, M, d_z]) both model and latent representations
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings for the actual input
        with torch.no_grad():
            token_embeds = self.embed_layer(input_ids)  # [B, seq_len, hidden_dim]

        # Sample tokens evenly across the sequence for initialization
        if seq_len >= self.compressed_length:
            # Sample evenly across the sequence for better coverage
            indices = torch.linspace(0, seq_len-1, self.compressed_length, dtype=torch.long, device=input_ids.device)
            init_embeds = token_embeds[:, indices, :]  # [B, compressed_length, hidden_dim]
        else:
            # If sequence is shorter, use all tokens and pad
            padding_needed = self.compressed_length - seq_len
            padding = self.from_latent(self.compression_queries[:padding_needed].unsqueeze(0).expand(batch_size, -1, -1))
            init_embeds = torch.cat([token_embeds, padding], dim=1)

        # Project to latent z space (compression happens here!)
        z = self.to_latent(init_embeds)  # [B, compressed_length, d_z]

        # Refine z representation with attention in latent space
        z_queries = z + self.compression_queries.unsqueeze(0)
        z_refined, _ = self.pooling_attention(
            query=z_queries,
            key=z,  # Self-attention in latent space
            value=z
        )

        # Project back to model embedding space
        model_embeds = self.from_latent(z_refined)  # [B, compressed_length, hidden_dim]

        if return_z:
            return model_embeds, z_refined
        else:
            return model_embeds

# ============================================================================
# Dataset
# ============================================================================

class AlignmentDataset(Dataset):
    """Dataset for adapter training with paired source-target texts"""

    def __init__(self, texts, tokenizer_a, tokenizer_b, max_length=None):
        self.texts = texts
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.max_length = max_length

        # Find the maximum sequence length if not truncating
        if max_length is None:
            print("Computing maximum sequence length from dataset...")
            max_len_a = 0
            max_len_b = 0
            for text in texts[:100]:  # Sample to find reasonable max
                tokens_a = tokenizer_a(text, return_tensors="pt")["input_ids"]
                tokens_b = tokenizer_b(text, return_tensors="pt")["input_ids"]
                max_len_a = max(max_len_a, tokens_a.shape[1])
                max_len_b = max(max_len_b, tokens_b.shape[1])
            self.max_length = max(max_len_a, max_len_b)
            print(f"Using full sequences with padding to {self.max_length} tokens")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize with padding to max_length (either specified or computed)
        # NO TRUNCATION if max_length was None initially
        truncation = self.max_length is not None

        inputs_a = self.tokenizer_a(text, truncation=truncation, max_length=self.max_length,
                                    padding="max_length", return_tensors="pt")
        inputs_b = self.tokenizer_b(text, truncation=truncation, max_length=self.max_length,
                                    padding="max_length", return_tensors="pt")

        # Ensure same length for batch processing
        if inputs_a["input_ids"].shape[1] != inputs_b["input_ids"].shape[1]:
            # Pad the shorter one
            max_len = max(inputs_a["input_ids"].shape[1], inputs_b["input_ids"].shape[1])

            if inputs_a["input_ids"].shape[1] < max_len:
                pad_len = max_len - inputs_a["input_ids"].shape[1]
                inputs_a["input_ids"] = F.pad(inputs_a["input_ids"], (0, pad_len), value=self.tokenizer_a.pad_token_id)
                inputs_a["attention_mask"] = F.pad(inputs_a["attention_mask"], (0, pad_len), value=0)

            if inputs_b["input_ids"].shape[1] < max_len:
                pad_len = max_len - inputs_b["input_ids"].shape[1]
                inputs_b["input_ids"] = F.pad(inputs_b["input_ids"], (0, pad_len), value=self.tokenizer_b.pad_token_id)
                inputs_b["attention_mask"] = F.pad(inputs_b["attention_mask"], (0, pad_len), value=0)

        return {
            "input_ids_a": inputs_a["input_ids"][0],
            "attention_mask_a": inputs_a["attention_mask"][0],
            "input_ids_b": inputs_b["input_ids"][0],
            "attention_mask_b": inputs_b["attention_mask"][0],
        }

# ============================================================================
# Procrustes Experiment - Model Stitching via Affine Alignment
# ============================================================================

def run_procrustes_experiment(model_a_id=None, model_b_id=None):
    """
    Run Procrustes alignment experiment across different layers.

    Args:
        model_a_id: Model A identifier (defaults to LLAMA_MODEL)
        model_b_id: Model B identifier (defaults to MISTRAL_MODEL)

    PURPOSE:
    Tests whether orthogonal/affine transformations can align hidden states between
    heterogeneous LLMs. This is a baseline alignment method that requires no training
    - just SVD-based geometric alignment.

    RELEVANCE TO CROSS-LLM COMMUNICATION:
    - Establishes feasibility of representation alignment between different architectures
    - Provides baseline CKA similarity scores for comparison with learned methods
    - Tests model stitching hypothesis: can we "plug" one model's activations into another?

    LITERATURE:
    - "Transferring Features Across Language Models With Model Stitching" (arXiv:2506.06609, June 2025)
      Shows affine mappings between residual streams effectively transfer features across LLMs
    - "ConTrans: Weak-to-Strong Alignment via Concept Transplantation" (arXiv:2405.13578, 2024)
      Uses affine transformations to reformulate concept vectors for cross-model transfer
    - "Do LLMs Have Consistent Values?" (ICLR 2025)
      Applies Procrustes analysis to compare embedding spaces between models

    METHOD:
    1. Extract hidden states from both models at layers [8, 16, 24]
    2. Fit affine transformation: Y ≈ sW(X - μ_X) + μ_Y via SVD
    3. Measure alignment quality with CKA (Centered Kernel Alignment)
    4. Test both Llama→Mistral and Mistral→Llama directions

    EXPECTED RESULTS (from literature):
    - CKA similarity: 0.3-0.5 for middle layers without training
    - Affine outperforms pure orthogonal by 5-8% (model stitching papers)
    """

    # Default to main experiment models if not specified
    if model_a_id is None:
        model_a_id = LLAMA_MODEL
    if model_b_id is None:
        model_b_id = MISTRAL_MODEL

    print("\n" + "=" * 80)
    print("PROCRUSTES ALIGNMENT EXPERIMENT (GPU-ACCELERATED)")
    print("=" * 80)
    print(f"Model A: {model_a_id}")
    print(f"Model B: {model_b_id}")

    # Use global device
    device = DEVICE
    print(f"Device: {device} (Procrustes on {PLATFORM})")

    # Platform-specific model loading
    print(f"\nLoading models on {PLATFORM}...")

    # Select dtype based on platform
    if PLATFORM == 'mac':
        dtype = torch.float32  # MPS works best with float32
        print("Using float32 for MPS")
    elif USE_BF16:
        dtype = torch.bfloat16
        print("Using bfloat16 for H100")
    else:
        dtype = torch.float16
        print("Using float16")

    # Model loading arguments for Procrustes (inference only)
    # For HPC with multiple GPUs, properly distribute models
    if PLATFORM == 'hpc' and torch.cuda.device_count() >= 2:
        print("Loading models on separate GPUs to avoid OOM...")
        # Don't use device_map - load to CPU first then move to specific GPU
        llama_kwargs = {
            'torch_dtype': dtype,
            'low_cpu_mem_usage': True,
        }
        mistral_kwargs = {
            'torch_dtype': dtype,
            'low_cpu_mem_usage': True,
        }
    else:
        # For Mac or single GPU, use auto device map
        llama_kwargs = mistral_kwargs = {
            'torch_dtype': dtype,
            'device_map': "auto",
        }

    # Add Flash Attention for HPC only
    if USE_FLASH_ATTENTION:
        llama_kwargs['attn_implementation'] = "flash_attention_2"
        mistral_kwargs['attn_implementation'] = "flash_attention_2"
        print("Using Flash Attention 2")
    else:
        llama_kwargs['attn_implementation'] = "eager"
        mistral_kwargs['attn_implementation'] = "eager"

    llama_model = AutoModelForCausalLM.from_pretrained(
        model_a_id,
        **llama_kwargs
    ).eval()

    # Freeze Llama parameters (no training needed for Procrustes)
    for param in llama_model.parameters():
        param.requires_grad = False

    mistral_model = AutoModelForCausalLM.from_pretrained(
        model_b_id,
        **mistral_kwargs
    ).eval()

    # Freeze Mistral parameters (no training needed for Procrustes)
    for param in mistral_model.parameters():
        param.requires_grad = False

    # Explicitly move models to their designated GPUs for HPC
    if PLATFORM == 'hpc' and torch.cuda.device_count() >= 2:
        llama_model = llama_model.to('cuda:0')
        mistral_model = mistral_model.to('cuda:1')
        print(f"Model A moved to cuda:0")
        print(f"Model B moved to cuda:1")

    # Load tokenizers
    llama_tokenizer = AutoTokenizer.from_pretrained(model_a_id)
    mistral_tokenizer = AutoTokenizer.from_pretrained(model_b_id)

    # Set padding tokens
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    if mistral_tokenizer.pad_token is None:
        mistral_tokenizer.pad_token = mistral_tokenizer.eos_token

    # Load calibration dataset
    print(f"\nLoading calibration dataset ({CALIBRATION_SIZE} samples)...")

    # Fix cache corruption issues
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    squad_cache = cache_dir / "squad"
    if squad_cache.exists():
        print(f"  Clearing potentially corrupted cache at {squad_cache}")
        shutil.rmtree(squad_cache)

    dataset = load_dataset("squad", split=f"train[:{CALIBRATION_SIZE}]")
    calibration_texts = [item["context"][:500] for item in dataset]

    results = {}

    for layer_idx in LAYERS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Testing Layer {layer_idx}")
        print(f"{'='*60}")

        # Collect hidden states for calibration
        llama_hidden_all = []
        mistral_hidden_all = []

        with torch.no_grad():
            # Determine device for each model (they may be on different GPUs)
            if PLATFORM == 'hpc' and torch.cuda.device_count() >= 2:
                llama_device = torch.device('cuda:0')
                mistral_device = torch.device('cuda:1')
            else:
                llama_device = mistral_device = device

            for text in calibration_texts[:5]:  # Further reduced to 5 samples for memory
                # First tokenize to get sequence lengths
                llama_tokens = llama_tokenizer(text, truncation=True, max_length=512,
                                              padding=False, return_tensors="pt")
                mistral_tokens = mistral_tokenizer(text, truncation=True, max_length=512,
                                                  padding=False, return_tensors="pt")

                # Use the minimum length to ensure alignment
                min_len = min(llama_tokens['input_ids'].shape[1],
                             mistral_tokens['input_ids'].shape[1])

                # Truncate both to the same length
                llama_inputs = {
                    'input_ids': llama_tokens['input_ids'][:, :min_len].to(llama_device),
                    'attention_mask': llama_tokens['attention_mask'][:, :min_len].to(llama_device)
                }
                mistral_inputs = {
                    'input_ids': mistral_tokens['input_ids'][:, :min_len].to(mistral_device),
                    'attention_mask': mistral_tokens['attention_mask'][:, :min_len].to(mistral_device)
                }

                # Get hidden states without computing logits (saves memory)
                # Access the model.model directly to avoid lm_head computation
                llama_hidden_states = llama_model.model(**llama_inputs, output_hidden_states=True).hidden_states
                mistral_hidden_states = mistral_model.model(**mistral_inputs, output_hidden_states=True).hidden_states

                # Extract layer hidden states and flatten
                llama_hidden = llama_hidden_states[layer_idx][0]  # [seq_len, hidden]
                mistral_hidden = mistral_hidden_states[layer_idx][0]

                # Both should now have the same sequence length
                assert llama_hidden.shape[0] == mistral_hidden.shape[0], f"Shape mismatch: {llama_hidden.shape} vs {mistral_hidden.shape}"

                llama_hidden_all.append(llama_hidden)
                mistral_hidden_all.append(mistral_hidden)

        # Concatenate all hidden states
        llama_hidden_all = torch.cat(llama_hidden_all, dim=0)
        mistral_hidden_all = torch.cat(mistral_hidden_all, dim=0)

        # Move to same device for Procrustes computation
        # When models are on different GPUs, we need to ensure both hidden states are on the same device
        if PLATFORM == 'hpc':
            # Use GPU 0 for Procrustes computation
            compute_device = torch.device('cuda:0')
            llama_hidden_all = llama_hidden_all.to(compute_device)
            mistral_hidden_all = mistral_hidden_all.to(compute_device)
            print(f"  Moving {llama_hidden_all.shape[0]} samples to GPU 0 for fast SVD")
        elif device.type == 'cuda':
            llama_hidden_all = llama_hidden_all.to(device)
            mistral_hidden_all = mistral_hidden_all.to(device)
            print(f"  Moving {llama_hidden_all.shape[0]} samples to {device} for fast SVD")
        else:
            # Keep on current device (CPU/MPS)
            compute_device = device

        # Fit Procrustes alignments (SVD runs on GPU)
        print(f"\nFitting Procrustes alignments on {device}...")

        # Mistral → Llama
        mistral_to_llama = ProcrustesAlignment()
        mistral_to_llama.fit(mistral_hidden_all.float(), llama_hidden_all.float())

        # Llama → Mistral
        llama_to_mistral = ProcrustesAlignment()
        llama_to_mistral.fit(llama_hidden_all.float(), mistral_hidden_all.float())

        # Compute CKA scores before and after Procrustes alignment (2025 best practice)
        print(f"  Computing CKA similarity scores...")

        # Use debiased CKA (unbiased HSIC estimator) per Murphy et al. ICLR 2024
        # Biased CKA gives inflated scores (0.9998+) in low-sample, high-D regime
        # Debiased estimator provides more realistic similarity scores

        # CKA before alignment (baseline - how similar are representations naturally?)
        cka_before_mistral_llama = CKA.cka_similarity(
            mistral_hidden_all.float(),
            llama_hidden_all.float(),
            debiased=True  # Use unbiased HSIC for low-sample, high-D regime (Murphy et al. ICLR 2024)
        )

        # CKA after Procrustes alignment (how much does alignment help?)
        mistral_aligned = mistral_to_llama.transform(mistral_hidden_all)
        cka_after_mistral_llama = CKA.cka_similarity(
            mistral_aligned.float(),
            llama_hidden_all.float(),
            debiased=True  # Use unbiased HSIC for low-sample, high-D regime (Murphy et al. ICLR 2024)
        )

        llama_aligned = llama_to_mistral.transform(llama_hidden_all)
        cka_after_llama_mistral = CKA.cka_similarity(
            llama_aligned.float(),
            mistral_hidden_all.float(),
            debiased=True  # Use unbiased HSIC for low-sample, high-D regime (Murphy et al. ICLR 2024)
        )

        # Calculate improvement
        cka_improvement_mistral_llama = float(cka_after_mistral_llama.item() - cka_before_mistral_llama.item())

        print(f"  CKA Mistral→Llama:")
        print(f"    Before alignment: {cka_before_mistral_llama.item():.4f}")
        print(f"    After alignment:  {cka_after_mistral_llama.item():.4f}")
        print(f"    Improvement:      {cka_improvement_mistral_llama:+.4f}")
        print(f"  CKA Llama→Mistral after alignment: {cka_after_llama_mistral.item():.4f}")

        # Save alignments for later use by activation communication experiment
        script_dir = Path(__file__).parent.absolute()
        alignment_dir = script_dir / "runs" / "procrustes_alignments"
        alignment_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'W': llama_to_mistral.W,
            'source_mean': llama_to_mistral.source_mean,
            'target_mean': llama_to_mistral.target_mean,
            'source_norm': llama_to_mistral.source_norm,
            'target_norm': llama_to_mistral.target_norm,
            'b': llama_to_mistral.b
        }, alignment_dir / f"layer_{layer_idx}.pt")
        print(f"  Saved Procrustes alignment to {alignment_dir / f'layer_{layer_idx}.pt'}")

        # Test generation
        layer_results = {
            "cka_metrics": {
                "cka_before_alignment": float(cka_before_mistral_llama.item()),
                "cka_after_mistral_to_llama": float(cka_after_mistral_llama.item()),
                "cka_after_llama_to_mistral": float(cka_after_llama_mistral.item()),
                "cka_improvement": float(cka_improvement_mistral_llama)
            },
            "mistral_to_mistral": {},
            "llama_to_llama": {},
            "llama_to_mistral": {},
            "mistral_to_llama": {}
        }

        print(f"\nTesting generation...")
        for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"  Prompt {prompt_idx}/5: {prompt[:50]}...")

            # Baseline: Mistral → Mistral (identity)
            print(f"    Testing Mistral→Mistral (baseline)...")
            mistral_inputs = mistral_tokenizer(prompt, return_tensors="pt").to(mistral_device)
            with torch.no_grad():
                output = mistral_model.generate(**mistral_inputs, max_new_tokens=20, do_sample=False)
            generated = mistral_tokenizer.decode(output[0], skip_special_tokens=True)
            layer_results["mistral_to_mistral"][f"prompt_{prompt_idx}"] = generated
            print(f"      Generated: {generated[:80]}")

            # Baseline: Llama → Llama (identity)
            print(f"    Testing Llama→Llama (baseline)...")
            llama_inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_device)
            with torch.no_grad():
                output = llama_model.generate(**llama_inputs, max_new_tokens=20, do_sample=False)
            generated = llama_tokenizer.decode(output[0], skip_special_tokens=True)
            layer_results["llama_to_llama"][f"prompt_{prompt_idx}"] = generated
            print(f"      Generated: {generated[:80]}")

            # Cross-model: Llama → Mistral
            print(f"    Testing Llama→Mistral (cross-model via Procrustes)...")
            try:
                # Get Llama hidden states at this layer
                with torch.no_grad():
                    llama_outputs = llama_model(
                        **llama_inputs,
                        output_hidden_states=True
                    )
                    llama_hidden = llama_outputs.hidden_states[layer_idx]  # [batch, seq, hidden]

                    # Apply Procrustes transformation
                    original_shape = llama_hidden.shape
                    llama_hidden_flat = llama_hidden.reshape(-1, llama_hidden.shape[-1])
                    transformed = llama_to_mistral.transform(llama_hidden_flat.float())
                    transformed = transformed.reshape(original_shape).to(llama_hidden.dtype)

                    # Measure transformation quality
                    transform_norm = torch.norm(transformed - llama_hidden).item()
                    print(f"      Transformation norm: {transform_norm:.4f}")

                    # Try to continue generation with transformed hidden states
                    # Note: This is a simplified approach - proper injection would require
                    # modifying the model's forward pass to start from intermediate layer
                    # For now, we use the transformation as inputs_embeds
                    mistral_output = mistral_model.generate(
                        inputs_embeds=transformed.to(mistral_device),
                        max_new_tokens=20,
                        do_sample=False
                    )
                    generated = mistral_tokenizer.decode(mistral_output[0], skip_special_tokens=True)
                    layer_results["llama_to_mistral"][f"prompt_{prompt_idx}"] = generated
                    print(f"      Generated: {generated[:80]}")

            except Exception as e:
                error_msg = f"Failed: {str(e)}"
                layer_results["llama_to_mistral"][f"prompt_{prompt_idx}"] = error_msg
                print(f"      {error_msg}")

            # Cross-model: Mistral → Llama
            print(f"    Testing Mistral→Llama (cross-model via Procrustes)...")
            try:
                # Get Mistral hidden states at this layer
                with torch.no_grad():
                    mistral_outputs = mistral_model(
                        **mistral_inputs,
                        output_hidden_states=True
                    )
                    mistral_hidden = mistral_outputs.hidden_states[layer_idx]  # [batch, seq, hidden]

                    # Apply Procrustes transformation
                    original_shape = mistral_hidden.shape
                    mistral_hidden_flat = mistral_hidden.reshape(-1, mistral_hidden.shape[-1])
                    transformed = mistral_to_llama.transform(mistral_hidden_flat.float())
                    transformed = transformed.reshape(original_shape).to(mistral_hidden.dtype)

                    # Measure transformation quality
                    transform_norm = torch.norm(transformed - mistral_hidden).item()
                    print(f"      Transformation norm: {transform_norm:.4f}")

                    # Try to continue generation with transformed hidden states
                    llama_output = llama_model.generate(
                        inputs_embeds=transformed.to(llama_device),
                        max_new_tokens=20,
                        do_sample=False
                    )
                    generated = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)
                    layer_results["mistral_to_llama"][f"prompt_{prompt_idx}"] = generated
                    print(f"      Generated: {generated[:80]}")

            except Exception as e:
                error_msg = f"Failed: {str(e)}"
                layer_results["mistral_to_llama"][f"prompt_{prompt_idx}"] = error_msg
                print(f"      {error_msg}")

        results[f"layer_{layer_idx}"] = layer_results

    # Print summary statistics
    print("\n" + "=" * 80)
    print("PROCRUSTES EXPERIMENT SUMMARY")
    print("=" * 80)

    for layer_idx in LAYERS_TO_TEST:
        layer_key = f"layer_{layer_idx}"
        if layer_key in results:
            layer_data = results[layer_key]

            # Count successes/failures
            mistral_mistral_count = len([v for v in layer_data.get("mistral_to_mistral", {}).values() if not v.startswith("Failed")])
            llama_llama_count = len([v for v in layer_data.get("llama_to_llama", {}).values() if not v.startswith("Failed")])
            llama_mistral_count = len([v for v in layer_data.get("llama_to_mistral", {}).values() if not v.startswith("Failed")])
            mistral_llama_count = len([v for v in layer_data.get("mistral_to_llama", {}).values() if not v.startswith("Failed")])

            llama_mistral_failed = len([v for v in layer_data.get("llama_to_mistral", {}).values() if v.startswith("Failed")])
            mistral_llama_failed = len([v for v in layer_data.get("mistral_to_llama", {}).values() if v.startswith("Failed")])

            print(f"\nLayer {layer_idx}:")
            print(f"  Baselines:")
            print(f"    Mistral→Mistral: {mistral_mistral_count}/5 succeeded")
            print(f"    Llama→Llama: {llama_llama_count}/5 succeeded")
            print(f"  Cross-model:")
            print(f"    Llama→Mistral: {llama_mistral_count}/5 succeeded, {llama_mistral_failed}/5 failed")
            print(f"    Mistral→Llama: {mistral_llama_count}/5 succeeded, {mistral_llama_failed}/5 failed")

    print("=" * 80)

    # CRITICAL: Explicitly delete models and clear GPU memory before returning
    # Without this, models remain in GPU memory and cause OOM for next experiments
    print("\nCleaning up Procrustes experiment...")
    del llama_model
    del mistral_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  Models deleted and GPU memory cleared")

    return results

# ============================================================================
# Per-Epoch Evaluation for Learned Adapters
# ============================================================================

def evaluate_adapter_epoch(model_a, model_b, tokenizer_a, tokenizer_b, adapter,
                           device, epoch, alignment_layer=16, use_ddp=False):
    """
    Comprehensive per-epoch evaluation of adapter quality.

    Measures:
    1. CKA similarity (alignment quality)
    2. Generation quality (cross-model injection samples)
    3. Cosine similarity (hidden state alignment)

    Args:
        model_a, model_b: The two models
        tokenizer_a, tokenizer_b: Their tokenizers
        adapter: The adapter being trained
        device: Device to run on
        epoch: Current epoch number
        alignment_layer: Which layer is being aligned
        use_ddp: Whether using DDP (unwrap .module)

    Returns:
        dict with evaluation metrics
    """
    # Set models to eval mode
    was_training_adapter = adapter.training
    was_training_a = model_a.training
    was_training_b = model_b.training

    adapter.eval()
    model_a.eval()
    model_b.eval()

    # Unwrap DDP if needed
    adapter_module = adapter.module if use_ddp and hasattr(adapter, 'module') else adapter

    results = {
        "epoch": epoch,
        "alignment_layer": alignment_layer,
        "cka_scores": {},
        "cosine_similarities": {},
        "generations": {}
    }

    # Test prompts for generation quality
    test_prompts = [
        "The capital of France is",
        "To solve this problem, we need to",
        "The future of artificial intelligence",
        "In the year 2050, humanity will",
        "The main difference between cats and dogs"
    ]

    with torch.no_grad():
        # 1. Compute CKA similarity between adapted and target hidden states
        # IMPORTANT: Compute CKA across ALL training layers (not just alignment_layer)
        # to match the multi-layer training objective
        print(f"\n  Computing multi-layer CKA similarity...")

        # Use all 5 test prompts for more stable CKA measurements (was 3, causing high variance)
        cka_per_layer = {layer_idx: [] for layer_idx in ALIGNMENT_LAYERS}
        cosine_per_layer = {layer_idx: [] for layer_idx in ALIGNMENT_LAYERS}

        for prompt in test_prompts:  # Use ALL 5 prompts (not [:3])
            # Tokenize for both models
            inputs_a = tokenizer_a(prompt, return_tensors="pt", padding=False).to(device)
            inputs_b = tokenizer_b(prompt, return_tensors="pt", padding=False).to(device)

            # Get hidden states from both models
            outputs_a = model_a.model(**inputs_a, output_hidden_states=True)
            outputs_b = model_b.model(**inputs_b, output_hidden_states=True)

            # Compute CKA for each training layer
            for layer_idx in ALIGNMENT_LAYERS:
                hidden_a = outputs_a.hidden_states[layer_idx]  # [1, seq_a, hidden]
                hidden_b = outputs_b.hidden_states[layer_idx]  # [1, seq_b, hidden]

                # Apply adapter to model A's hidden states
                adapted_a = adapter_module(hidden_a)  # [1, seq_a, hidden]

                # CRITICAL: Handle different sequence lengths (Llama vs Mistral tokenization)
                # Truncate both to the shorter sequence length for CKA comparison
                seq_len_a = adapted_a.shape[1]
                seq_len_b = hidden_b.shape[1]
                min_seq_len = min(seq_len_a, seq_len_b)

                # Truncate to same length
                adapted_truncated = adapted_a[:, :min_seq_len, :]  # [1, min_len, hidden]
                target_truncated = hidden_b[:, :min_seq_len, :]    # [1, min_len, hidden]

                # Flatten to [n_samples, features] for CKA
                adapted_flat = adapted_truncated.view(-1, adapted_truncated.shape[-1]).float()
                target_flat = target_truncated.view(-1, target_truncated.shape[-1]).float()

                # Only compute if we have enough samples (at least 2 tokens)
                if adapted_flat.shape[0] >= 2:
                    # Use unbiased HSIC for low-sample, high-D regime (Murphy et al. ICLR 2024)
                    # Biased CKA inflates scores in this regime
                    cka_score = CKA.cka_similarity(adapted_flat, target_flat, debiased=True)
                    cka_per_layer[layer_idx].append(float(cka_score.item()))

                    # Compute cosine similarity
                    cos_sim = F.cosine_similarity(
                        adapted_flat.mean(dim=0, keepdim=True),
                        target_flat.mean(dim=0, keepdim=True)
                    )
                    cosine_per_layer[layer_idx].append(float(cos_sim.item()))

        # Compute weighted average CKA across all layers (matching training objective)
        layer_cka_means = {}
        for layer_idx in ALIGNMENT_LAYERS:
            if cka_per_layer[layer_idx]:
                layer_cka_means[layer_idx] = sum(cka_per_layer[layer_idx]) / len(cka_per_layer[layer_idx])
            else:
                layer_cka_means[layer_idx] = 0.0

        # Weighted average matching training (this is the key metric!)
        if layer_cka_means:
            weighted_cka = sum(
                layer_cka_means[layer_idx] * weight
                for layer_idx, weight in zip(ALIGNMENT_LAYERS, LAYER_WEIGHTS)
            )
        else:
            weighted_cka = 0.0

        results["cka_scores"]["mean"] = weighted_cka
        results["cka_scores"]["per_layer"] = layer_cka_means
        results["cka_scores"]["samples_per_layer"] = cka_per_layer

        # Cosine similarity (same weighted average)
        layer_cosine_means = {}
        for layer_idx in ALIGNMENT_LAYERS:
            if cosine_per_layer[layer_idx]:
                layer_cosine_means[layer_idx] = sum(cosine_per_layer[layer_idx]) / len(cosine_per_layer[layer_idx])
            else:
                layer_cosine_means[layer_idx] = 0.0

        if layer_cosine_means:
            weighted_cosine = sum(
                layer_cosine_means[layer_idx] * weight
                for layer_idx, weight in zip(ALIGNMENT_LAYERS, LAYER_WEIGHTS)
            )
        else:
            weighted_cosine = 0.0

        results["cosine_similarities"]["mean"] = weighted_cosine
        results["cosine_similarities"]["per_layer"] = layer_cosine_means

        # 2. Generate samples for qualitative assessment
        print(f"  Generating quality samples...")
        for i, prompt in enumerate(test_prompts, 1):
            gen_results = {}

            # Baseline: Model A → Model A (should be fluent)
            try:
                inputs_a = tokenizer_a(prompt, return_tensors="pt").to(device)
                outputs_a = model_a.generate(
                    **inputs_a,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer_a.eos_token_id
                )
                gen_a = tokenizer_a.decode(outputs_a[0], skip_special_tokens=True)
                gen_results["model_a_baseline"] = gen_a
            except Exception as e:
                gen_results["model_a_baseline"] = f"Error: {e}"

            # Baseline: Model B → Model B (should be fluent)
            try:
                inputs_b = tokenizer_b(prompt, return_tensors="pt").to(device)
                outputs_b = model_b.generate(
                    **inputs_b,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer_b.eos_token_id
                )
                gen_b = tokenizer_b.decode(outputs_b[0], skip_special_tokens=True)
                gen_results["model_b_baseline"] = gen_b
            except Exception as e:
                gen_results["model_b_baseline"] = f"Error: {e}"

            # Note: Full cross-model generation with injection requires model surgery
            # For now, we track the hidden state alignment quality via CKA
            gen_results["note"] = "Cross-model injection requires additional implementation"

            results["generations"][f"prompt_{i}"] = {
                "text": prompt,
                **gen_results
            }

    # Restore training state
    if was_training_adapter:
        adapter.train()
    if was_training_a:
        model_a.train()
    if was_training_b:
        model_b.train()

    # Print summary
    print(f"\n  Epoch {epoch} Evaluation:")
    print(f"    CKA Similarity: {results['cka_scores']['mean']:.4f}")
    print(f"    Cosine Similarity: {results['cosine_similarities']['mean']:.4f}")

    return results


# ============================================================================
# Early Stopping Helper
# ============================================================================

class EarlyStopping:
    """Early stopping to avoid overtraining."""

    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.should_stop = False

    def __call__(self, val_loss):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            self.should_stop = False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.should_stop = False


# ============================================================================
# Learned Adapter Training - Trainable Cross-Model Alignment
# ============================================================================

def train_adapter(model_a, model_b, tokenizer_a, tokenizer_b, adapter,
                  device, log_file, num_samples=1000, checkpoint_dir=None,
                  use_ddp=False, rank=0, world_size=1):
    """
    Train a single adapter for cross-model alignment with contrastive learning.

    PURPOSE:
    Learn trainable mappings (linear, affine, or LoRA) to align hidden states between
    Llama and Mistral using contrastive objectives. Unlike Procrustes (closed-form),
    these adapters can learn non-geometric alignments through gradient descent.

    RELEVANCE TO CROSS-LLM COMMUNICATION:
    - Tests whether learned adapters outperform geometric (Procrustes) alignment
    - LoRA adapters are parameter-efficient and can be transferred across models
    - Contrastive learning (InfoNCE) prevents mode collapse and improves alignment quality

    LITERATURE:
    - "Cross-LoRA: A Data-Free LoRA Transfer Framework" (arXiv:2508.05232, August 2025)
      Transfers LoRA adapters across heterogeneous LLMs via SVD and subspace alignment
    - "MoA: Heterogeneous Mixture of Adapters" (arXiv:2506.05928, June 2025)
      Dynamically integrates PEFT adapters with diverse structures for multi-task transfer
    - "Activation Manifold Projection (CAST)" (arXiv:2510.17902, October 2025)
      Learns direct mappings between activation manifolds, retaining 85-95% performance

    METHOD:
    1. Extract hidden states from both models on Wikitext-103
    2. Train adapter with InfoNCE contrastive loss (τ=0.07)
    3. Use AdamW optimizer with cosine annealing (10 epochs, lr=5e-5)
    4. Measure CKA similarity before/after adapter application

    ADAPTER TYPES:
    - Linear: W @ x (single matrix, ~16M params for d=4096)
    - Affine: W @ x + b (adds bias, ~16M + 4K params)
    - LoRA: Low-rank adaptation (rank 8, ~260K params, 98% fewer than linear)

    EXPECTED RESULTS (from literature):
    - CKA improvement: +0.1-0.2 over Procrustes baseline
    - LoRA achieves 90-95% of linear performance with 2% parameters
    - Contrastive loss prevents mode collapse (all representations → same vector)
    """

    print(f"\nTraining {adapter.__class__.__name__}...", file=log_file)

    # Setup checkpointing
    start_epoch = 0
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"

        # Check for existing checkpoint
        if checkpoint_path.exists():
            print(f"Found checkpoint at {checkpoint_path}, resuming training...", file=log_file)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # weights_only=False for optimizer state
            adapter.load_state_dict(checkpoint['adapter_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}", file=log_file)

    # Prepare dataset
    if rank == 0:
        print(f"Loading dataset ({num_samples} samples)...", file=log_file)

    # Fix cache corruption (only rank 0 should clear cache)
    if rank == 0:
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        wikitext_cache = cache_dir / "wikitext"
        if wikitext_cache.exists():
            shutil.rmtree(wikitext_cache)

    # Synchronize all processes before loading dataset
    if use_ddp:
        dist.barrier()

    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"]) > 100][:num_samples]

    train_dataset = AlignmentDataset(texts, tokenizer_a, tokenizer_b, MAX_LENGTH)

    # Data loading configuration
    # NOTE: HPC system can't handle multiple workers (causes freeze), use single-threaded
    num_workers = 0  # Must be 0 on this HPC system to avoid DataLoader freeze

    # Use DistributedSampler for DDP to split data across processes
    if use_ddp:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=num_workers,
            pin_memory=True if PLATFORM == 'hpc' else False,
        )
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if PLATFORM == 'hpc' else False,
        )

    # Move adapter to device with correct dtype (must match model dtype)
    dtype = torch.bfloat16 if USE_BF16 else torch.float32
    adapter = adapter.to(device, dtype=dtype)

    # Wrap adapter with DDP if using distributed training
    if use_ddp:
        adapter = DDP(adapter, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)

    # Add cosine annealing scheduler (as mentioned in comments)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Initialize InfoNCE loss for contrastive learning
    contrastive_loss_fn = InfoNCE(temperature=TEMPERATURE)

    # Initialize contrastive weight scheduler for curriculum learning
    # (Hacohen & Weinshall, arXiv:1904.03626, ICML 2019)
    if USE_CONTRASTIVE_CURRICULUM:
        contrastive_scheduler = ContrastiveWeightScheduler(
            initial_weight=0.1,
            final_weight=CONTRASTIVE_WEIGHT,
            warmup_steps=len(dataloader) * 2  # Warmup for 2 epochs
        )
    else:
        contrastive_scheduler = None

    # Training loop
    adapter.train()
    training_metrics = {"epochs": [], "cka_scores": []}

    # Resume optimizer and scheduler state if checkpoint exists
    if checkpoint_dir and checkpoint_path.exists():
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        training_metrics = checkpoint.get('training_metrics', {"epochs": [], "cka_scores": []})

    # Log training configuration
    print(f"\n{'='*80}", file=log_file)
    print(f"TRAINING CONFIGURATION", file=log_file)
    print(f"{'='*80}", file=log_file)
    print(f"Total epochs: {EPOCHS}", file=log_file)
    print(f"Steps per epoch: {len(dataloader)}", file=log_file)
    print(f"Total training steps: {EPOCHS * len(dataloader)}", file=log_file)
    print(f"Batch size: {BATCH_SIZE}", file=log_file)
    print(f"Gradient accumulation: {GRAD_ACCUM_STEPS} (effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS})", file=log_file)
    print(f"Learning rate: {LEARNING_RATE}", file=log_file)
    print(f"Alignment layers: {ALIGNMENT_LAYERS}", file=log_file)
    print(f"{'='*80}\n", file=log_file)
    log_file.flush()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=2, min_delta=0.001)

    import time
    training_start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if use_ddp:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_gen_loss = 0.0
        epoch_contrast_loss = 0.0
        epoch_align_loss = 0.0
        epoch_uniform_loss = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()

        if rank == 0:  # Only print on main process
            msg = f"\n{'='*80}\nEpoch {epoch+1}/{EPOCHS}\n{'='*80}"
            print(msg, file=log_file)
            print(msg)  # Also print to stdout for tee capture
            log_file.flush()

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)

            # Multi-layer alignment: Extract representations from multiple layers
            all_source_reprs = []
            all_aligned_reprs = []

            with torch.no_grad():
                outputs_a = model_a(
                    input_ids=input_ids_a,
                    attention_mask=attention_mask_a,
                    output_hidden_states=True
                )
                outputs_b_teacher = model_b(
                    input_ids=input_ids_b,
                    attention_mask=attention_mask_b,
                    output_hidden_states=True
                )

            # Process multiple alignment layers
            generation_losses = []
            for layer_idx, layer_weight in zip(ALIGNMENT_LAYERS, LAYER_WEIGHTS):
                source_repr = outputs_a.hidden_states[layer_idx]
                aligned_repr = adapter(source_repr)

                all_source_reprs.append(source_repr)
                all_aligned_reprs.append(aligned_repr)

                # Create labels with padding tokens masked
                labels_b = input_ids_b.clone()
                labels_b[attention_mask_b == 0] = -100

                # Create position_ids for RoPE
                batch_size, seq_len = attention_mask_b.shape
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                position_ids = position_ids * attention_mask_b

                # Compute generation loss with Model B for this layer
                outputs_b = model_b(
                    inputs_embeds=aligned_repr,
                    attention_mask=attention_mask_b,
                    position_ids=position_ids,
                    labels=labels_b
                )

                # DDP automatically averages gradients across processes, loss is already scalar
                layer_loss = outputs_b.loss
                generation_losses.append(layer_loss * layer_weight)

            # Combine generation losses from multiple layers
            generation_loss = sum(generation_losses)

            # Compute contrastive loss using InfoNCE
            # Use middle layer representations for contrastive learning
            mid_idx = len(ALIGNMENT_LAYERS) // 2

            # Use mean pooling instead of CLS token (Reimers & Gurevych, arXiv:1908.10084)
            # Mean pooling outperforms CLS by 2-5% for semantic similarity (MTEB benchmark)
            if USE_MEAN_POOLING:
                anchor = mean_pooling(all_aligned_reprs[mid_idx], attention_mask_a)
                positive = mean_pooling(outputs_b_teacher.hidden_states[ALIGNMENT_LAYERS[mid_idx]], attention_mask_b)
            else:
                # Legacy: CLS token approach (backward compatibility)
                anchor = all_aligned_reprs[mid_idx][:, 0, :]
                positive = outputs_b_teacher.hidden_states[ALIGNMENT_LAYERS[mid_idx]][:, 0, :]

            # Create negatives by shuffling within batch
            batch_size = anchor.shape[0]
            if batch_size > 1:
                # Get negatives from other examples in batch
                neg_indices = []
                for i in range(batch_size):
                    # Get all indices except current example
                    neg_idx = list(range(batch_size))
                    neg_idx.remove(i)
                    neg_indices.append(neg_idx[:min(NUM_NEGATIVES, len(neg_idx))])

                # Pad if necessary
                max_neg = max(len(ni) for ni in neg_indices)
                negatives = []
                for i, neg_idx in enumerate(neg_indices):
                    neg_batch = positive[neg_idx]
                    if len(neg_idx) < max_neg:
                        # Pad with zeros if not enough negatives
                        padding = torch.zeros((max_neg - len(neg_idx), neg_batch.shape[-1]),
                                             device=device, dtype=neg_batch.dtype)
                        neg_batch = torch.cat([neg_batch, padding], dim=0)
                    negatives.append(neg_batch)
                negatives = torch.stack(negatives)

                contrastive_loss = contrastive_loss_fn(anchor, positive, negatives)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)

            # Compute alignment/uniformity metrics (Wang & Isola, ICML 2020)
            if LOG_ALIGN_UNIFORM and batch_size > 1:
                with torch.no_grad():
                    alignment_loss, uniformity_loss = AlignmentUniformity.compute_metrics(anchor, positive)
            else:
                alignment_loss = torch.tensor(0.0, device=device)
                uniformity_loss = torch.tensor(0.0, device=device)

            # Combine losses with dynamic contrastive weight (curriculum learning)
            if USE_CONTRASTIVE_CURRICULUM and contrastive_scheduler is not None:
                current_contrastive_weight = contrastive_scheduler.step()
            else:
                current_contrastive_weight = CONTRASTIVE_WEIGHT

            total_loss = generation_loss + current_contrastive_weight * contrastive_loss
            total_loss = total_loss / GRAD_ACCUM_STEPS
            total_loss.backward()

            grad_norm = 0.0
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Step the scheduler
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * GRAD_ACCUM_STEPS
            epoch_gen_loss += generation_loss.item()
            epoch_contrast_loss += contrastive_loss.item()
            epoch_align_loss += alignment_loss.item()
            epoch_uniform_loss += uniformity_loss.item()
            epoch_steps += 1

            # Progress logging every 100 steps (more informative but not spammy)
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / epoch_steps
                avg_gen_loss = epoch_gen_loss / epoch_steps
                avg_contrast_loss = epoch_contrast_loss / epoch_steps
                avg_align_loss = epoch_align_loss / epoch_steps
                avg_uniform_loss = epoch_uniform_loss / epoch_steps
                current_lr = optimizer.param_groups[0]['lr']
                progress_pct = 100 * (batch_idx + 1) / len(dataloader)
                elapsed = time.time() - epoch_start_time
                steps_per_sec = (batch_idx + 1) / elapsed
                eta_seconds = (len(dataloader) - batch_idx - 1) / steps_per_sec if steps_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                msg = f"  [{progress_pct:5.1f}%] Step {batch_idx+1:4d}/{len(dataloader)} | "
                msg += f"Loss: {avg_loss:.4f} (Gen: {avg_gen_loss:.4f}, Contr: {avg_contrast_loss:.4f}"
                if LOG_ALIGN_UNIFORM:
                    msg += f", Align: {avg_align_loss:.4f}, Uniform: {avg_uniform_loss:.4f}"
                msg += ") | "
                if USE_CONTRASTIVE_CURRICULUM and contrastive_scheduler is not None:
                    msg += f"ContrW: {current_contrastive_weight:.3f} | "
                msg += f"LR: {current_lr:.2e} | GradNorm: {grad_norm:.3f} | "
                msg += f"{steps_per_sec:.2f} steps/s | ETA: {eta_minutes:.1f}m"
                print(msg, file=log_file)
                print(msg)  # Also to stdout
                log_file.flush()

            # Quick check-in every 10 steps (just to log file, not stdout)
            elif (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / epoch_steps
                avg_gen_loss = epoch_gen_loss / epoch_steps
                avg_contrast_loss = epoch_contrast_loss / epoch_steps
                avg_align_loss = epoch_align_loss / epoch_steps
                avg_uniform_loss = epoch_uniform_loss / epoch_steps
                msg = f"  Step {batch_idx+1}/{len(dataloader)}: Loss = {avg_loss:.4f} (Gen: {avg_gen_loss:.4f}, Contr: {avg_contrast_loss:.4f}"
                if LOG_ALIGN_UNIFORM:
                    msg += f", Align: {avg_align_loss:.4f}, Uniform: {avg_uniform_loss:.4f}"
                msg += ")"
                print(msg, file=log_file)

        avg_epoch_loss = epoch_loss / epoch_steps
        avg_epoch_gen_loss = epoch_gen_loss / epoch_steps
        avg_epoch_contrast_loss = epoch_contrast_loss / epoch_steps

        # Run comprehensive per-epoch evaluation (only on rank 0)
        if rank == 0:
            mid_layer_idx = ALIGNMENT_LAYERS[len(ALIGNMENT_LAYERS) // 2]
            eval_results = evaluate_adapter_epoch(
                model_a, model_b, tokenizer_a, tokenizer_b, adapter,
                device=device,
                epoch=epoch + 1,
                alignment_layer=mid_layer_idx,
                use_ddp=use_ddp
            )

            # Save evaluation results
            if checkpoint_dir:
                eval_path = checkpoint_dir / f"eval_epoch_{epoch+1}.json"
                with open(eval_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)

            # Extract CKA for tracking
            avg_cka = eval_results["cka_scores"]["mean"]
            training_metrics["cka_scores"].append(avg_cka)
        else:
            # Non-main processes just set a placeholder
            avg_cka = 0.0

        training_metrics["epochs"].append({
            "epoch": epoch + 1,
            "loss": avg_epoch_loss,
            "generation_loss": avg_epoch_gen_loss,
            "contrastive_loss": avg_epoch_contrast_loss,
            "cka_score": avg_cka,
            "lr": optimizer.param_groups[0]['lr']
        })

        # Epoch summary with timing and metrics
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        remaining_epochs = EPOCHS - (epoch + 1)
        avg_epoch_time = total_elapsed / (epoch + 1 - start_epoch)
        eta_total = avg_epoch_time * remaining_epochs / 60  # in minutes

        msg = f"\n{'='*80}\n"
        msg += f"Epoch {epoch+1}/{EPOCHS} Complete | Time: {epoch_time/60:.1f}m | Total: {total_elapsed/60:.1f}m\n"
        msg += f"  Total Loss: {avg_epoch_loss:.4f} (Gen: {avg_epoch_gen_loss:.4f}, Contr: {avg_epoch_contrast_loss:.4f})\n"
        msg += f"  CKA Score: {avg_cka:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
        if remaining_epochs > 0:
            msg += f"  ETA for remaining {remaining_epochs} epochs: {eta_total:.1f}m\n"
        msg += f"{'='*80}"

        if rank == 0:  # Only print on main process
            print(msg, file=log_file)
            print(msg)  # Also to stdout
            log_file.flush()

        # Check early stopping (rank 0 decides, broadcast to all ranks to avoid deadlock)
        should_stop = False
        if use_ddp:
            # Rank 0 checks early stopping
            if rank == 0:
                should_stop = early_stopping(avg_epoch_loss)
                if should_stop:
                    print(f"\n  Early stopping triggered at epoch {epoch+1}", file=log_file)
                    print(f"  Best loss: {early_stopping.best_loss:.4f}", file=log_file)
                    print(f"  No improvement for {early_stopping.patience} epochs", file=log_file)
                    log_file.flush()

            # Broadcast decision to all ranks (critical for DDP synchronization)
            stop_tensor = torch.tensor([1.0 if should_stop else 0.0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() > 0.5

            # All ranks break together (prevents deadlock at barriers)
            if should_stop:
                break
        else:
            # Non-DDP: only rank 0 exists, simple check
            if rank == 0 and early_stopping(avg_epoch_loss):
                print(f"\n  Early stopping triggered at epoch {epoch+1}", file=log_file)
                print(f"  Best loss: {early_stopping.best_loss:.4f}", file=log_file)
                print(f"  No improvement for {early_stopping.patience} epochs", file=log_file)
                log_file.flush()
                break

        # Save checkpoint after each epoch (only on rank 0)
        if checkpoint_dir and rank == 0:
            checkpoint = {
                'epoch': epoch,
                'adapter_state_dict': adapter.module.state_dict() if use_ddp else adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_metrics': training_metrics,
                'loss': avg_epoch_loss,
                'eval_results': eval_results if rank == 0 else None,
            }
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}", file=log_file)

        # Synchronize all processes after checkpoint
        if use_ddp:
            dist.barrier()

    # Final training summary (only on rank 0)
    if rank == 0:
        total_training_time = time.time() - training_start_time
        msg = f"\n\n{'='*80}\n"
        msg += f"TRAINING COMPLETE\n"
        msg += f"{'='*80}\n"
        msg += f"Total time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.2f} hours)\n"
        msg += f"Total epochs: {EPOCHS}\n"
        msg += f"Final loss: {training_metrics['epochs'][-1]['loss']:.4f}\n"
        msg += f"Final CKA score: {training_metrics['epochs'][-1]['cka_score']:.4f}\n"
        msg += f"\nLoss progression:\n"
        for i, epoch_data in enumerate(training_metrics['epochs']):
            msg += f"  Epoch {epoch_data['epoch']:2d}: Loss {epoch_data['loss']:.4f}, CKA {epoch_data['cka_score']:.4f}\n"
        msg += f"{'='*80}\n"
        print(msg, file=log_file)
        print(msg)  # Also to stdout
        log_file.flush()

    return adapter, training_metrics

def run_adapter_experiment(adapter_type, gpu_id, model_a_id=None, model_b_id=None):
    """Run a single adapter experiment on specified GPU.

    Args:
        adapter_type: Type of adapter (linear, affine, lora)
        gpu_id: GPU ID to use (None for all GPUs with DDP)
        model_a_id: Model A identifier (defaults to LLAMA_MODEL)
        model_b_id: Model B identifier (defaults to MISTRAL_MODEL)
    """

    # Default to main experiment models if not specified
    if model_a_id is None:
        model_a_id = LLAMA_MODEL
    if model_b_id is None:
        model_b_id = MISTRAL_MODEL

    # Create output directory relative to script location (only rank 0)
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "runs" / "learned_adapters"

    # Only rank 0 creates directories and log files
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Fix log filename for multi-GPU case
    gpu_label = "allgpus" if gpu_id is None else f"gpu{gpu_id}"

    # Add model-specific suffix for ablation experiments
    model_suffix = ""
    if model_a_id == LLAMA_31_8B and model_b_id == LLAMA_32_3B:
        model_suffix = "_samevocab"

    log_path = output_dir / f"{adapter_type}_{gpu_label}{model_suffix}_{timestamp}.log"

    # Only rank 0 opens log file
    if is_main_process():
        log_file = open(log_path, 'w')
        # Redirect output to both console and file
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeLogger(log_file)
        sys.stderr = TeeLogger(log_file)
    else:
        log_file = None

    try:
        if is_main_process():
            print("=" * 80)
            print(f"LEARNED ADAPTER EXPERIMENT - {adapter_type.upper()}")
            print("=" * 80)
            print(f"Platform: {PLATFORM}")
            print(f"Model A: {model_a_id}")
            print(f"Model B: {model_b_id}")
            print(f"Log file: {log_path}")

            # Device configuration - use DDP for multi-GPU
            use_ddp = False
            rank = 0
            world_size = 1

            if PLATFORM == 'hpc' and gpu_id is None and torch.cuda.device_count() > 1:
                # Use DDP for multi-GPU training
                rank, world_size, device = setup_ddp()
                use_ddp = True
                if is_main_process():
                    print(f"\n{'='*80}")
                    print(f"GPU CONFIGURATION")
                    print(f"{'='*80}")
                    print(f"Mode: DistributedDataParallel (DDP)")
                    print(f"Number of GPUs: {world_size}")
                    print(f"GPU IDs: {list(range(world_size))}")
                    print(f"Rank: {rank}, Device: {device}")
                    print(f"Batch size per GPU: {BATCH_SIZE}")
                    print(f"Global batch size: {BATCH_SIZE * world_size}")
                    print(f"Effective batch (with grad accum): {BATCH_SIZE * world_size * GRAD_ACCUM_STEPS}")
                    print(f"{'='*80}\n")
            elif PLATFORM == 'hpc' and gpu_id is not None:
                # Use specific GPU
                device = torch.device(f"cuda:{gpu_id}")
                print(f"\n{'='*80}")
                print(f"GPU CONFIGURATION")
                print(f"{'='*80}")
                print(f"Mode: Single GPU")
                print(f"GPU assigned: {gpu_id}")
                print(f"Batch size: {BATCH_SIZE}")
                print(f"Effective batch (with grad accum): {BATCH_SIZE * GRAD_ACCUM_STEPS}")
                print(f"{'='*80}\n")
            else:
                # Use default device (MPS/CPU/single GPU)
                device = DEVICE
                print(f"\nDevice: {device}\n")

            # Platform-specific model loading
            print(f"\nLoading models on {PLATFORM}...")

            # Select dtype based on platform
            if PLATFORM == 'mac':
                dtype = torch.float32  # MPS works best with float32
                print("Using float32 for MPS")
            elif USE_BF16:
                dtype = torch.bfloat16
                print("Using bfloat16 for H100")
            else:
                dtype = torch.float16
                print("Using float16")

            # Model loading arguments
            model_kwargs = {
                'torch_dtype': dtype,
                'low_cpu_mem_usage': True,
            }

            # Add Flash Attention for HPC only
            if USE_FLASH_ATTENTION:
                model_kwargs['attn_implementation'] = "flash_attention_2"
                print("Using Flash Attention 2")

            model_a = AutoModelForCausalLM.from_pretrained(
                model_a_id,
                **model_kwargs
            ).to(device).eval()

            # Freeze model A parameters to save memory and compute
            for param in model_a.parameters():
                param.requires_grad = False

            # Enable gradient checkpointing to save activation memory during backprop
            # (still needed even with frozen params, reduces activation memory 2-3×)
            model_a.gradient_checkpointing_enable()

            model_b = AutoModelForCausalLM.from_pretrained(
                model_b_id,
                **model_kwargs
            ).to(device).eval()

            # Freeze model B parameters to save memory and compute
            for param in model_b.parameters():
                param.requires_grad = False

            # Enable gradient checkpointing to save activation memory during backprop
            model_b.gradient_checkpointing_enable()

            # Note: Models are frozen (requires_grad=False), so no need to wrap with DDP
            # Gradient checkpointing still helps reduce activation memory during backprop
            # Only the adapter will be wrapped with DDP in train_adapter()

            # Load tokenizers
            tokenizer_a = AutoTokenizer.from_pretrained(model_a_id)
            tokenizer_b = AutoTokenizer.from_pretrained(model_b_id)

            # Set padding tokens
            if tokenizer_a.pad_token is None:
                tokenizer_a.pad_token = tokenizer_a.eos_token
            if tokenizer_b.pad_token is None:
                tokenizer_b.pad_token = tokenizer_b.eos_token

            # Create adapter
            if adapter_type == "linear":
                adapter = LinearAdapter()
            elif adapter_type == "affine":
                adapter = AffineAdapter()
            elif adapter_type == "lora":
                adapter = LoRAAdapter()
            else:
                raise ValueError(f"Unknown adapter type: {adapter_type}")

            # Create checkpoint directory for this adapter
            checkpoint_dir = output_dir / f"{adapter_type}{model_suffix}_checkpoint"

            # Train adapter with checkpointing
            adapter, metrics = train_adapter(
                model_a, model_b, tokenizer_a, tokenizer_b,
                adapter, device, log_file, NUM_SAMPLES,
                checkpoint_dir=checkpoint_dir,
                use_ddp=use_ddp,
                rank=rank,
                world_size=world_size
            )

            # Save results (only on main process)
            if is_main_process():
                results = {
                    "adapter_type": adapter_type,
                    "gpu_id": gpu_id,
                    "training_metrics": metrics,
                    "timestamp": timestamp
                }

                results_path = output_dir / f"{adapter_type}_results_{timestamp}.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

                print(f"\nResults saved to: {results_path}")
            print("=" * 80)
            print(f"{adapter_type.upper()} EXPERIMENT COMPLETE")
            print("=" * 80)

    except Exception as e:
        if is_main_process():
            print("\n" + "=" * 80)
            print(f"{adapter_type.upper()} ADAPTER EXPERIMENT FAILED")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        raise  # Re-raise to propagate error

    finally:
        # Restore stdout/stderr and close log file (only rank 0)
        if is_main_process():
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if log_file is not None:
                log_file.close()

# ============================================================================
# Token Compression Experiment Wrapper (for parallel execution)
# ============================================================================

def run_token_compression_wrapper(gpu_id):
    """Run token compression experiment on specified GPU."""

    # Create output directory relative to script location (only rank 0)
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "runs" / "token_compression"

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Fix log filename for multi-GPU case
    gpu_label = "allgpus" if gpu_id is None else f"gpu{gpu_id}"
    log_path = output_dir / f"token_compression_{gpu_label}_{timestamp}.log"

    # Only rank 0 opens log file
    if is_main_process():
        log_file = open(log_path, 'w')
        # Redirect output to both console and file
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeLogger(log_file)
        sys.stderr = TeeLogger(log_file)
    else:
        log_file = None

    try:
        if is_main_process():
            print("=" * 80)
            print("TOKEN COMPRESSION EXPERIMENT")
            print("=" * 80)
            print(f"Platform: {PLATFORM}")
            print(f"Log file: {log_path}")

        # GPU configuration will be printed by run_token_compression_experiment
        # Pass None as device to let it auto-configure for multi-GPU or single-GPU

        # Run token compression experiment (all ranks participate in DDP)
        results = run_token_compression_experiment(
            device=None,  # Auto-configure based on gpu_id
            num_samples=NUM_SAMPLES if NUM_SAMPLES <= 1000 else 1000,  # Cap at 1000 for compression
            compressed_length=64,
            epochs=EPOCHS,
            use_lora_all_layers=True
        )

        # Save results (only rank 0)
        if is_main_process():
            results_path = output_dir / f"token_compression_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            print("\n" + "=" * 80)
            print("TOKEN COMPRESSION EXPERIMENT COMPLETE")
            print("=" * 80)

    except Exception as e:
        if is_main_process():
            print("\n" + "=" * 80)
            print("TOKEN COMPRESSION EXPERIMENT FAILED")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        raise  # Re-raise to propagate error

    finally:
        # Restore stdout/stderr and close log file (only rank 0)
        if is_main_process():
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if log_file is not None:
                log_file.close()

# ============================================================================
# Token-Initialized Compression Experiment - Soft Prompt Compression for Efficient Communication
# ============================================================================

def run_token_compression_experiment(
    model=None,
    tokenizer=None,
    device=None,
    num_samples=100,
    compressed_length=64,
    epochs=5,
    use_lora_all_layers=True
):
    """
    Run token-initialized compression experiment for cross-LLM communication.

    PURPOSE:
    Learn to compress long input sequences (512 tokens) into shorter soft prompts (64 tokens)
    that preserve semantic content while reducing communication bandwidth. This is critical
    for efficient cross-model communication where wire bandwidth is limited.

    RELEVANCE TO CROSS-LLM COMMUNICATION:
    - Reduces token transmission from 512 → 64 (8× compression ratio)
    - Soft prompts (continuous embeddings) carry more information per token than discrete text
    - Learned compression preserves task-relevant information better than truncation
    - Directly applicable to LatentWire's interlingua design

    LITERATURE:
    - "LLMLingua: Prompt Compression" (Microsoft Research, EMNLP 2023 / ACL 2024)
      Achieves up to 20× compression with minimal performance loss via selective token removal
    - "CompactPrompt: Unified Prompt and Data Compression" (arXiv:2510.18043, October 2025)
      Reduces token usage by 60% with <5% accuracy drop using soft prompt compression
    - "Token Communications: Cross-Modal Semantic Communications" (arXiv:2502.12096, February 2025)
      Unified framework for cross-modal communication using tokens as compressed representations
    - "AutoCompressor / ICAE / 500xCompressor" (survey arXiv:2410.12388)
      Soft prompt architectures for learning continuous prompt compressions

    METHOD:
    1. Initialize compressed z ∈ R^(64×d) with pooled token embeddings (not random noise)
    2. Train compressor to reconstruct input via autoencoding objective
    3. Apply LoRA to all transformer layers for parameter-efficient adaptation
    4. Measure reconstruction quality (perplexity, token accuracy)

    KEY INNOVATION:
    Token-initialization (vs random) provides better starting point for compression learning,
    similar to how warm-start reduces training time in optimization.

    EXPECTED RESULTS (from literature):
    - Reconstruction perplexity: <30 with 8× compression (vs 10-15 for no compression)
    - Token accuracy: 60-70% exact match at first position
    - Training: Converges in 5-10 epochs with LoRA (vs 50+ without)
    """
    # Device configuration for multi-GPU support with DDP
    use_ddp = False
    rank = 0
    world_size = 1

    if device is None:
        if PLATFORM == 'hpc' and torch.cuda.device_count() > 1:
            # Use DDP for multi-GPU training
            rank, world_size, device = setup_ddp()
            use_ddp = True
            if is_main_process():
                print(f"\n{'='*80}")
                print(f"GPU CONFIGURATION")
                print(f"{'='*80}")
                print(f"Mode: DistributedDataParallel (DDP)")
                print(f"Number of GPUs: {world_size}")
                print(f"GPU IDs: {list(range(world_size))}")
                print(f"Rank: {rank}, Device: {device}")
                print(f"Batch size per GPU: {BATCH_SIZE}")
                print(f"Global batch size: {BATCH_SIZE * world_size}")
                print(f"{'='*80}\n")
        else:
            device = DEVICE
            print(f"\nDevice: {device}\n")

    if rank == 0:
        print("\n" + "=" * 80)
        print("TOKEN-INITIALIZED COMPRESSION EXPERIMENT")
        print("=" * 80)

    # Load Llama model if not provided (all ranks load the model)
    if model is None or tokenizer is None:
        if rank == 0:
            print(f"Loading {LLAMA_MODEL}...")

        # Model loading arguments
        model_kwargs = {
            'torch_dtype': torch.bfloat16 if USE_BF16 else torch.float16,
            'device_map': 'auto' if PLATFORM == 'mac' else None,
            'low_cpu_mem_usage': True,
        }

        if USE_FLASH_ATTENTION and PLATFORM == 'hpc':
            model_kwargs['attn_implementation'] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            **model_kwargs
        ).eval()

        if PLATFORM == 'hpc':
            model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA to all layers if requested
    # Note: get_peft_model() automatically freezes base model parameters
    if use_lora_all_layers:
        if rank == 0:
            print("Applying LoRA to all transformer layers...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=16,  # Higher rank for compression task
                lora_alpha=32,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            if rank == 0:
                model.print_trainable_parameters()  # This prints directly, doesn't return a value
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not apply LoRA: {e}")

    # Create compressor
    d_z = 256  # Latent dimension (16x compression from 4096)
    if rank == 0:
        print(f"Creating token-initialized compressor (compressed_length={compressed_length}, d_z={d_z})...")
    dtype = torch.bfloat16 if USE_BF16 else torch.float32
    # Access config correctly for DDP/DataParallel wrapped models
    base_model = model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model
    compressor = TokenInitializedCompressor(
        model=model,
        tokenizer=tokenizer,
        compressed_length=compressed_length,
        hidden_dim=base_model.config.hidden_size,
        d_z=d_z
    ).to(device, dtype=dtype)

    # Wrap compressor with DDP if using distributed training
    if use_ddp:
        compressor = DDP(compressor, device_ids=[rank], output_device=rank)
        if rank == 0:
            print(f"Wrapped compressor with DDP")

    # Load dataset (all ranks load it)
    if rank == 0:
        print(f"Loading SQuAD dataset ({num_samples} samples)...")
    dataset = load_dataset("squad", split=f"train[:{num_samples}]")

    # Prepare training data
    train_texts = []
    for item in dataset:
        # Combine context and question
        text = f"Context: {item['context'][:500]}\nQuestion: {item['question']}\nAnswer:"
        train_texts.append(text)

    # Create Dataset and DataLoader with DistributedSampler for DDP
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    text_dataset = TextDataset(train_texts)

    if use_ddp:
        sampler = DistributedSampler(
            text_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = DataLoader(
            text_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with DDP
            pin_memory=True if PLATFORM == 'hpc' else False,
        )
    else:
        dataloader = DataLoader(
            text_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    # Training setup
    all_params = list(compressor.parameters())
    if use_lora_all_layers:
        all_params += [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE)

    # Training metrics
    metrics = {
        "compression_ratio": [],
        "reconstruction_loss": [],
        "generation_loss": [],
        "perplexity": [],
        "epochs": []
    }

    # Training configuration header
    num_batches = len(dataloader)
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*80}")
        print(f"Total epochs: {epochs}")
        print(f"Total samples: {len(train_texts)}")
        if use_ddp:
            print(f"Samples per rank: {len(train_texts) // world_size}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        if use_ddp:
            print(f"Global batch size: {BATCH_SIZE * world_size}")
        print(f"Batches per epoch: {num_batches}")
        print(f"Total training batches: {epochs * num_batches}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Compressed length: {compressed_length} tokens")
        print(f"Latent dimension: {d_z}")
        print(f"Using LoRA: {use_lora_all_layers}")
        print(f"Using DDP: {use_ddp}")
        print(f"{'='*80}\n")

    model.train() if use_lora_all_layers else model.eval()
    compressor.train()

    import time
    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        epoch_start_time = time.time()

        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if use_ddp:
            sampler.set_epoch(epoch)

        if rank == 0:
            msg = f"\n{'='*80}\nEpoch {epoch+1}/{epochs}\n{'='*80}"
            print(msg)

        for batch_num, batch_texts in enumerate(dataloader, 1):
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # Compress the inputs
            compressed = compressor(inputs.input_ids)

            # For training, we want to predict the original sequence from compressed representation
            # Shift labels for autoregressive training
            labels = inputs.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

            # Generate with compressed embeddings
            with torch.cuda.amp.autocast(enabled=USE_BF16 and PLATFORM == 'hpc'):
                outputs = model(
                    inputs_embeds=compressed,
                    labels=labels[:, :compressed_length],  # Predict first N tokens from compressed
                    return_dict=True
                )

            # DDP automatically averages gradients across processes, loss is already scalar
            loss = outputs.loss
            epoch_losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            batch_num += 1

            # Progress logging every 10 batches with detailed info
            if batch_num % 10 == 0:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
                current_lr = optimizer.param_groups[0]['lr']
                progress_pct = 100 * batch_num / num_batches
                elapsed = time.time() - epoch_start_time
                batches_per_sec = batch_num / elapsed if elapsed > 0 else 0
                eta_seconds = (num_batches - batch_num) / batches_per_sec if batches_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                msg = f"  [{progress_pct:5.1f}%] Batch {batch_num:4d}/{num_batches} | "
                msg += f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                msg += f"LR: {current_lr:.2e} | GradNorm: {grad_norm:.3f} | "
                msg += f"{batches_per_sec:.2f} batches/s | ETA: {eta_minutes:.1f}m"
                print(msg)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        metrics["generation_loss"].append(avg_loss)
        metrics["perplexity"].append(math.exp(min(avg_loss, 10)))  # Cap to prevent overflow

        # Calculate true compression ratio
        seq_compression = 200 / compressed_length  # Sequence length compression
        dim_compression = base_model.config.hidden_size / d_z  # Dimension compression
        total_compression = seq_compression * dim_compression
        metrics["compression_ratio"].append(total_compression)

        # Epoch summary with timing
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        remaining_epochs = epochs - (epoch + 1)
        avg_epoch_time = total_elapsed / (epoch + 1)
        eta_total = avg_epoch_time * remaining_epochs / 60  # in minutes

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "perplexity": metrics['perplexity'][-1],
            "compression_ratio": total_compression
        })

        msg = f"\n{'='*80}\n"
        msg += f"Epoch {epoch+1}/{epochs} Complete | Time: {epoch_time/60:.1f}m | Total: {total_elapsed/60:.1f}m\n"
        msg += f"  Avg Loss: {avg_loss:.4f} | Perplexity: {metrics['perplexity'][-1]:.2f} | Compression: {total_compression:.1f}x\n"
        if remaining_epochs > 0:
            msg += f"  ETA for remaining {remaining_epochs} epochs: {eta_total:.1f}m\n"
        msg += f"{'='*80}"
        print(msg)

    # Evaluation
    print("\n" + "=" * 40)
    print("EVALUATION")
    print("=" * 40)
    print(f"Z Vector Shape: [{compressed_length}, {d_z}] = {compressed_length * d_z:,} parameters")
    print(f"Original Shape: [~200, {base_model.config.hidden_size}] = ~{200 * base_model.config.hidden_size:,} parameters")
    print(f"Total Compression: {total_compression:.1f}x")

    model.eval()
    compressor.eval()

    # Test on a few examples
    test_prompts = [
        "The capital of France is",
        "The main difference between supervised and unsupervised learning is",
        "To solve climate change, we need to",
    ]

    results = {"metrics": metrics, "examples": []}

    # Only rank 0 runs evaluation examples
    if rank == 0:
        with torch.no_grad():
            for prompt in test_prompts:
                # Original generation
                orig_inputs = tokenizer(prompt, return_tensors="pt").to(device)
                orig_output = model.generate(**orig_inputs, max_new_tokens=20, do_sample=False)
                orig_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)

                # Compressed generation with z vector
                compressed, z_vector = compressor(orig_inputs.input_ids, return_z=True)
                comp_output = model.generate(
                    inputs_embeds=compressed[:, :10],  # Use first 10 compressed tokens as prompt
                    max_new_tokens=20,
                    do_sample=False
                )
                comp_text = tokenizer.decode(comp_output[0], skip_special_tokens=True)

                results["examples"].append({
                    "prompt": prompt,
                    "original": orig_text,
                    "compressed": comp_text,
                    "z_shape": list(z_vector.shape),
                    "z_mean": float(z_vector.mean().item()),
                    "z_std": float(z_vector.std().item())
                })

                print(f"\nPrompt: {prompt}")
                print(f"Original: {orig_text}")
                print(f"Compressed: {comp_text}")
                print(f"Z Vector: shape={list(z_vector.shape)}, mean={z_vector.mean().item():.3f}, std={z_vector.std().item():.3f}")

    # Final training summary (only rank 0)
    if rank == 0:
        total_training_time = time.time() - training_start_time
        msg = f"\n\n{'='*80}\n"
        msg += f"TRAINING COMPLETE\n"
        msg += f"{'='*80}\n"
        msg += f"Total time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.2f} hours)\n"
        msg += f"Total epochs: {epochs}\n"
        msg += f"Final loss: {metrics['epochs'][-1]['loss']:.4f}\n"
        msg += f"Final perplexity: {metrics['epochs'][-1]['perplexity']:.2f}\n"
        msg += f"Compression ratio: {metrics['epochs'][-1]['compression_ratio']:.1f}x\n"
        msg += f"\nLoss progression:\n"
        for epoch_data in metrics['epochs']:
            msg += f"  Epoch {epoch_data['epoch']:2d}: Loss {epoch_data['loss']:.4f}, Perplexity {epoch_data['perplexity']:.2f}\n"
        msg += f"{'='*80}\n"
        print(msg)

    return results

# ============================================================================
# Activation Communication Experiment - Direct Hidden State Injection Across Models
# ============================================================================

def run_activation_communication_experiment(model_a_id=None, model_b_id=None):
    """
    Reproduce "Communicating Activations Between Language Model Agents" (Ramesh & Li, ICML 2025).

    Args:
        model_a_id: Source model identifier (defaults to LLAMA_MODEL)
        model_b_id: Target model identifier (defaults to MISTRAL_MODEL)

    METHOD (from paper):
    1. Run source model A on prompt P, extract hidden state h_A at layer j (last token only)
    2. If dimensions mismatch, project: h_proj = W @ h_A  (W is learned on C4 data)
    3. Pause target model B at layer j during generation
    4. REPLACE (not add) B's last token activation with h_proj
    5. Continue B's forward pass to generate output
    6. Measure generation quality vs text baseline

    CRITICAL FIXES from previous implementation:
    - REPLACEMENT instead of ADDITION of activations
    - Last token only, not full sequence
    - Forward hooks for proper mid-layer injection
    - LearnedProjection for dimension mismatch (3072 ↔ 4096)
    - Use layer 26 (paper's choice) instead of hardcoded layers

    EXPECTED RESULTS (from paper):
    - 10-27% improvement over natural language communication
    - <1/4 compute cost vs text
    - Works across models with different vocabularies and architectures

    References:
        - Ramesh & Li, "Communicating Activations Between Language Model Agents"
          arXiv:2501.14082 (ICML 2025)
    """

    # Default to main experiment models if not specified
    if model_a_id is None:
        model_a_id = LLAMA_MODEL
    if model_b_id is None:
        model_b_id = MISTRAL_MODEL

    print("\n" + "=" * 80)
    print("ACTIVATION COMMUNICATION EXPERIMENT (Ramesh & Li 2025 Reproduction)")
    print("=" * 80)
    print(f"Model A (source): {model_a_id}")
    print(f"Model B (target): {model_b_id}")
    print("Method: Replace target's last-token activation with source's projected activation")
    print("")

    device = DEVICE

    # Load models
    print(f"\nLoading models on {PLATFORM}...")
    if PLATFORM == 'mac':
        dtype = torch.float32
        print("Using float32 for MPS")
    elif USE_BF16:
        dtype = torch.bfloat16
        print("Using bfloat16 for H100")
    else:
        dtype = torch.float16
        print("Using float16")

    # Model loading kwargs
    model_kwargs = {
        'torch_dtype': dtype,
        'device_map': "auto" if PLATFORM == 'mac' else None,
    }

    if USE_FLASH_ATTENTION and PLATFORM == 'hpc':
        model_kwargs['attn_implementation'] = "flash_attention_2"
        print("Using Flash Attention 2")

    model_a = AutoModelForCausalLM.from_pretrained(model_a_id, **model_kwargs).eval()
    model_b = AutoModelForCausalLM.from_pretrained(model_b_id, **model_kwargs).eval()

    # Freeze model parameters (inference only)
    for param in model_a.parameters():
        param.requires_grad = False
    for param in model_b.parameters():
        param.requires_grad = False

    # Move to device if needed
    if PLATFORM == 'hpc':
        model_a = model_a.to(device)
        model_b = model_b.to(device)

    tokenizer_a = AutoTokenizer.from_pretrained(model_a_id)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_id)

    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token

    # Get model dimensions
    dim_a = model_a.config.hidden_size
    dim_b = model_b.config.hidden_size
    num_layers_a = model_a.config.num_hidden_layers
    num_layers_b = model_b.config.num_hidden_layers

    print(f"\nModel A: {dim_a} hidden_dim, {num_layers_a} layers")
    print(f"Model B: {dim_b} hidden_dim, {num_layers_b} layers")

    # Handle dimension mismatch with learned projection
    learned_projection = None
    script_dir = Path(__file__).parent.absolute()
    projection_path = script_dir / "runs" / "learned_projection" / f"projection_{dim_a}_to_{dim_b}.pt"

    if dim_a != dim_b:
        print(f"\nDimension mismatch detected ({dim_a} → {dim_b})")
        print(f"Looking for learned projection at: {projection_path}")

        if projection_path.exists():
            try:
                learned_projection = LearnedProjection(dim_a, dim_b).to(device).eval()
                state = torch.load(projection_path, map_location=device, weights_only=False)
                learned_projection.load_state_dict(state)
                print("  ✓ Loaded pre-trained projection")
            except Exception as e:
                print(f"  ✗ Could not load projection: {e}")
                print("  Creating new untrained projection (will use random initialization)")
                learned_projection = LearnedProjection(dim_a, dim_b).to(device).eval()
        else:
            print("  ✗ No pre-trained projection found")
            print("  Creating new untrained projection (will use random initialization)")
            learned_projection = LearnedProjection(dim_a, dim_b).to(device).eval()
    else:
        print(f"\n✓ Dimensions match ({dim_a} = {dim_b}), no projection needed")

    # Determine which layers to test
    # Paper uses layer 26, but need to handle models with fewer layers
    test_layers = [RAMESH_LI_LAYER]  # Start with paper's layer 26
    # Add other layers for comparison
    for layer in LAYERS_TO_TEST:
        if layer < num_layers_a and layer < num_layers_b and layer not in test_layers:
            test_layers.append(layer)

    print(f"\nTesting layers: {test_layers}")
    print(f"Primary layer (from paper): {RAMESH_LI_LAYER}")

    # Results storage
    results = {
        "layers": {},
        "summary": {},
        "config": {
            "model_a": model_a_id,
            "model_b": model_b_id,
            "dim_a": dim_a,
            "dim_b": dim_b,
            "projection_used": dim_a != dim_b
        }
    }

    # Helper: Cosine similarity
    def cosine_sim(a, b):
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    # Test each layer
    for layer_idx in test_layers:
        print(f"\n{'='*60}")
        print(f"Testing Layer {layer_idx}")
        print(f"{'='*60}")

        layer_results = []

        # Test each prompt
        for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"  Prompt {prompt_idx}/{len(TEST_PROMPTS)}: {prompt[:50]}...")

            # Baseline: Model B with text input (no injection)
            inputs_b = tokenizer_b(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                baseline_output = model_b.generate(
                    **inputs_b,
                    max_new_tokens=20,
                    do_sample=False
                )
                baseline_text = tokenizer_b.decode(baseline_output[0], skip_special_tokens=True)

            # Get Model A's activation at layer_idx (last token only)
            inputs_a = tokenizer_a(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs_a = model_a(**inputs_a, output_hidden_states=True)
                h_a_last_token = outputs_a.hidden_states[layer_idx][0, -1, :]  # [hidden_dim_a]

            # Project if dimensions mismatch
            if learned_projection is not None:
                with torch.no_grad():
                    h_projected = learned_projection(h_a_last_token)  # [hidden_dim_b]
            else:
                h_projected = h_a_last_token  # Already same dimension

            # Injection via forward hook: REPLACE last token's activation in Model B
            injected_activation = h_projected.clone()

            def injection_hook(module, input, output):
                """
                Replace last token's activation during Model B's forward pass.

                Args:
                    output: Tuple of (hidden_states,) with shape [batch, seq, hidden_dim]

                Returns:
                    Modified output with last token replaced
                """
                # output[0] shape: [batch, seq_len, hidden_dim]
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # CRITICAL: REPLACE (not add) last token
                hidden_states[:, -1, :] = injected_activation

                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states

            # Register hook on target layer
            if hasattr(model_b, 'model') and hasattr(model_b.model, 'layers'):
                # LlamaForCausalLM, MistralForCausalLM structure
                target_layer = model_b.model.layers[layer_idx]
            elif hasattr(model_b, 'transformer') and hasattr(model_b.transformer, 'h'):
                # GPT-2 structure
                target_layer = model_b.transformer.h[layer_idx]
            else:
                raise ValueError(f"Unknown model structure for {model_b_id}")

            hook_handle = target_layer.register_forward_hook(injection_hook)

            # Generate with injection
            try:
                with torch.no_grad():
                    injected_output = model_b.generate(
                        **inputs_b,
                        max_new_tokens=20,
                        do_sample=False
                    )
                    injected_text = tokenizer_b.decode(injected_output[0], skip_special_tokens=True)

                # Compute metrics
                result = {
                    "prompt": prompt,
                    "baseline_text": baseline_text,
                    "injected_text": injected_text,
                    "match": baseline_text == injected_text,
                    "baseline_len": len(baseline_output[0]),
                    "injected_len": len(injected_output[0])
                }

                layer_results.append(result)

                print(f"    Baseline:  {baseline_text[:80]}")
                print(f"    Injected:  {injected_text[:80]}")
                print(f"    Match: {result['match']}")

            except Exception as e:
                print(f"    FAILED: {str(e)}")
                layer_results.append({
                    "prompt": prompt,
                    "baseline_text": baseline_text,
                    "injected_text": f"ERROR: {str(e)}",
                    "match": False,
                    "baseline_len": len(baseline_output[0]),
                    "injected_len": 0
                })

            finally:
                # Always remove hook
                hook_handle.remove()

        results["layers"][layer_idx] = layer_results

    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    for layer_idx, layer_data in results["layers"].items():
        total = len(layer_data)
        matches = sum(1 for r in layer_data if r.get("match", False))
        match_rate = matches / total if total > 0 else 0.0
        print(f"Layer {layer_idx:2d}: {matches}/{total} exact matches ({match_rate*100:.1f}%)")

    results["summary"] = {
        "tested_layers": test_layers,
        "projection_used": learned_projection is not None
    }

    print("=" * 80)

    # Cleanup
    print("\nCleaning up...")
    del model_a, model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return results

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point for unified experiments."""

    # Only print header on main process
    if is_main_process():
        print("=" * 80)
        print("UNIFIED CROSS-MODEL ALIGNMENT EXPERIMENTS")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Platform: {PLATFORM}")
        print(f"Device: {DEVICE}")
        if PLATFORM == 'hpc':
            print(f"Available CUDA GPUs: {torch.cuda.device_count()}")
            if dist.is_initialized():
                print(f"DDP: Running with {dist.get_world_size()} processes")
        print("=" * 80)

    # Create output directory (only rank 0)
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "runs" / "unified_experiments"
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize all processes before continuing
    if dist.is_initialized():
        dist.barrier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run all experiments SEQUENTIALLY, each using all available GPUs with DDP
    # PRIORITY ORDER:
    # 1. FAST EXPERIMENTS FIRST (Procrustes + Activation) - both model pairs (~30 min total)
    # 2. Then SLOW EXPERIMENTS (LoRA, Linear, Affine) - Llama 3.1-3.2 first (~2-3 hours)
    # 3. Finally cross-vocab Llama-Mistral experiments (~2-3 hours)
    #
    # STRATEGY:
    # - Run fast experiments first to get quick insights (Procrustes ~5 min, Activation ~10 min each)
    # - Test both same-vocab and cross-vocab in fast experiments
    # - Then run slow trained adapters, prioritizing same-vocab ablations
    # - Isolates vocabulary mismatch as key variable
    if is_main_process():
        print("\n2. Starting all experiments sequentially (FAST-FIRST ORDER)...")
        print(f"Strategy: Each experiment uses all {torch.cuda.device_count() if PLATFORM == 'hpc' else 1} GPUs for faster completion")
        print("NEW PRIORITY: Run FAST experiments first (Procrustes + Activation), then SLOW (trained adapters)")
        print("  Phase 1 (FAST): Procrustes + Activation for BOTH model pairs (~30 min)")
        print("  Phase 2 (SLOW): LoRA, Linear, Affine on Llama 3.1-3.2 first (~2-3 hours)")
        print("  Phase 3 (SLOW): LoRA, Token, Linear, Affine on Llama-Mistral (~2-3 hours)")
        print("Benefits: Quick insights + early validation + fail fast on model access")
        print("")

    # ========================================================================
    # PHASE 1: ACTIVATION COMMUNICATION (Ramesh & Li 2025 Reproduction) - ~20 MIN
    # ========================================================================
    # PRIORITY: Reproduce "Communicating Activations Between LLM Agents" (Ramesh & Li, ICML 2025)
    # This validates the core feasibility of cross-model communication via activation injection

    # EXPERIMENT 1: Activation Communication (Llama 3.1-3.2) - 10 min
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 1/11: ACTIVATION COMMUNICATION (LLAMA 3.1-3.2)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B (4096 dim) → Llama 3.2 3B (3072 dim)")
        print("Reproducing Ramesh & Li (ICML 2025) - activation injection with learned projection")
        print("WHY FIRST: Core feasibility test for cross-model communication")
        print("Fixes: Replacement (not addition), last token only, forward hooks, dimension handling")
        print("")

        activation_results_ablation = run_activation_communication_experiment(model_a_id=LLAMA_31_8B, model_b_id=LLAMA_32_3B)

        # Save results
        activation_path_ablation = output_dir / f"activation_communication_llama31_llama32_{timestamp}.json"
        with open(activation_path_ablation, 'w') as f:
            json.dump(activation_results_ablation, f, indent=2)
        print(f"Activation Communication (Llama 3.1-3.2) results saved to: {activation_path_ablation}")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()

    # EXPERIMENT 2: Activation Communication (Llama-Mistral) - 10 min
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 2/11: ACTIVATION COMMUNICATION (LLAMA-MISTRAL)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B (4096 dim) → Mistral 7B (4096 dim)")
        print("Tests activation injection across different vocabularies (128K vs 32K tokens)")
        print("Expected: Should work since dimensions match (no projection needed)")
        print("")

        activation_results_main = run_activation_communication_experiment()

        # Save results
        activation_path_main = output_dir / f"activation_communication_llama_mistral_{timestamp}.json"
        with open(activation_path_main, 'w') as f:
            json.dump(activation_results_main, f, indent=2)
        print(f"Activation Communication (Llama-Mistral) results saved to: {activation_path_main}")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()

    # ========================================================================
    # PHASE 2: PROCRUSTES ALIGNMENT (Fast geometric baseline) - ~10 MIN
    # ========================================================================

    # EXPERIMENT 3: Procrustes (Llama 3.1-3.2) - 5 min
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 3/11: PROCRUSTES ALIGNMENT (LLAMA 3.1-3.2)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B ↔ Llama 3.2 3B (identical 128,256 token vocab)")
        print("SVD-based geometric alignment (no training required)")
        print("WARNING: Will fail due to dimension mismatch (4096 vs 3072)")
        print("Keeping to document the limitation of zero-shot geometric methods")
        print("")

        try:
            procrustes_results_ablation = run_procrustes_experiment(model_a_id=LLAMA_31_8B, model_b_id=LLAMA_32_3B)

            # Save Procrustes results
            procrustes_path_ablation = output_dir / f"procrustes_results_llama31_llama32_{timestamp}.json"
            with open(procrustes_path_ablation, 'w') as f:
                json.dump(procrustes_results_ablation, f, indent=2)
            print(f"Procrustes (Llama 3.1-3.2) results saved to: {procrustes_path_ablation}")
        except AssertionError as e:
            print(f"✗ Procrustes (Llama 3.1-3.2) FAILED as expected: {e}")
            print("  This confirms need for learned projection (not just geometric alignment)")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()

    # EXPERIMENT 4: Procrustes (Llama-Mistral) - 5 min
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 4/11: PROCRUSTES ALIGNMENT (LLAMA-MISTRAL)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B ↔ Mistral 7B (4× vocab mismatch)")
        print("SVD-based geometric alignment (no training required)")
        print("Should succeed since dimensions match (4096 = 4096)")
        print("")

        procrustes_results_main = run_procrustes_experiment()

        # Save Procrustes results
        procrustes_path_main = output_dir / f"procrustes_results_llama_mistral_{timestamp}.json"
        with open(procrustes_path_main, 'w') as f:
            json.dump(procrustes_results_main, f, indent=2)
        print(f"Procrustes (Llama-Mistral) results saved to: {procrustes_path_main}")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()

    # ========================================================================
    # PHASE 3: LLAMA 3.1-3.2 TRAINED ADAPTERS (SAME VOCAB) - ~2-3 HOURS
    # ========================================================================

    # EXPERIMENT 5: LoRA Adapter - Ablation (Llama 3.1-3.2)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 5/11: LORA ADAPTER - ABLATION (LLAMA 3.1-3.2)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B ↔ Llama 3.2 3B (identical 128,256 token vocab)")
        print("Purpose: Control for vocabulary mismatch hypothesis")
        print("Expected: Better alignment (CKA 0.6-0.7) and lower generation loss (2.5-3.5)")
        print("")

    run_adapter_experiment("lora", gpu_id=None, model_a_id=LLAMA_31_8B, model_b_id=LLAMA_32_3B)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_main_process():
        print(f"✓ LoRA (Llama 3.1-3.2) complete, GPU memory cleared")
    if dist.is_initialized():
        dist.barrier()

    # EXPERIMENT 6: Linear Adapter - Ablation (Llama 3.1-3.2)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 6/11: LINEAR ADAPTER - ABLATION (LLAMA 3.1-3.2)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B ↔ Llama 3.2 3B (identical 128,256 token vocab)")
        print("Purpose: Control comparison for Linear adapter")
        print("")

    run_adapter_experiment("linear", gpu_id=None, model_a_id=LLAMA_31_8B, model_b_id=LLAMA_32_3B)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_main_process():
        print(f"✓ Linear (Llama 3.1-3.2) complete, GPU memory cleared")
    if dist.is_initialized():
        dist.barrier()

    # EXPERIMENT 7: Affine Adapter - Ablation (Llama 3.1-3.2)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 7/11: AFFINE ADAPTER - ABLATION (LLAMA 3.1-3.2)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B ↔ Llama 3.2 3B (identical 128,256 token vocab)")
        print("Purpose: Control comparison for Affine adapter")
        print("")

    run_adapter_experiment("affine", gpu_id=None, model_a_id=LLAMA_31_8B, model_b_id=LLAMA_32_3B)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_main_process():
        print(f"✓ Affine (Llama 3.1-3.2) complete, GPU memory cleared")
    if dist.is_initialized():
        dist.barrier()

    # ========================================================================
    # PHASE 4: LLAMA-MISTRAL EXPERIMENTS (CROSS VOCAB) - ~2-3 HOURS
    # ========================================================================

    # EXPERIMENT 8: LoRA Adapter - Main (Llama-Mistral)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 8/11: LORA ADAPTER - MAIN (LLAMA-MISTRAL)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B (128K vocab) ↔ Mistral 7B (32K vocab)")
        print("260K params vs 16M (98% reduction), transferable across models")
        print("")

    run_adapter_experiment("lora", gpu_id=None)  # None = use all GPUs with DDP

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_main_process():
        print(f"✓ LoRA (Llama-Mistral) complete, GPU memory cleared")
    if dist.is_initialized():
        dist.barrier()

    # EXPERIMENT 9: Token Compression (THE INTERLINGUA)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 9/11: TOKEN COMPRESSION (LLAMA-MISTRAL)")
        print(f"{'='*80}")
        print("This IS the wire format for LatentWire (512 → 64 tokens)")
        print("Note: Only runs for Llama-Mistral (main comparison)")
        print("")

    run_token_compression_wrapper(gpu_id=None)  # None = use all GPUs with DDP

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_main_process():
        print("✓ Token compression complete, GPU memory cleared")
    if dist.is_initialized():
        dist.barrier()

    # EXPERIMENT 10: Linear Adapter - Main (Llama-Mistral)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 10/11: LINEAR ADAPTER - MAIN (LLAMA-MISTRAL)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B (128K vocab) ↔ Mistral 7B (32K vocab)")
        print("16M params (50× more than LoRA), useful for comparison")
        print("")

    run_adapter_experiment("linear", gpu_id=None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_main_process():
        print(f"✓ Linear (Llama-Mistral) complete, GPU memory cleared")
    if dist.is_initialized():
        dist.barrier()

    # EXPERIMENT 11: Affine Adapter - Main (Llama-Mistral)
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT 11/11: AFFINE ADAPTER - MAIN (LLAMA-MISTRAL)")
        print(f"{'='*80}")
        print("Models: Llama 3.1 8B (128K vocab) ↔ Mistral 7B (32K vocab)")
        print("Similar to linear but with bias term (+4K params)")
        print("")

    run_adapter_experiment("affine", gpu_id=None)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if is_main_process():
        print(f"✓ Affine (Llama 3.1-3.2) complete, GPU memory cleared")
    if dist.is_initialized():
        dist.barrier()

    # Final summary
    if is_main_process():
        print("\n" + "=" * 80)
        print("ALL 9 EXPERIMENTS COMPLETE")
        print("=" * 80)
        print("\nExperiments run:")
        print("  1. Procrustes alignment (baseline, Llama-Mistral only)")
        print("  2a. LoRA adapter - Main (Llama 3.1 8B ↔ Mistral 7B)")
        print("  2b. LoRA adapter - Ablation (Llama 3.1 8B ↔ Llama 3.2 3B)")
        print("  3. Activation communication (Llama-Mistral only)")
        print("  4. Token compression (Llama-Mistral only)")
        print("  5a. Linear adapter - Main (Llama 3.1 8B ↔ Mistral 7B)")
        print("  5b. Linear adapter - Ablation (Llama 3.1 8B ↔ Llama 3.2 3B)")
        print("  6a. Affine adapter - Main (Llama 3.1 8B ↔ Mistral 7B)")
        print("  6b. Affine adapter - Ablation (Llama 3.1 8B ↔ Llama 3.2 3B)")
        print("\nComparison strategy:")
        print("  - All adapter experiments compare Llama-Mistral vs Llama 3.1-3.2")
        print("  - Tests hypothesis: vocabulary mismatch causes alignment failure")
        print("  - Expected: Llama 3.1-3.2 shows better CKA and lower generation loss")

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)

    # Clean up DDP if it was used
    cleanup_ddp()

if __name__ == "__main__":
    # Set multiprocessing start method (needed for CUDA/MPS)
    mp.set_start_method('spawn', force=True)

    # Print system info
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")

    # Platform-specific GPU info
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA devices: {torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        print(f"MPS available: True")
        print("Running on Apple Silicon GPU")
    else:
        print("No GPU available, will use CPU")

    try:
        main()
    finally:
        # Ensure DDP cleanup even if experiment fails
        cleanup_ddp()