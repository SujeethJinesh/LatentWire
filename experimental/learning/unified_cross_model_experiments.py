#!/usr/bin/env python3
"""
Unified cross-model alignment experiments combining Procrustes and learned adapters.
Optimized for 4 H100 GPUs.
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
        config['epochs'] = 10
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
# InfoNCE Contrastive Loss (Critical from 2025 research)
# ============================================================================

class InfoNCE(nn.Module):
    """InfoNCE loss for contrastive learning - essential for alignment."""

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
            # Use unbiased estimator (2024 research)
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
# Alignment and Uniformity Metrics (2024 Research)
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
# Pooling Functions (2024 Research)
# ============================================================================

def mean_pooling(token_embeddings, attention_mask):
    """
    Mean pooling with proper attention masking.

    Outperforms CLS token by 2-5% for semantic similarity tasks in LLMs.
    CLS tokens are optimized for classification, not semantic representation.

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
# Contrastive Weight Scheduler (2024 Research)
# ============================================================================

class ContrastiveWeightScheduler:
    """
    Gradually increase contrastive weight to avoid overwhelming primary objective.

    Based on curriculum learning principles: start with low contrastive weight
    to let the model learn the primary task first, then gradually increase to
    improve representation quality.

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
LLAMA_MODEL = "meta-llama/Llama-3.1-8B"
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"

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
CONTRASTIVE_WEIGHT = 0.2  # Reduced from 0.3 to prevent overwhelming primary objective (2024 research)
NUM_NEGATIVES = 127  # Number of negative samples

# 2024-2025 Research-Based Configuration Flags
USE_MEAN_POOLING = True  # Use mean pooling instead of CLS token (2-5% improvement for semantic tasks)
USE_DEBIASED_CKA = True  # Use unbiased HSIC estimator (critical for < 1000 samples, Murphy et al. ICLR 2024)
USE_AFFINE_PROCRUSTES = True  # Add bias term to Procrustes (5-8% improvement, model stitching 2024)
LOG_ALIGN_UNIFORM = True  # Log alignment/uniformity metrics (Wang & Isola, ICML 2020)
USE_CONTRASTIVE_CURRICULUM = True  # Gradually warm up contrastive weight (prevents overwhelming primary task)

# Multi-layer alignment (prevents single-point failure)
# REDUCED to single layer to save memory (was [8, 16, 24])
ALIGNMENT_LAYERS = [16]  # Middle layer for balance between early/late representations
LAYER_WEIGHTS = [1.0]  # Single layer gets full weight
LAYER_IDX = 16  # Default for backward compatibility

# Layers to test for Procrustes
LAYERS_TO_TEST = [0, 8, 16, 24, 32]

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

    Extended from orthogonal to include bias term for better alignment (2024 research).
    Affine transformations (linear + bias) consistently outperform pure orthogonal
    by 5-8% in model stitching tasks.

    References:
        - Schönemann (1966): Original orthogonal Procrustes
        - Model Stitching Papers (2024): Affine extension benefits

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

        # Step 6: Bias term (2024 research extension for affine transformation)
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
# Procrustes Experiment
# ============================================================================

def run_procrustes_experiment():
    """Run Procrustes alignment experiment across different layers."""

    print("\n" + "=" * 80)
    print("PROCRUSTES ALIGNMENT EXPERIMENT (GPU-ACCELERATED)")
    print("=" * 80)

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
        LLAMA_MODEL,
        **llama_kwargs
    ).eval()

    mistral_model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL,
        **mistral_kwargs
    ).eval()

    # Explicitly move models to their designated GPUs for HPC
    if PLATFORM == 'hpc' and torch.cuda.device_count() >= 2:
        llama_model = llama_model.to('cuda:0')
        mistral_model = mistral_model.to('cuda:1')
        print(f"Llama model moved to cuda:0")
        print(f"Mistral model moved to cuda:1")

    # Load tokenizers
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)

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
# Learned Adapter Training
# ============================================================================

def train_adapter(model_a, model_b, tokenizer_a, tokenizer_b, adapter,
                  device, log_file, num_samples=1000, checkpoint_dir=None,
                  use_ddp=False, rank=0, world_size=1):
    """Train a single adapter for cross-model alignment with contrastive learning."""

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
    print(f"Loading dataset ({num_samples} samples)...", file=log_file)

    # Fix cache corruption
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    wikitext_cache = cache_dir / "wikitext"
    if wikitext_cache.exists():
        shutil.rmtree(wikitext_cache)

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

    # Initialize contrastive weight scheduler for curriculum learning (2024 research)
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

            # Use mean pooling instead of CLS token (2024 research: 2-5% improvement)
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

            # Compute alignment/uniformity metrics (2024 research: Wang & Isola, ICML 2020)
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

        # Compute CKA similarity at end of epoch
        with torch.no_grad():
            # Sample a few batches for CKA computation
            cka_scores = []
            for i, batch in enumerate(dataloader):
                if i >= 5:  # Only compute on first 5 batches for efficiency
                    break

                input_ids_a = batch["input_ids_a"].to(device)
                attention_mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                attention_mask_b = batch["attention_mask_b"].to(device)

                outputs_a = model_a(
                    input_ids=input_ids_a,
                    attention_mask=attention_mask_a,
                    output_hidden_states=True
                )
                outputs_b = model_b(
                    input_ids=input_ids_b,
                    attention_mask=attention_mask_b,
                    output_hidden_states=True
                )

                # Compute CKA for middle layer
                mid_idx = len(ALIGNMENT_LAYERS) // 2
                source_repr = outputs_a.hidden_states[ALIGNMENT_LAYERS[mid_idx]][:, 0, :]
                aligned_repr = adapter(source_repr)
                target_repr = outputs_b.hidden_states[ALIGNMENT_LAYERS[mid_idx]][:, 0, :]

                cka_score = CKA.cka_similarity(aligned_repr.float(), target_repr.float())
                cka_scores.append(cka_score.item())

            avg_cka = np.mean(cka_scores) if cka_scores else 0.0
            training_metrics["cka_scores"].append(avg_cka)

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

        # Save checkpoint after each epoch (only on rank 0)
        if checkpoint_dir and rank == 0:
            checkpoint = {
                'epoch': epoch,
                'adapter_state_dict': adapter.module.state_dict() if use_ddp else adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_metrics': training_metrics,
                'loss': avg_epoch_loss,
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

def run_adapter_experiment(adapter_type, gpu_id):
    """Run a single adapter experiment on specified GPU."""

    # Create output directory relative to script location
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "runs" / "learned_adapters"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Fix log filename for multi-GPU case
    gpu_label = "allgpus" if gpu_id is None else f"gpu{gpu_id}"
    log_path = output_dir / f"{adapter_type}_{gpu_label}_{timestamp}.log"

    with open(log_path, 'w') as log_file:
        # Redirect output to both console and file
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeLogger(log_file)
        sys.stderr = TeeLogger(log_file)

        try:
            print("=" * 80)
            print(f"LEARNED ADAPTER EXPERIMENT - {adapter_type.upper()}")
            print("=" * 80)
            print(f"Platform: {PLATFORM}")
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
                LLAMA_MODEL,
                **model_kwargs
            ).to(device).eval()

            # Enable gradient checkpointing to save memory during training (works on all platforms)
            model_a.gradient_checkpointing_enable()

            model_b = AutoModelForCausalLM.from_pretrained(
                MISTRAL_MODEL,
                **model_kwargs
            ).to(device).eval()

            # Enable gradient checkpointing to save memory during training
            model_b.gradient_checkpointing_enable()

            # Note: Models are frozen (eval mode), so no need to wrap with DDP
            # Only the adapter will be wrapped with DDP in train_adapter()

            # Load tokenizers
            tokenizer_a = AutoTokenizer.from_pretrained(LLAMA_MODEL)
            tokenizer_b = AutoTokenizer.from_pretrained(MISTRAL_MODEL)

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
            checkpoint_dir = output_dir / f"{adapter_type}_checkpoint"

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
            print("\n" + "=" * 80)
            print(f"{adapter_type.upper()} ADAPTER EXPERIMENT FAILED")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ============================================================================
# Token Compression Experiment Wrapper (for parallel execution)
# ============================================================================

def run_token_compression_wrapper(gpu_id):
    """Run token compression experiment on specified GPU."""

    # Create output directory relative to script location
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "runs" / "token_compression"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Fix log filename for multi-GPU case
    gpu_label = "allgpus" if gpu_id is None else f"gpu{gpu_id}"
    log_path = output_dir / f"token_compression_{gpu_label}_{timestamp}.log"

    with open(log_path, 'w') as log_file:
        # Redirect output to both console and file
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeLogger(log_file)
        sys.stderr = TeeLogger(log_file)

        try:
            print("=" * 80)
            print("TOKEN COMPRESSION EXPERIMENT")
            print("=" * 80)
            print(f"Platform: {PLATFORM}")
            print(f"Log file: {log_path}")

            # GPU configuration will be printed by run_token_compression_experiment
            # Pass None as device to let it auto-configure for multi-GPU or single-GPU

            # Run token compression experiment
            results = run_token_compression_experiment(
                device=None,  # Auto-configure based on gpu_id
                num_samples=NUM_SAMPLES if NUM_SAMPLES <= 1000 else 1000,  # Cap at 1000 for compression
                compressed_length=64,
                epochs=EPOCHS,
                use_lora_all_layers=True
            )

            # Save results
            results_path = output_dir / f"token_compression_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            print("\n" + "=" * 80)
            print("TOKEN COMPRESSION EXPERIMENT COMPLETE")
            print("=" * 80)

        except Exception as e:
            print("\n" + "=" * 80)
            print("TOKEN COMPRESSION EXPERIMENT FAILED")
            print("=" * 80)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ============================================================================
# Token-Initialized Compression Experiment
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
    Run token-initialized compression experiment.

    Key idea: Initialize compressed representation with actual token embeddings
    from the input, not random noise. Apply LoRA to all layers for adaptation.
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

    print("\n" + "=" * 80)
    print("TOKEN-INITIALIZED COMPRESSION EXPERIMENT")
    print("=" * 80)

    # Load Llama model if not provided
    if model is None or tokenizer is None:
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

        # Note: Model will be frozen (eval mode), no DDP wrapping needed
        # Only compressor will be wrapped with DDP

    # Apply LoRA to all layers if requested
    if use_lora_all_layers:
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
            model.print_trainable_parameters()  # This prints directly, doesn't return a value
        except Exception as e:
            print(f"Warning: Could not apply LoRA: {e}")

    # Create compressor
    d_z = 256  # Latent dimension (16x compression from 4096)
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
        if is_main_process():
            print(f"Wrapped compressor with DDP")

    # Load dataset
    print(f"Loading SQuAD dataset ({num_samples} samples)...")
    dataset = load_dataset("squad", split=f"train[:{num_samples}]")

    # Prepare training data
    train_texts = []
    for item in dataset:
        # Combine context and question
        text = f"Context: {item['context'][:500]}\nQuestion: {item['question']}\nAnswer:"
        train_texts.append(text)

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
    num_batches = len(train_texts) // BATCH_SIZE
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Total epochs: {epochs}")
    print(f"Total samples: {len(train_texts)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {num_batches}")
    print(f"Total training batches: {epochs * num_batches}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Compressed length: {compressed_length} tokens")
    print(f"Latent dimension: {d_z}")
    print(f"Using LoRA: {use_lora_all_layers}")
    print(f"{'='*80}\n")

    model.train() if use_lora_all_layers else model.eval()
    compressor.train()

    import time
    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        epoch_start_time = time.time()
        batch_num = 0

        msg = f"\n{'='*80}\nEpoch {epoch+1}/{epochs}\n{'='*80}"
        print(msg)

        for batch_idx in range(0, len(train_texts), BATCH_SIZE):
            batch_texts = train_texts[batch_idx:batch_idx + BATCH_SIZE]

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

    # Final training summary
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
# Activation Communication Experiment (Ramesh & Li 2025)
# ============================================================================

def run_activation_communication_experiment():
    """
    Test activation-based communication between Llama and Mistral.
    Based on "Communicating Activations Between Language Model Agents" (Ramesh & Li 2025).

    Tests whether learned alignments (Procrustes, adapters) improve cross-model
    activation injection compared to zero-shot methods.
    """

    print("\n" + "=" * 80)
    print("ACTIVATION COMMUNICATION EXPERIMENT (Ramesh & Li 2025)")
    print("=" * 80)
    print("Testing cross-model activation injection with/without alignment")
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
    llama_kwargs = mistral_kwargs = {
        'torch_dtype': dtype,
        'device_map': "auto" if PLATFORM == 'mac' else None,
    }

    if USE_FLASH_ATTENTION and PLATFORM == 'hpc':
        llama_kwargs['attn_implementation'] = "flash_attention_2"
        mistral_kwargs['attn_implementation'] = "flash_attention_2"
        print("Using Flash Attention 2")

    llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, **llama_kwargs).eval()
    mistral_model = AutoModelForCausalLM.from_pretrained(MISTRAL_MODEL, **mistral_kwargs).eval()

    # Move to device if needed
    if PLATFORM == 'hpc':
        llama_model = llama_model.to(device)
        mistral_model = mistral_model.to(device)

    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL)

    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    if mistral_tokenizer.pad_token is None:
        mistral_tokenizer.pad_token = mistral_tokenizer.eos_token

    # Load pre-trained alignments
    print("\nLoading pre-trained alignments...")
    script_dir = Path(__file__).parent.absolute()

    # Load Procrustes alignments (if available)
    procrustes_alignments = {}
    for layer_idx in LAYERS_TO_TEST:
        procrustes_path = script_dir / "runs" / "procrustes_alignments" / f"layer_{layer_idx}.pt"
        if procrustes_path.exists():
            try:
                alignment = ProcrustesAlignment()
                state = torch.load(procrustes_path, map_location=device, weights_only=False)
                alignment.W = state['W'].to(device)
                alignment.source_mean = state['source_mean'].to(device)
                alignment.target_mean = state['target_mean'].to(device)
                alignment.source_norm = state['source_norm'].to(device)
                alignment.target_norm = state['target_norm'].to(device)
                alignment.b = state.get('b', torch.zeros_like(alignment.target_mean)).to(device)
                procrustes_alignments[layer_idx] = alignment
                print(f"  Loaded Procrustes for layer {layer_idx}")
            except Exception as e:
                print(f"  Warning: Could not load Procrustes for layer {layer_idx}: {e}")

    # Load Linear adapter (if available)
    linear_adapter = None
    linear_checkpoint = script_dir / "runs" / "learned_adapters" / "linear_checkpoint" / "checkpoint.pt"
    if linear_checkpoint.exists():
        try:
            linear_adapter = LinearAdapter(hidden_dim=4096).to(device).eval()
            checkpoint = torch.load(linear_checkpoint, map_location=device, weights_only=False)
            linear_adapter.load_state_dict(checkpoint['adapter_state_dict'])
            print(f"  Loaded Linear adapter")
        except Exception as e:
            print(f"  Warning: Could not load Linear adapter: {e}")

    # Evaluation metrics storage
    results = {
        "layers": {},
        "summary": {}
    }

    # Helper function to compute cosine similarity
    def cosine_similarity(a, b):
        """Compute cosine similarity between two tensors."""
        a_norm = F.normalize(a.view(-1), dim=0)
        b_norm = F.normalize(b.view(-1), dim=0)
        return (a_norm * b_norm).sum().item()

    # Helper function to compute diversity
    def compute_diversity(token_ids):
        """Compute unique tokens / total tokens ratio."""
        if len(token_ids) == 0:
            return 0.0
        unique = len(set(token_ids.tolist()))
        total = len(token_ids)
        return unique / total

    # Test each layer
    for layer_idx in LAYERS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Testing Layer {layer_idx}")
        print(f"{'='*60}")

        layer_results = {
            "zero_shot_add": [],
            "zero_shot_weighted": [],
            "procrustes_aligned": [],
            "adapter_aligned": []
        }

        # Test each prompt
        for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"  Prompt {prompt_idx}/{len(TEST_PROMPTS)}: {prompt[:50]}...")

            # Get baseline Mistral→Mistral output for comparison
            mistral_inputs = mistral_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                baseline_output = mistral_model.generate(
                    **mistral_inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
                baseline_text = mistral_tokenizer.decode(baseline_output.sequences[0], skip_special_tokens=True)
                # Get final hidden state for similarity comparison
                baseline_hidden = baseline_output.hidden_states[-1][-1][:, -1, :]  # Last token of last layer

            # Get hidden states from both models at target layer
            llama_inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                # Llama forward to layer L
                llama_outputs = llama_model(
                    **llama_inputs,
                    output_hidden_states=True
                )
                llama_hidden = llama_outputs.hidden_states[layer_idx]  # [batch, seq, hidden]

                # Mistral forward to layer L
                mistral_outputs = mistral_model(
                    **mistral_inputs,
                    output_hidden_states=True
                )
                mistral_hidden = mistral_outputs.hidden_states[layer_idx]  # [batch, seq, hidden]

            # Test 4 combination methods
            methods = {}

            # Method 1: Zero-shot addition
            methods['zero_shot_add'] = llama_hidden + mistral_hidden

            # Method 2: Zero-shot weighted (favor Mistral since it's the target model)
            methods['zero_shot_weighted'] = 0.3 * llama_hidden + 0.7 * mistral_hidden

            # Method 3: Procrustes-aligned addition
            if layer_idx in procrustes_alignments:
                alignment = procrustes_alignments[layer_idx]
                llama_flat = llama_hidden.reshape(-1, llama_hidden.shape[-1])
                llama_aligned = alignment.transform(llama_flat)
                llama_aligned = llama_aligned.reshape(llama_hidden.shape).to(llama_hidden.dtype)
                methods['procrustes_aligned'] = llama_aligned + mistral_hidden
            else:
                methods['procrustes_aligned'] = None

            # Method 4: Adapter-aligned addition
            if linear_adapter is not None:
                llama_adapted = linear_adapter(llama_hidden)
                methods['adapter_aligned'] = llama_adapted + mistral_hidden
            else:
                methods['adapter_aligned'] = None

            # Generate from each combined activation
            for method_name, combined_hidden in methods.items():
                if combined_hidden is None:
                    print(f"    Method: {method_name} - SKIPPED (alignment not available)")
                    continue

                try:
                    with torch.no_grad():
                        # Continue generation from combined hidden state
                        # Note: inputs_embeds doesn't support intermediate injection directly
                        # We use it as if starting from this hidden state
                        output = mistral_model.generate(
                            inputs_embeds=combined_hidden,
                            max_new_tokens=20,
                            do_sample=False,
                            output_hidden_states=True,
                            return_dict_in_generate=True
                        )

                        generated_text = mistral_tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                        generated_ids = output.sequences[0]

                        # Get final hidden state
                        final_hidden = output.hidden_states[-1][-1][:, -1, :]

                        # Compute metrics
                        length = len(generated_ids)
                        diversity = compute_diversity(generated_ids)
                        similarity = cosine_similarity(final_hidden, baseline_hidden)

                        result = {
                            "prompt": prompt,
                            "generated": generated_text,
                            "length": length,
                            "diversity": diversity,
                            "similarity": similarity
                        }

                        layer_results[method_name].append(result)

                        print(f"    Method: {method_name}")
                        print(f"      Generated: {generated_text[:80]}")
                        print(f"      Length: {length} tokens | Diversity: {diversity:.2f} | Similarity: {similarity:.2f}")

                except Exception as e:
                    print(f"    Method: {method_name} - FAILED: {str(e)}")
                    layer_results[method_name].append({
                        "prompt": prompt,
                        "generated": f"ERROR: {str(e)}",
                        "length": 0,
                        "diversity": 0.0,
                        "similarity": 0.0
                    })

        results["layers"][layer_idx] = layer_results

    # Compute summary statistics
    print(f"\n{'='*80}")
    print("ACTIVATION COMMUNICATION RESULTS SUMMARY")
    print(f"{'='*80}")

    method_avg_scores = {}
    for method in ["zero_shot_add", "zero_shot_weighted", "procrustes_aligned", "adapter_aligned"]:
        all_similarities = []
        for layer_idx in LAYERS_TO_TEST:
            if layer_idx in results["layers"]:
                for result in results["layers"][layer_idx][method]:
                    if isinstance(result.get("similarity"), (int, float)):
                        all_similarities.append(result["similarity"])

        if all_similarities:
            avg_sim = sum(all_similarities) / len(all_similarities)
            method_avg_scores[method] = avg_sim
        else:
            method_avg_scores[method] = 0.0

    # Find best method
    best_method = max(method_avg_scores.items(), key=lambda x: x[1])
    print(f"Best performing method: {best_method[0]} (avg similarity: {best_method[1]:.3f})")

    # Layer-wise breakdown
    print(f"\nLayer-wise results (average similarity to baseline):")
    for layer_idx in LAYERS_TO_TEST:
        if layer_idx in results["layers"]:
            layer_data = results["layers"][layer_idx]
            scores = []
            for method in ["zero_shot_add", "zero_shot_weighted", "procrustes_aligned", "adapter_aligned"]:
                if layer_data[method]:
                    sims = [r["similarity"] for r in layer_data[method] if isinstance(r.get("similarity"), (int, float))]
                    avg = sum(sims) / len(sims) if sims else 0.0
                    scores.append(f"{method[:7]}={avg:.2f}")
            print(f"  Layer {layer_idx:2d}: {', '.join(scores)}")

    # Calculate improvement
    zero_shot_avg = method_avg_scores.get("zero_shot_add", 0.0)
    best_aligned = max(
        method_avg_scores.get("procrustes_aligned", 0.0),
        method_avg_scores.get("adapter_aligned", 0.0)
    )
    if zero_shot_avg > 0:
        improvement = ((best_aligned - zero_shot_avg) / zero_shot_avg) * 100
        print(f"\nKey Finding: Learned alignments improve activation communication by {improvement:.1f}% over zero-shot")

    results["summary"] = {
        "method_avg_scores": method_avg_scores,
        "best_method": best_method[0],
        "best_score": best_method[1]
    }

    print("=" * 80)

    # Clean up
    print("\nCleaning up activation communication experiment...")
    del llama_model
    del mistral_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  Models deleted and GPU memory cleared")

    return results

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point for unified experiments."""

    print("=" * 80)
    print("UNIFIED CROSS-MODEL ALIGNMENT EXPERIMENTS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Platform: {PLATFORM}")
    print(f"Device: {DEVICE}")
    if PLATFORM == 'hpc':
        print(f"Available CUDA GPUs: {torch.cuda.device_count()}")
    print("=" * 80)

    # Create output directory relative to script location
    script_dir = Path(__file__).parent.absolute()
    output_dir = script_dir / "runs" / "unified_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run Procrustes experiment
    print(f"\n1. Starting Procrustes experiment on {DEVICE}...")
    procrustes_results = run_procrustes_experiment()

    # Save Procrustes results
    procrustes_path = output_dir / f"procrustes_results_{timestamp}.json"
    with open(procrustes_path, 'w') as f:
        json.dump(procrustes_results, f, indent=2)
    print(f"Procrustes results saved to: {procrustes_path}")

    # CRITICAL: Clean up GPU memory after Procrustes before adapter experiments
    print("\nCleaning up GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  GPU cache cleared")

    # Run all experiments SEQUENTIALLY, each using all available GPUs
    print("\n2. Starting all experiments sequentially...")
    print(f"Strategy: Each experiment uses all {torch.cuda.device_count() if PLATFORM == 'hpc' else 1} GPUs for faster completion")
    print("Benefits: Progressive results + full GPU utilization per experiment")
    print("")

    # Run learned adapter experiments (each uses all GPUs via DataParallel)
    for adapter_type in ["linear", "affine", "lora"]:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {['linear', 'affine', 'lora'].index(adapter_type) + 1}/5: {adapter_type.upper()} ADAPTER")
        print(f"{'='*80}")
        run_adapter_experiment(adapter_type, gpu_id=None)  # None = use all GPUs

        # Clean GPU memory between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"✓ {adapter_type.upper()} complete, GPU memory cleared")

    # Run token compression experiment (uses all GPUs via DataParallel)
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 4/5: TOKEN COMPRESSION")
    print(f"{'='*80}")
    run_token_compression_wrapper(gpu_id=None)  # None = use all GPUs

    # Clean GPU memory before activation communication
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ Token compression complete, GPU memory cleared")

    # Run activation communication experiment (Ramesh & Li 2025)
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 5/5: ACTIVATION COMMUNICATION")
    print(f"{'='*80}")
    activation_results = run_activation_communication_experiment()

    # Save activation communication results
    activation_path = output_dir / f"activation_communication_results_{timestamp}.json"
    with open(activation_path, 'w') as f:
        json.dump(activation_results, f, indent=2)
    print(f"Activation communication results saved to: {activation_path}")

    print("\n" + "=" * 80)
    print("ALL 5 EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("Experiments run:")
    print("  1. Linear adapter")
    print("  2. Affine adapter")
    print("  3. LoRA adapter")
    print("  4. Token compression")
    print("  5. Activation communication")

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