#!/usr/bin/env python3
"""
Unified cross-model alignment experiments combining Procrustes and learned adapters.
Optimized for 4 H100 GPUs.

GPU allocation (with 4 GPUs):
- GPU 0: Linear adapter (parallel)
- GPU 1: Affine adapter (parallel)
- GPU 2: LoRA adapter (parallel)
- GPU 3: Available for overflow/future experiments
- Procrustes: CPU (no GPU needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
        # HPC configuration - batch size optimized for multi-GPU
        num_gpus = torch.cuda.device_count() if platform == 'hpc' else 1
        config['batch_size'] = 10 * num_gpus  # Scale batch size with GPU count (DataParallel splits batches)
        config['num_samples'] = 10000  # Conservative for preemptible cluster
        config['epochs'] = 10
        config['use_bf16'] = torch.cuda.is_bf16_supported() if platform == 'hpc' else False
        config['use_flash_attention'] = not disable_flash and platform == 'hpc'
        config['grad_accum_steps'] = 8  # Gradient accumulation
        if platform == 'hpc':
            print(f"  - Batch size: {config['batch_size']} ({num_gpus} GPUs × 10 per GPU)")
            print(f"  - Effective batch (with grad accum): {config['batch_size'] * config['grad_accum_steps']}")
            print(f"  - Samples: {config['num_samples']}")
            print(f"  - Epochs: {config['epochs']}")
            print(f"  - BF16: {config['use_bf16']}")
            print(f"  - Flash Attention: {config['use_flash_attention']}")

    return device, platform, config

# Get device and config at module level
DEVICE, PLATFORM, PLATFORM_CONFIG = get_device_and_config()

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
    """CKA for measuring similarity between representations."""

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
    def cka_similarity(X, Y):
        """
        Compute CKA similarity between two representation matrices.
        X, Y: [n_samples, n_features]
        Returns: scalar similarity score
        """
        # Compute gram matrices
        K = CKA.linear_kernel(X)
        L = CKA.linear_kernel(Y)

        # Center gram matrices
        K_c = CKA.center_gram(K)
        L_c = CKA.center_gram(L)

        # Compute CKA
        hsic = torch.sum(K_c * L_c)
        var_x = torch.sqrt(torch.sum(K_c * K_c))
        var_y = torch.sqrt(torch.sum(L_c * L_c))

        return hsic / (var_x * var_y + 1e-8)

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
TEMPERATURE = 0.07  # Optimal for InfoNCE loss
CONTRASTIVE_WEIGHT = 0.3  # Weight for contrastive loss vs generation loss
NUM_NEGATIVES = 127  # Number of negative samples

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
    Orthogonal Procrustes alignment between hidden spaces.
    FIXED: Uses torch.norm() for numerical stability.
    """

    def __init__(self):
        self.W = None
        self.source_mean = None
        self.target_mean = None
        self.source_norm = None
        self.target_norm = None

    def fit(self, source, target):
        """
        Fit orthogonal transformation W such that ||source @ W - target||_F is minimized.
        Uses numerically stable Frobenius norm computation.
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

        # Step 3: Compute cross-covariance matrix
        M = source_normalized.T @ target_normalized  # [D, D]

        # Step 4: SVD of M
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)

        # Step 5: Optimal orthogonal transformation
        self.W = U @ Vt

        # Verify orthogonality
        I = self.W @ self.W.T
        ortho_error = torch.norm(I - torch.eye(I.shape[0], device=I.device), 'fro')
        if ortho_error > 1e-3:
            print(f"  WARNING: Orthogonality error = {ortho_error:.6f}")

        return self

    def transform(self, source):
        """Apply the fitted transformation to new data."""
        assert self.W is not None, "Must fit before transform"

        # Apply same centering and normalization as in fit
        source_centered = source - self.source_mean
        source_normalized = source_centered / (self.source_norm + 1e-8)

        # Apply orthogonal transformation
        transformed = source_normalized @ self.W

        # Rescale and recenter to target space
        transformed = transformed * self.target_norm
        transformed = transformed + self.target_mean

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

        # Get the embedding layer
        self.embed_layer = model.get_input_embeddings()

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

    return results

# ============================================================================
# Learned Adapter Training
# ============================================================================

def train_adapter(model_a, model_b, tokenizer_a, tokenizer_b, adapter,
                  device, log_file, num_samples=1000, checkpoint_dir=None):
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
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if PLATFORM == 'hpc' else False,  # Faster GPU transfer
    )

    # Move adapter to device with correct dtype (must match model dtype)
    dtype = torch.bfloat16 if USE_BF16 else torch.float32
    adapter = adapter.to(device, dtype=dtype)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)

    # Add cosine annealing scheduler (as mentioned in comments)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Initialize InfoNCE loss for contrastive learning
    contrastive_loss_fn = InfoNCE(temperature=TEMPERATURE)

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
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()

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

                generation_losses.append(outputs_b.loss * layer_weight)

            # Combine generation losses from multiple layers
            generation_loss = sum(generation_losses)

            # Compute contrastive loss using InfoNCE
            # Use middle layer representations for contrastive learning
            mid_idx = len(ALIGNMENT_LAYERS) // 2
            anchor = all_aligned_reprs[mid_idx][:, 0, :]  # Use [CLS] or first token
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

            # Combine losses
            total_loss = generation_loss + CONTRASTIVE_WEIGHT * contrastive_loss
            total_loss = total_loss / GRAD_ACCUM_STEPS
            total_loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Step the scheduler
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * GRAD_ACCUM_STEPS
            epoch_steps += 1

            # Progress logging every 100 steps (more informative but not spammy)
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / epoch_steps
                progress_pct = 100 * (batch_idx + 1) / len(dataloader)
                elapsed = time.time() - epoch_start_time
                steps_per_sec = (batch_idx + 1) / elapsed
                eta_seconds = (len(dataloader) - batch_idx - 1) / steps_per_sec if steps_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                msg = f"  [{progress_pct:5.1f}%] Step {batch_idx+1:4d}/{len(dataloader)} | Loss: {avg_loss:.4f} | {steps_per_sec:.2f} steps/s | ETA: {eta_minutes:.1f}m"
                print(msg, file=log_file)
                print(msg)  # Also to stdout
                log_file.flush()

            # Quick check-in every 10 steps (just to log file, not stdout)
            elif (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / epoch_steps
                print(f"  Step {batch_idx+1}/{len(dataloader)}: Loss = {avg_loss:.4f}",
                      file=log_file)

        avg_epoch_loss = epoch_loss / epoch_steps

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
        msg += f"  Avg Loss: {avg_epoch_loss:.4f} | CKA Score: {avg_cka:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
        if remaining_epochs > 0:
            msg += f"  ETA for remaining {remaining_epochs} epochs: {eta_total:.1f}m\n"
        msg += f"{'='*80}"

        print(msg, file=log_file)
        print(msg)  # Also to stdout
        log_file.flush()

        # Save checkpoint after each epoch
        if checkpoint_dir:
            checkpoint = {
                'epoch': epoch,
                'adapter_state_dict': adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_metrics': training_metrics,
                'loss': avg_epoch_loss,
            }
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}", file=log_file)

    # Final training summary
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
    log_path = output_dir / f"{adapter_type}_gpu{gpu_id}_{timestamp}.log"

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

            # Device configuration
            use_data_parallel = False
            if PLATFORM == 'hpc' and gpu_id is None and torch.cuda.device_count() > 1:
                # Use all GPUs with DataParallel
                device = torch.device("cuda:0")  # Primary device
                use_data_parallel = True
                print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
                print(f"Primary device: cuda:0")
            elif PLATFORM == 'hpc' and gpu_id is not None:
                # Use specific GPU
                device = torch.device(f"cuda:{gpu_id}")
                print(f"GPU assigned: {gpu_id}")
            else:
                # Use default device (MPS/CPU/single GPU)
                device = DEVICE
                print(f"Device: {device}")

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

            # Wrap with DataParallel if using multiple GPUs
            if use_data_parallel:
                model_a = torch.nn.DataParallel(model_a)
                print(f"Wrapped Llama model with DataParallel")

            model_b = AutoModelForCausalLM.from_pretrained(
                MISTRAL_MODEL,
                **model_kwargs
            ).to(device).eval()

            # Enable gradient checkpointing to save memory during training
            model_b.gradient_checkpointing_enable()

            # Wrap with DataParallel if using multiple GPUs
            if use_data_parallel:
                model_b = torch.nn.DataParallel(model_b)
                print(f"Wrapped Mistral model with DataParallel")

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
                checkpoint_dir=checkpoint_dir
            )

            # Save results
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
    log_path = output_dir / f"token_compression_gpu{gpu_id}_{timestamp}.log"

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

            # Use specified GPU
            if PLATFORM == 'hpc' and gpu_id is not None:
                device = torch.device(f"cuda:{gpu_id}")
                print(f"GPU assigned: {gpu_id}")
            else:
                device = DEVICE
                print(f"Device: {device}")

            # Run token compression experiment
            results = run_token_compression_experiment(
                device=device,
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
    # Device configuration for multi-GPU support
    use_data_parallel = False
    if device is None:
        if PLATFORM == 'hpc' and torch.cuda.device_count() > 1:
            # Use all GPUs with DataParallel
            device = torch.device("cuda:0")  # Primary device
            use_data_parallel = True
            print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
            print(f"Primary device: cuda:0")
        else:
            device = DEVICE
            print(f"Device: {device}")

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

        # Wrap with DataParallel if using multiple GPUs
        if use_data_parallel:
            model = torch.nn.DataParallel(model)
            print(f"Wrapped Llama model with DataParallel")

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
    compressor = TokenInitializedCompressor(
        model=model,
        tokenizer=tokenizer,
        compressed_length=compressed_length,
        hidden_dim=model.config.hidden_size,
        d_z=d_z
    ).to(device, dtype=dtype)

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
        "perplexity": []
    }

    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")

    model.train() if use_lora_all_layers else model.eval()
    compressor.train()

    for epoch in range(epochs):
        epoch_losses = []

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

            loss = outputs.loss
            epoch_losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx//BATCH_SIZE}, Loss: {loss.item():.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        metrics["generation_loss"].append(avg_loss)
        metrics["perplexity"].append(math.exp(min(avg_loss, 10)))  # Cap to prevent overflow

        # Calculate true compression ratio
        # Original: ~200 tokens × 4096 dims = 819,200 parameters
        # Compressed: 64 tokens × 256 dims = 16,384 parameters
        # Compression ratio: 819,200 / 16,384 = 50x
        seq_compression = 200 / compressed_length  # Sequence length compression
        dim_compression = model.config.hidden_size / d_z  # Dimension compression
        total_compression = seq_compression * dim_compression
        metrics["compression_ratio"].append(total_compression)

        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Perplexity: {metrics['perplexity'][-1]:.2f}, Compression: {total_compression:.1f}x")

    # Evaluation
    print("\n" + "=" * 40)
    print("EVALUATION")
    print("=" * 40)
    print(f"Z Vector Shape: [{compressed_length}, {d_z}] = {compressed_length * d_z:,} parameters")
    print(f"Original Shape: [~200, {model.config.hidden_size}] = ~{200 * model.config.hidden_size:,} parameters")
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
        print(f"EXPERIMENT {['linear', 'affine', 'lora'].index(adapter_type) + 1}/4: {adapter_type.upper()} ADAPTER")
        print(f"{'='*80}")
        run_adapter_experiment(adapter_type, gpu_id=None)  # None = use all GPUs

        # Clean GPU memory between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"✓ {adapter_type.upper()} complete, GPU memory cleared")

    # Run token compression experiment (uses all GPUs via DataParallel)
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 4/4: TOKEN COMPRESSION")
    print(f"{'='*80}")
    run_token_compression_wrapper(gpu_id=None)  # None = use all GPUs

    print("\n" + "=" * 80)
    print("ALL 4 EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("Experiments run:")
    print("  1. Linear adapter")
    print("  2. Affine adapter")
    print("  3. LoRA adapter")
    print("  4. Token compression")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)

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

    main()