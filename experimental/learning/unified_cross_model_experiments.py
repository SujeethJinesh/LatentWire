#!/usr/bin/env python3
"""
Unified cross-model alignment experiments combining Procrustes and learned adapters.
Optimized for 2 GPUs instead of 4.

GPU allocation:
- GPU 0: Linear adapter
- GPU 1: Affine adapter
- CPU: Procrustes alignment (no GPU needed)
- LoRA: Run sequentially after Linear/Affine complete
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import time
from pathlib import Path
from datetime import datetime
import math
import multiprocessing as mp
import os
import sys
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
        config['batch_size'] = 16
        config['num_samples'] = 10000
        config['epochs'] = 10
        config['use_bf16'] = torch.cuda.is_bf16_supported() if platform == 'hpc' else False
        config['use_flash_attention'] = not disable_flash and platform == 'hpc'
        config['grad_accum_steps'] = 4
        if platform == 'hpc':
            print(f"  - Batch size: {config['batch_size']}")
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
MAX_LENGTH = None  # Use full sequences - no artificial truncation!
GRAD_ACCUM_STEPS = PLATFORM_CONFIG['grad_accum_steps']
USE_BF16 = PLATFORM_CONFIG['use_bf16']
USE_FLASH_ATTENTION = PLATFORM_CONFIG['use_flash_attention']

# Contrastive Learning Parameters (NEW from 2025 research)
TEMPERATURE = 0.07  # Optimal for InfoNCE loss
CONTRASTIVE_WEIGHT = 0.3  # Weight for contrastive loss vs generation loss
NUM_NEGATIVES = 127  # Number of negative samples

# Multi-layer alignment (prevents single-point failure)
ALIGNMENT_LAYERS = [8, 16, 24]  # Align multiple layers simultaneously
LAYER_WEIGHTS = [0.2, 0.5, 0.3]  # Weight importance of each layer
LAYER_IDX = 16  # Default for backward compatibility

# Layers to test for Procrustes
LAYERS_TO_TEST = [0, 8, 16, 24, 32]

# Calibration data size
CALIBRATION_SIZE = 100

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
    # For HPC with multiple GPUs, distribute models across GPUs
    if PLATFORM == 'hpc' and torch.cuda.device_count() >= 2:
        # Load models on different GPUs to avoid OOM
        llama_kwargs = {
            'torch_dtype': dtype,
            'device_map': {"": 0},  # Force Llama on GPU 0
        }
        mistral_kwargs = {
            'torch_dtype': dtype,
            'device_map': {"": 1},  # Force Mistral on GPU 1
        }
        print("Distributing models: Llama on GPU 0, Mistral on GPU 1")
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
            for text in calibration_texts[:20]:  # Use subset for memory
                # Tokenize with padding
                llama_inputs = llama_tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                                              padding="max_length", return_tensors="pt").to(device)
                mistral_inputs = mistral_tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                                                  padding="max_length", return_tensors="pt").to(device)

                # Get hidden states
                llama_outputs = llama_model(**llama_inputs, output_hidden_states=True)
                mistral_outputs = mistral_model(**mistral_inputs, output_hidden_states=True)

                # Extract layer hidden states and flatten
                llama_hidden = llama_outputs.hidden_states[layer_idx][0]  # [seq_len, hidden]
                mistral_hidden = mistral_outputs.hidden_states[layer_idx][0]

                # Only use non-padding positions
                llama_mask = llama_inputs["attention_mask"][0].bool()
                mistral_mask = mistral_inputs["attention_mask"][0].bool()

                llama_hidden_all.append(llama_hidden[llama_mask])
                mistral_hidden_all.append(mistral_hidden[mistral_mask])

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
            "llama_to_mistral": {},
            "mistral_to_llama": {}
        }

        print(f"\nTesting generation...")
        for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
            # Baseline: Mistral → Mistral (identity)
            mistral_inputs = mistral_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = mistral_model.generate(**mistral_inputs, max_new_tokens=20, do_sample=False)
            generated = mistral_tokenizer.decode(output[0], skip_special_tokens=True)
            layer_results["mistral_to_mistral"][f"prompt_{prompt_idx}"] = generated

            # Cross-model generation would require more complex injection
            # For now, we'll just store the alignment quality metrics

        results[f"layer_{layer_idx}"] = layer_results

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
            checkpoint = torch.load(checkpoint_path, map_location=device)
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
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Move adapter to device
    adapter = adapter.to(device)
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

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}", file=log_file)

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

            if (batch_idx + 1) % 10 == 0:
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
                source_repr = outputs_a.hidden_states[ALIGNMENT_LAYERS[1]][:, 0, :]
                aligned_repr = adapter(source_repr)
                target_repr = outputs_b.hidden_states[ALIGNMENT_LAYERS[1]][:, 0, :]

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
        print(f"  Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}, CKA: {avg_cka:.4f}", file=log_file)

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

            # Use global device (supports MPS/CUDA/CPU)
            if PLATFORM == 'hpc' and gpu_id is not None:
                device = torch.device(f"cuda:{gpu_id}")
                print(f"GPU assigned: {gpu_id}")
            else:
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

            model_b = AutoModelForCausalLM.from_pretrained(
                MISTRAL_MODEL,
                **model_kwargs
            ).to(device).eval()

            # Enable gradient checkpointing to save memory during training
            model_b.gradient_checkpointing_enable()

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

    # Run learned adapter experiments
    print("\n2. Starting learned adapter experiments...")

    if PLATFORM == 'hpc' and torch.cuda.device_count() >= 3:
        # HPC with 3+ GPUs: Run all adapters in parallel
        print(f"Running all 3 adapters in parallel on {torch.cuda.device_count()} GPUs...")
        processes = []

        p1 = mp.Process(target=run_adapter_experiment, args=("linear", 0))
        p2 = mp.Process(target=run_adapter_experiment, args=("affine", 1))
        p3 = mp.Process(target=run_adapter_experiment, args=("lora", 2))

        p1.start()
        p2.start()
        p3.start()

        processes.extend([p1, p2, p3])

        # Wait for all to complete
        print("Waiting for all adapter experiments to complete...")
        for p in processes:
            p.join()

        print("\n3. All adapter experiments completed in parallel!")

    elif PLATFORM == 'hpc' and torch.cuda.device_count() >= 2:
        # HPC with 2 GPUs: Run 2 in parallel, then LoRA
        print("Running Linear and Affine adapters in parallel on 2 GPUs...")
        processes = []

        p1 = mp.Process(target=run_adapter_experiment, args=("linear", 0))
        p2 = mp.Process(target=run_adapter_experiment, args=("affine", 1))

        p1.start()
        p2.start()

        processes.extend([p1, p2])

        # Wait for completion
        for p in processes:
            p.join()

        # Run LoRA sequentially after others complete
        print("\n3. Starting LoRA experiment (sequential on GPU 0)...")
        run_adapter_experiment("lora", 0)

    else:
        # Mac/CPU/Single GPU: Run sequentially
        if PLATFORM == 'mac':
            print("Running adapters sequentially on MPS device...")
        elif PLATFORM == 'cpu':
            print("Running adapters sequentially on CPU...")
        else:
            print(f"Only {torch.cuda.device_count()} GPU available, running sequentially...")

        # Run all experiments sequentially (None means use default device)
        for adapter_type in ["linear", "affine", "lora"]:
            print(f"\nRunning {adapter_type} adapter...")
            run_adapter_experiment(adapter_type, None)

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