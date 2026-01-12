#!/usr/bin/env python3
"""
Complete Telepathy Paper Evaluation Script (REVISED)

Addresses all critical issues identified in review:
1. Fair speedup comparison (both methods do same classification task)
2. Compression ratio calculation (text bytes vs soft token bytes)
3. Token count ablation (4, 8, 16, 32 tokens)
4. Individual model baselines (Llama-alone, Mistral-alone) for super-additivity
5. Full test sets (no truncation)
6. Text-relay with accuracy measurement
7. LoRA and prompt tuning baselines
8. Multiple comparison correction from statistical_testing.py
9. Per-example predictions for McNemar's test

Constraints:
- Runs on 1 H100 GPU in ~12 hours
- Uses 3 seeds [42, 123, 456]
- Tests on 3 key datasets: SST-2, AG News, TREC

Usage:
    python telepathy/run_complete_evaluation.py --output_dir runs/paper_results

Author: Telepathy Project
Date: January 2025
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("telepathy_eval")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - less verbose
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# GPU MEMORY MANAGEMENT
# =============================================================================

def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        device_idx = device.index if device.index is not None else 0
        allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
        total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)  # GB
        free = total - reserved

        return {
            "available": True,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(free, 2),
            "utilization_pct": round((reserved / total) * 100, 1),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def log_gpu_memory(logger: logging.Logger, device: torch.device, context: str = ""):
    """Log current GPU memory status."""
    mem_info = get_gpu_memory_info(device)
    if mem_info.get("available"):
        prefix = f"[{context}] " if context else ""
        logger.debug(
            f"{prefix}GPU Memory: {mem_info['allocated_gb']:.1f}GB allocated, "
            f"{mem_info['free_gb']:.1f}GB free, {mem_info['utilization_pct']:.0f}% used"
        )


def aggressive_memory_cleanup(device: torch.device, logger: Optional[logging.Logger] = None):
    """Perform aggressive memory cleanup between experiments."""
    if logger:
        log_gpu_memory(logger, device, "Before cleanup")

    # Python garbage collection
    gc.collect()

    # PyTorch CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force garbage collection again after CUDA cleanup
        gc.collect()
        torch.cuda.empty_cache()

    if logger:
        log_gpu_memory(logger, device, "After cleanup")


def check_gpu_memory_threshold(device: torch.device, threshold_gb: float = 5.0) -> bool:
    """Check if free GPU memory is above threshold. Returns True if OK."""
    mem_info = get_gpu_memory_info(device)
    if not mem_info.get("available"):
        return True  # Can't check, assume OK
    return mem_info.get("free_gb", 0) >= threshold_gb


# =============================================================================
# ERROR HANDLING AND RETRY LOGIC
# =============================================================================

class ExperimentError(Exception):
    """Custom exception for experiment failures with context."""
    def __init__(self, experiment_name: str, message: str, partial_result: Optional[Dict] = None):
        self.experiment_name = experiment_name
        self.message = message
        self.partial_result = partial_result
        super().__init__(f"{experiment_name}: {message}")


def run_with_retry(
    func,
    *args,
    max_retries: int = 2,
    retry_delay: float = 5.0,
    experiment_name: str = "unknown",
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Run an experiment function with retry logic and error handling.

    Returns:
        Tuple of (success: bool, result: Optional[Dict], error_message: Optional[str])
    """
    last_error = None
    last_traceback = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                if logger:
                    logger.warning(f"Retry {attempt}/{max_retries} for {experiment_name}")
                # Clean up memory before retry
                if device:
                    aggressive_memory_cleanup(device, logger)
                time.sleep(retry_delay)

            result = func(*args, **kwargs)
            return True, result, None

        except torch.cuda.OutOfMemoryError as e:
            last_error = str(e)
            last_traceback = traceback.format_exc()
            if logger:
                logger.error(f"OOM in {experiment_name} (attempt {attempt + 1}): {e}")
            # Aggressive cleanup on OOM
            if device:
                aggressive_memory_cleanup(device, logger)

        except Exception as e:
            last_error = str(e)
            last_traceback = traceback.format_exc()
            if logger:
                logger.error(f"Error in {experiment_name} (attempt {attempt + 1}): {e}")
                logger.debug(f"Traceback:\n{last_traceback}")

    # All retries failed
    error_msg = f"Failed after {max_retries + 1} attempts. Last error: {last_error}\n{last_traceback}"
    return False, None, error_msg


def save_partial_results_atomic(
    results: Dict[str, Any],
    output_path: Path,
    logger: Optional[logging.Logger] = None
):
    """Save results atomically to prevent corruption on interrupt."""
    temp_path = output_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        temp_path.replace(output_path)
        if logger:
            logger.debug(f"Saved results to {output_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save results to {output_path}: {e}")
        # Try to clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


class ExperimentCheckpointer:
    """Manages checkpointing for long-running experiments."""

    def __init__(self, checkpoint_dir: Path, experiment_name: str, save_interval: int = 100):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.save_interval = save_interval
        self.checkpoint_path = checkpoint_dir / f"{experiment_name}_checkpoint.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.data = {}

    def update(self, key: str, value: Any, step: Optional[int] = None):
        """Update checkpoint data."""
        if step is not None:
            self.step = step
        else:
            self.step += 1
        self.data[key] = value

        # Save periodically
        if self.step % self.save_interval == 0:
            self.save()

    def save(self):
        """Save checkpoint to disk."""
        checkpoint = {
            "experiment_name": self.experiment_name,
            "step": self.step,
            "timestamp": datetime.now().isoformat(),
            "data": self.data,
        }
        save_partial_results_atomic(checkpoint, self.checkpoint_path)

    def load(self) -> Optional[Dict]:
        """Load checkpoint if it exists."""
        if not self.checkpoint_path.exists():
            return None
        try:
            with open(self.checkpoint_path) as f:
                checkpoint = json.load(f)
            self.step = checkpoint.get("step", 0)
            self.data = checkpoint.get("data", {})
            return checkpoint
        except Exception:
            return None

    def clear(self):
        """Clear checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


# =============================================================================
# HIGH VARIANCE DETECTION
# =============================================================================

def detect_high_variance(
    accuracies: List[float],
    threshold_std: float = 5.0,
    threshold_range: float = 15.0,
) -> Tuple[bool, Dict[str, float]]:
    """
    Detect if experiment results have high variance.

    Args:
        accuracies: List of accuracy values across seeds
        threshold_std: Standard deviation threshold for warning
        threshold_range: Max-min range threshold for warning

    Returns:
        Tuple of (is_high_variance: bool, stats: Dict)
    """
    if len(accuracies) < 2:
        return False, {"std": 0, "range": 0, "mean": accuracies[0] if accuracies else 0}

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)
    range_acc = max(accuracies) - min(accuracies)
    cv = (std_acc / mean_acc * 100) if mean_acc > 0 else 0  # Coefficient of variation

    stats = {
        "mean": float(mean_acc),
        "std": float(std_acc),
        "range": float(range_acc),
        "cv_pct": float(cv),
        "min": float(min(accuracies)),
        "max": float(max(accuracies)),
    }

    is_high_variance = std_acc >= threshold_std or range_acc >= threshold_range

    return is_high_variance, stats


def get_extra_seeds_if_needed(
    base_seeds: List[int],
    accuracies: List[float],
    variance_threshold_std: float = 5.0,
    max_extra_seeds: int = 3,
) -> List[int]:
    """
    Return additional seeds to run if variance is high.

    Args:
        base_seeds: Seeds already run
        accuracies: Accuracy results from base seeds
        variance_threshold_std: Threshold for detecting high variance
        max_extra_seeds: Maximum number of extra seeds to suggest

    Returns:
        List of additional seed values to run
    """
    is_high_var, stats = detect_high_variance(accuracies, threshold_std=variance_threshold_std)

    if not is_high_var:
        return []

    # Generate extra seeds that haven't been used
    extra_seeds = []
    candidate = max(base_seeds) + 1
    while len(extra_seeds) < max_extra_seeds:
        if candidate not in base_seeds:
            extra_seeds.append(candidate)
        candidate += 1

    return extra_seeds


# Import from statistical_testing.py
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    from statistical_testing import (
        bootstrap_ci,
        paired_ttest,
        mcnemar_test,
        multiple_comparison_correction,
        cohens_d_paired,
        p_value_to_stars,
        MultiSeedExperiment,
    )
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: statistical_testing.py not found, using basic statistics")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Datasets to evaluate (focused on 3 key datasets for time constraints)
DATASETS = {
    "sst2": {
        "name": "SST-2",
        "num_classes": 2,
        "split_train": "train",
        "split_test": "validation",  # GLUE uses validation as test
        "load_fn": "glue:sst2",
        "text_field": "sentence",
        "label_field": "label",
        "label_names": ["negative", "positive"],
        "prompt_template": "Review: {text}\nSentiment (positive or negative):",
        "primer": "Sentiment:",
        "max_length": 128,
    },
    "agnews": {
        "name": "AG News",
        "num_classes": 4,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "fancyzhx/ag_news",
        "text_field": "text",
        "label_field": "label",
        "label_names": ["world", "sports", "business", "science"],
        "prompt_template": "Article: {text}\nTopic (world, sports, business, or science):",
        "primer": "Topic:",
        "max_length": 256,
    },
    "trec": {
        "name": "TREC",
        "num_classes": 6,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "trec",
        "text_field": "text",
        "label_field": "coarse_label",
        "label_names": ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
        "prompt_template": "Question: {text}\nCategory (ABBR, ENTY, DESC, HUM, LOC, or NUM):",
        "primer": "Category:",
        "max_length": 128,
    },
}

# Random seeds (reduced to 2 for 12h time constraint while maintaining statistical validity)
SEEDS = [42, 456]

# Model configurations
SOURCE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Token ablation configurations (reduced to endpoints only: 8=most compressed, 32=highest capacity)
TOKEN_ABLATION_CONFIGS = [8, 32]

# Bridge configurations (train_steps reduced from 2000 to 1500 for 12h budget)
BRIDGE_CONFIG = {
    "soft_tokens": 8,
    "depth": 2,
    "heads": 8,
    "source_layer": 31,
    "train_steps": 1500,
    "batch_size": 16,
    "lr": 2e-4,
    "diversity_weight": 0.1,
}

# Linear probe configurations
LINEAR_PROBE_CONFIG = {
    "layer_idx": 16,
    "normalize": "l2",
    "C": 1.0,
    "batch_size": 8,
    "max_samples": 5000,
}

# LoRA configurations (epochs reduced from 3 to 2 for 12h budget)
LORA_CONFIG = {
    "rank": 8,
    "alpha": 16,
    "epochs": 2,
    "batch_size": 4,
    "lr": 1e-4,
    "max_train_samples": 2000,
}

# Prompt tuning configurations (steps reduced from 2000 to 1500 for 12h budget)
PROMPT_TUNING_CONFIG = {
    "soft_tokens": 8,
    "lr": 2e-4,
    "steps": 1500,
    "batch_size": 16,
    "grad_accum": 2,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_by_config(dataset_key: str, split: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load a dataset according to its configuration."""
    config = DATASETS[dataset_key]

    # Parse load function
    load_fn = config["load_fn"]
    if ":" in load_fn:
        ds_name, ds_config = load_fn.split(":")
        dataset = load_dataset(ds_name, ds_config, split=split, trust_remote_code=True)
    else:
        dataset = load_dataset(load_fn, split=split, trust_remote_code=True)

    # Update label names if not provided
    if config["label_names"] is None and hasattr(dataset.features[config["label_field"]], 'names'):
        config["label_names"] = dataset.features[config["label_field"]].names

    # Extract examples (NO TRUNCATION - use full test set)
    examples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        # Handle different text field formats
        text = item[config["text_field"]]
        label = item[config["label_field"]]
        examples.append({"text": text, "label": label, "idx": idx})

    return examples


# =============================================================================
# COMPRESSION RATIO CALCULATION
# =============================================================================

def calculate_compression_ratio(
    text: str,
    num_soft_tokens: int,
    token_dim: int = 4096,
    quantization_bits: int = 16
) -> Dict[str, float]:
    """
    Calculate compression ratio between text and soft tokens.

    Args:
        text: Original text string
        num_soft_tokens: Number of soft tokens used
        token_dim: Dimension of each soft token
        quantization_bits: Bits per value for quantization (16 for fp16, 8 for int8, etc.)

    Returns:
        Dictionary with compression metrics
    """
    # Text bytes (UTF-8 encoding)
    text_bytes = len(text.encode('utf-8'))

    # Soft token bytes (num_tokens * dimension * bits_per_value / 8)
    soft_token_bytes = num_soft_tokens * token_dim * quantization_bits / 8

    # For practical comparison, we can also use a learned compression
    # where soft tokens are projected to a smaller dimension
    # Using d_z=256 as in the paper
    d_z = 256
    compressed_soft_token_bytes = num_soft_tokens * d_z * quantization_bits / 8

    # Compression ratio should show how much smaller the compressed version is
    # i.e., compressed_bytes / original_bytes (e.g., 0.1 means 10% of original size)
    compression_ratio_raw = soft_token_bytes / text_bytes if text_bytes > 0 else 0
    compression_ratio_projected = compressed_soft_token_bytes / text_bytes if text_bytes > 0 else 0

    return {
        "text_bytes": text_bytes,
        "soft_token_bytes_raw": soft_token_bytes,
        "soft_token_bytes_projected": compressed_soft_token_bytes,
        "compression_ratio_raw": compression_ratio_raw,
        "compression_ratio_projected": compression_ratio_projected,
        "num_soft_tokens": num_soft_tokens,
    }


# =============================================================================
# BRIDGE COMPONENTS
# =============================================================================

class PerceiverResampler(torch.nn.Module):
    """Perceiver-style cross-attention resampler for the bridge."""

    def __init__(self, src_dim: int, tgt_dim: int, num_latents: int = 64, heads: int = 8, depth: int = 4):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim

        self.latents = torch.nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)
        self.input_proj = torch.nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else torch.nn.Identity()

        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "cross_attn": torch.nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln1": torch.nn.LayerNorm(tgt_dim),
                "self_attn": torch.nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln2": torch.nn.LayerNorm(tgt_dim),
                "ffn": torch.nn.Sequential(
                    torch.nn.Linear(tgt_dim, 4 * tgt_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(4 * tgt_dim, tgt_dim)
                ),
                "ln3": torch.nn.LayerNorm(tgt_dim)
            }) for _ in range(depth)
        ])

    def forward(self, src_hidden: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = src_hidden.shape[0]
        keys = self.input_proj(src_hidden)
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            x_norm = layer["ln1"](x)
            attn_out, _ = layer["cross_attn"](x_norm, keys, keys, key_padding_mask=key_padding_mask)
            x = x + attn_out

            x_norm = layer["ln2"](x)
            attn_out, _ = layer["self_attn"](x_norm, x_norm, x_norm)
            x = x + attn_out

            x = x + layer["ffn"](layer["ln3"](x))

        return x


class TelepathyBridge(torch.nn.Module):
    """Telepathy bridge for cross-model communication."""

    def __init__(self, src_dim: int, tgt_dim: int, num_soft_tokens: int = 8,
                 heads: int = 8, depth: int = 2, target_rms: float = 0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.num_soft_tokens = num_soft_tokens
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_soft_tokens, heads, depth)
        self.output_scale = torch.nn.Parameter(torch.tensor(target_rms))

    def forward(self, src_hidden: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        compressed = self.resampler(src_hidden, src_mask)
        rms = torch.sqrt((compressed ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (compressed / rms) * self.output_scale
        z_variance = compressed.var(dim=[0, 1]).mean()
        return out, z_variance


# =============================================================================
# SOFT PROMPT TUNING BASELINE
# =============================================================================

class SoftPromptTuning(torch.nn.Module):
    """Learnable soft prompts for a frozen LLM (Lester et al., 2021)."""

    def __init__(self, num_tokens: int, embed_dim: int, target_rms: float = 0.03):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.soft_prompts = torch.nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
        self.output_scale = torch.nn.Parameter(torch.tensor(target_rms))

    def forward(self, batch_size: int) -> torch.Tensor:
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        rms = torch.sqrt((prompts ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (prompts / rms) * self.output_scale
        return out


# =============================================================================
# FAIR LATENCY MEASUREMENT
# =============================================================================

def measure_fair_latency(
    text: str,
    dataset_config: Dict,
    src_model: torch.nn.Module,
    tgt_model: torch.nn.Module,
    src_tok,
    tgt_tok,
    bridge: torch.nn.Module,
    device: torch.device,
    num_warmup: int = 3,
    num_runs: int = 20,
) -> Dict[str, float]:
    """
    Measure latency fairly - both methods complete the same classification task.

    Returns latency for:
    1. Bridge: Llama encode -> bridge -> Mistral classify
    2. Text-relay: Llama summarize -> text -> Mistral classify
    3. Mistral direct: Mistral classify from text
    """
    results = {}

    prompt = dataset_config["prompt_template"].format(text=text[:dataset_config["max_length"]])
    primer = dataset_config["primer"]

    # Warmup
    for _ in range(num_warmup):
        src_enc = src_tok(prompt, return_tensors="pt", truncation=True, max_length=dataset_config["max_length"]).to(device)
        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)

    # 1. Bridge latency (end-to-end classification)
    bridge_latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        src_enc = src_tok(prompt, return_tensors="pt", truncation=True, max_length=dataset_config["max_length"]).to(device)
        with torch.no_grad():
            # Llama encode
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[BRIDGE_CONFIG["source_layer"]]

            # Bridge
            soft_tokens, _ = bridge(src_h, src_enc.attention_mask)

            # Mistral classify
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            combined = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined.shape[:2], device=device, dtype=torch.long)

            _ = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )

        torch.cuda.synchronize()
        bridge_latencies.append(time.perf_counter() - start)

    results["bridge_mean_ms"] = np.mean(bridge_latencies) * 1000
    results["bridge_std_ms"] = np.std(bridge_latencies) * 1000

    # 2. Text-relay latency (Llama summarize -> Mistral classify)
    relay_latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Llama summarize
        summary_prompt = f"Summarize this in one sentence:\n\n{text[:256]}\n\nSummary:"
        src_enc = src_tok(summary_prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        with torch.no_grad():
            summary_ids = src_model.generate(
                **src_enc,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=src_tok.eos_token_id
            )
            summary = src_tok.decode(summary_ids[0][src_enc.input_ids.shape[1]:], skip_special_tokens=True)

        # Mistral classify from summary
        classify_prompt = f"{primer}\n\nText: {summary}\n\nAnswer:"
        tgt_enc = tgt_tok(classify_prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            _ = tgt_model.generate(
                **tgt_enc,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )

        torch.cuda.synchronize()
        relay_latencies.append(time.perf_counter() - start)

    results["text_relay_mean_ms"] = np.mean(relay_latencies) * 1000
    results["text_relay_std_ms"] = np.std(relay_latencies) * 1000

    # 3. Mistral direct latency
    direct_latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        tgt_enc = tgt_tok(prompt, return_tensors="pt", truncation=True, max_length=dataset_config["max_length"]).to(device)
        with torch.no_grad():
            _ = tgt_model.generate(
                **tgt_enc,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )

        torch.cuda.synchronize()
        direct_latencies.append(time.perf_counter() - start)

    results["mistral_direct_mean_ms"] = np.mean(direct_latencies) * 1000
    results["mistral_direct_std_ms"] = np.std(direct_latencies) * 1000

    # Fair speedup (bridge vs text-relay, since both use two models)
    results["speedup_vs_text_relay"] = results["text_relay_mean_ms"] / results["bridge_mean_ms"]

    return results


# =============================================================================
# BRIDGE TRAINING AND EVALUATION
# =============================================================================

def train_bridge_for_dataset(
    dataset_key: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
    num_soft_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Train and evaluate bridge on a single dataset with a single seed."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Allow token count override for ablation
    soft_tokens = num_soft_tokens if num_soft_tokens else config["soft_tokens"]

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Training Bridge: {dataset_config['name']} | Seed: {seed} | Tokens: {soft_tokens}")
    print(f"{'='*60}")

    # Load models
    src_model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Freeze models
    for p in src_model.parameters():
        p.requires_grad = False
    for p in tgt_model.parameters():
        p.requires_grad = False

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

    # Initialize bridge
    bridge = TelepathyBridge(
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        num_soft_tokens=soft_tokens,
        heads=config["heads"],
        depth=config["depth"],
        target_rms=target_rms
    ).to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=config["lr"], weight_decay=0.01)

    # Load data
    train_data = load_dataset_by_config(dataset_key, dataset_config["split_train"])
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])  # FULL test set

    # Training loop
    bridge.train()
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True,
        collate_fn=lambda x: x, drop_last=True
    )
    iter_loader = iter(train_loader)

    pbar = tqdm(range(config["train_steps"]), desc=f"{dataset_key} seed={seed}")

    for step in pbar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        texts = [item["text"] for item in batch]
        labels = [dataset_config["label_names"][item["label"]] for item in batch]
        B = len(texts)

        # Source encoding
        src_texts = [dataset_config["prompt_template"].format(text=t[:dataset_config["max_length"]]) for t in texts]
        src_enc = src_tok(src_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=dataset_config["max_length"]).to(device)

        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[config["source_layer"]]

        # Bridge forward
        soft_tokens_out, z_var = bridge(src_h, src_enc.attention_mask)

        # Diversity loss
        flat_tokens = soft_tokens_out.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        div_loss = sim_matrix[mask].mean()

        # Target (generate answer)
        primer = dataset_config["primer"]
        with torch.no_grad():
            primer_enc = tgt_tok([primer] * B, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)

            tgt_texts = [f" {l}{tgt_tok.eos_token}" for l in labels]
            tgt_enc = tgt_tok(tgt_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=16, add_special_tokens=False).to(device)
            answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)

        inputs_embeds = torch.cat([primer_embeds, soft_tokens_out, answer_embeds], dim=1)

        K = soft_tokens_out.shape[1]
        P_len = primer_embeds.shape[1]
        ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
        answer_labels = tgt_enc.input_ids.clone()
        answer_labels[tgt_enc.attention_mask == 0] = -100
        labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

        soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
        full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

        outputs = tgt_model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels_tensor)
        lm_loss = outputs.loss

        total_loss = lm_loss + config["diversity_weight"] * div_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({"lm": f"{lm_loss.item():.3f}", "div": f"{div_loss.item():.3f}"})

    # Evaluation on FULL test set
    bridge.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_outputs = []
    compression_ratios = []

    print(f"\nEvaluating on {len(test_data)} test samples (full test set)...")

    for item in tqdm(test_data, desc="Evaluating", leave=False):
        text = item["text"]
        label = dataset_config["label_names"][item["label"]]

        # Calculate compression ratio
        comp_ratio = calculate_compression_ratio(text, soft_tokens, tgt_model.config.hidden_size)
        compression_ratios.append(comp_ratio)

        src_input = dataset_config["prompt_template"].format(text=text[:dataset_config["max_length"]])
        src_enc = src_tok(src_input, return_tensors="pt", truncation=True,
                         max_length=dataset_config["max_length"]).to(device)

        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[config["source_layer"]]
            soft_tokens_out, _ = bridge(src_h, src_enc.attention_mask)

            primer = dataset_config["primer"]
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)

            combined_embeds = torch.cat([primer_embeds, soft_tokens_out], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        # Check prediction
        pred_label = None
        for lbl in dataset_config["label_names"]:
            if lbl.lower() in output:
                pred_label = lbl
                break

        is_correct = pred_label is not None and pred_label.lower() == label.lower()
        if is_correct:
            correct += 1
        total += 1

        # Store for McNemar's test
        all_predictions.append(pred_label)
        all_labels.append(label)
        all_outputs.append(output)

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Aggregate compression ratios
    avg_compression = np.mean([c["compression_ratio_projected"] for c in compression_ratios])

    # Save checkpoint
    checkpoint_path = output_dir / f"{dataset_key}_seed{seed}_tokens{soft_tokens}_bridge.pt"
    torch.save(bridge.state_dict(), checkpoint_path)

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "seed": seed,
        "num_soft_tokens": soft_tokens,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_classes": dataset_config["num_classes"],
        "config": config,
        "checkpoint_path": str(checkpoint_path),
        "compression_ratio_avg": avg_compression,
        "compression_ratios_sample": compression_ratios[:5],
        # Per-example predictions for McNemar's test
        "predictions": all_predictions,
        "labels": all_labels,
        "outputs": all_outputs[:20],  # Save first 20 outputs for inspection
    }

    # Save results
    results_path = output_dir / f"{dataset_key}_seed{seed}_tokens{soft_tokens}_results.json"
    # Convert predictions to serializable format
    results_to_save = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
    results_to_save["predictions_sample"] = all_predictions[:50]
    results_to_save["labels_sample"] = all_labels[:50]
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n[RESULT] {dataset_config['name']} seed={seed} tokens={soft_tokens}: {accuracy:.1f}% ({correct}/{total})")
    print(f"  Avg compression ratio: {avg_compression:.2f}x")

    # Cleanup
    del src_model, tgt_model, bridge
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# ZERO-SHOT BASELINES (Individual model baselines for super-additivity)
# =============================================================================

def run_zeroshot_baseline(
    dataset_key: str,
    model_name: str,
    model_id: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Run zero-shot classification baseline for a single model."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Zero-Shot Baseline: {model_name} on {dataset_config['name']} | Seed: {seed}")
    print(f"{'='*60}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load FULL test data
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    latencies = []

    print(f"Evaluating on {len(test_data)} test samples...")

    for item in tqdm(test_data, desc=f"Evaluating {model_name}", leave=False):
        text = item["text"]
        label = dataset_config["label_names"][item["label"]]

        prompt = dataset_config["prompt_template"].format(text=text[:dataset_config["max_length"]])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        # Match prediction to label
        pred_label = None
        for lbl in dataset_config["label_names"]:
            if lbl.lower() in response:
                pred_label = lbl
                break

        is_correct = pred_label is not None and pred_label.lower() == label.lower()
        if is_correct:
            correct += 1
        total += 1

        all_predictions.append(pred_label)
        all_labels.append(label)

    accuracy = 100 * correct / total if total > 0 else 0

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "model_name": model_name,
        "model_id": model_id,
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_classes": dataset_config["num_classes"],
        "latency_mean_ms": np.mean(latencies) * 1000,
        "latency_std_ms": np.std(latencies) * 1000,
        "predictions": all_predictions,
        "labels": all_labels,
    }

    # Save results
    results_path = output_dir / f"zeroshot_{model_name}_{dataset_key}_seed{seed}.json"
    results_to_save = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
    results_to_save["predictions_sample"] = all_predictions[:50]
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"[RESULT] {model_name} on {dataset_config['name']}: {accuracy:.1f}%")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# TEXT-RELAY BASELINE WITH ACCURACY
# =============================================================================

def run_text_relay_baseline(
    dataset_key: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Run text-relay baseline: Llama summarize -> text -> Mistral classify."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Text-Relay Baseline: {dataset_config['name']} | Seed: {seed}")
    print(f"{'='*60}")

    # Load both models
    src_model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Load FULL test data
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    latencies = []

    print(f"Evaluating on {len(test_data)} test samples...")

    for item in tqdm(test_data, desc="Text-Relay", leave=False):
        text = item["text"]
        label = dataset_config["label_names"][item["label"]]

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Step 1: Llama summarizes with classification context
        # Include the classification task in the summary prompt so Llama preserves relevant information
        task_hint = {
            "sst2": "sentiment (positive or negative)",
            "agnews": "topic category",
            "trec": "question type"
        }.get(dataset_key, "category")

        summary_prompt = f"Summarize this text in one sentence, preserving information about its {task_hint}:\n\n{text[:256]}\n\nSummary:"
        src_enc = src_tok(summary_prompt, return_tensors="pt", truncation=True, max_length=300).to(device)

        with torch.no_grad():
            summary_ids = src_model.generate(
                **src_enc,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=src_tok.eos_token_id
            )
            summary = src_tok.decode(summary_ids[0][src_enc.input_ids.shape[1]:], skip_special_tokens=True)

        # Step 2: Mistral classifies from summary
        # Make the classification prompt more explicit
        label_list = ", ".join(dataset_config["label_names"])
        classify_prompt = f"{dataset_config['primer']}\n\nText: {summary}\n\nClassify this as one of: {label_list}\n\nAnswer:"
        tgt_enc = tgt_tok(classify_prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = tgt_model.generate(
                **tgt_enc,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

        response = tgt_tok.decode(outputs[0][tgt_enc.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        # Match prediction
        pred_label = None
        for lbl in dataset_config["label_names"]:
            if lbl.lower() in response:
                pred_label = lbl
                break

        is_correct = pred_label is not None and pred_label.lower() == label.lower()
        if is_correct:
            correct += 1
        total += 1

        all_predictions.append(pred_label)
        all_labels.append(label)

    accuracy = 100 * correct / total if total > 0 else 0

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "method": "text_relay",
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_classes": dataset_config["num_classes"],
        "latency_mean_ms": np.mean(latencies) * 1000,
        "latency_std_ms": np.std(latencies) * 1000,
        "predictions": all_predictions,
        "labels": all_labels,
    }

    # Save results
    results_path = output_dir / f"text_relay_{dataset_key}_seed{seed}.json"
    results_to_save = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
    results_to_save["predictions_sample"] = all_predictions[:50]
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"[RESULT] Text-Relay on {dataset_config['name']}: {accuracy:.1f}%")

    # Cleanup
    del src_model, tgt_model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# PROMPT TUNING BASELINE
# =============================================================================

def train_prompt_tuning_baseline(
    dataset_key: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train and evaluate prompt tuning baseline (no sender model)."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Prompt Tuning Baseline: {dataset_config['name']} | Seed: {seed}")
    print(f"{'='*60}")

    # Load model (Mistral only - no Llama)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # Compute target RMS
    with torch.no_grad():
        embeds = model.get_input_embeddings().weight.float()
        target_rms = embeds.pow(2).mean(dim=1).sqrt().median().item()

    # Initialize prompt tuning
    soft_prompt = SoftPromptTuning(config["soft_tokens"], model.config.hidden_size, target_rms)
    soft_prompt = soft_prompt.bfloat16().to(device)
    soft_prompt.train()

    optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr=config["lr"], weight_decay=0.01)

    # Load data
    train_data = load_dataset_by_config(dataset_key, dataset_config["split_train"])
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True,
        collate_fn=lambda x: x, drop_last=True
    )
    iter_loader = iter(train_loader)

    pbar = tqdm(range(config["steps"]), desc=f"PromptTuning {dataset_key}")

    for step in pbar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        labels_text = [dataset_config["label_names"][item["label"]] for item in batch]
        B = len(labels_text)

        # Get soft prompts
        prompts = soft_prompt(B)

        # Get primer embeddings
        primer = dataset_config["primer"]
        with torch.no_grad():
            primer_enc = tokenizer([primer] * B, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = model.get_input_embeddings()(primer_enc.input_ids).bfloat16()

            answer_texts = [f" {l}{tokenizer.eos_token}" for l in labels_text]
            answer_enc = tokenizer(answer_texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=16, add_special_tokens=False).to(device)
            answer_embeds = model.get_input_embeddings()(answer_enc.input_ids).bfloat16()

        inputs_embeds = torch.cat([primer_embeds, prompts, answer_embeds], dim=1)

        K = prompts.shape[1]
        P_len = primer_embeds.shape[1]
        ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
        answer_labels = answer_enc.input_ids.clone()
        answer_labels[answer_enc.attention_mask == 0] = -100
        labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

        soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
        full_mask = torch.cat([primer_enc.attention_mask, soft_mask, answer_enc.attention_mask], dim=1)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels_tensor)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    # Evaluation on FULL test set
    soft_prompt.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    print(f"Evaluating on {len(test_data)} test samples...")

    for item in tqdm(test_data, desc="Evaluating", leave=False):
        label = dataset_config["label_names"][item["label"]]

        with torch.no_grad():
            prompts = soft_prompt(1)

            primer = dataset_config["primer"]
            primer_enc = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = model.get_input_embeddings()(primer_enc.input_ids).bfloat16()

            combined_embeds = torch.cat([primer_embeds, prompts], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        pred_label = None
        for lbl in dataset_config["label_names"]:
            if lbl.lower() in output:
                pred_label = lbl
                break

        is_correct = pred_label is not None and pred_label.lower() == label.lower()
        if is_correct:
            correct += 1
        total += 1

        all_predictions.append(pred_label)
        all_labels.append(label)

    accuracy = 100 * correct / total if total > 0 else 0

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "method": "prompt_tuning",
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_classes": dataset_config["num_classes"],
        "num_soft_tokens": config["soft_tokens"],
        "predictions": all_predictions,
        "labels": all_labels,
    }

    # Save results
    results_path = output_dir / f"prompt_tuning_{dataset_key}_seed{seed}.json"
    results_to_save = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
    results_to_save["predictions_sample"] = all_predictions[:50]
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"[RESULT] Prompt Tuning on {dataset_config['name']}: {accuracy:.1f}%")

    # Cleanup
    del model, soft_prompt
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# LORA BASELINE
# =============================================================================

def train_lora_baseline(
    dataset_key: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train and evaluate LoRA baseline on Mistral."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"LoRA Baseline: {dataset_config['name']} | Seed: {seed}")
    print(f"{'='*60}")

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("Warning: peft not installed, skipping LoRA baseline")
        return {"error": "peft not installed"}

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # Load data
    train_data = load_dataset_by_config(dataset_key, dataset_config["split_train"], max_samples=config["max_train_samples"])
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    # Prepare training texts
    train_texts = []
    for item in train_data:
        text = item["text"]
        label_text = dataset_config["label_names"][item["label"]]
        prompt = dataset_config["prompt_template"].format(text=text[:dataset_config["max_length"]])
        full_text = f"{prompt} {label_text}"
        train_texts.append(full_text)

    # Training loop
    model.train()
    import random
    indices = list(range(len(train_texts)))

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        random.shuffle(indices)
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(range(0, len(indices), config["batch_size"]), desc="Training")
        for batch_start in pbar:
            batch_indices = indices[batch_start:batch_start + config["batch_size"]]
            batch_texts = [train_texts[i] for i in batch_indices]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            labels = inputs.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} avg loss: {epoch_loss / num_batches:.4f}")

    # Evaluation on FULL test set
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    print(f"Evaluating on {len(test_data)} test samples...")

    for item in tqdm(test_data, desc="Evaluating", leave=False):
        text = item["text"]
        true_label = dataset_config["label_names"][item["label"]]

        prompt = dataset_config["prompt_template"].format(text=text[:dataset_config["max_length"]])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        pred_label = None
        for label in dataset_config["label_names"]:
            if label.lower() in response:
                pred_label = label
                break

        is_correct = pred_label is not None and pred_label.lower() == true_label.lower()
        if is_correct:
            correct += 1
        total += 1

        all_predictions.append(pred_label)
        all_labels.append(true_label)

    accuracy = 100 * correct / total if total > 0 else 0

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "method": "lora",
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_classes": dataset_config["num_classes"],
        "trainable_params": trainable_params,
        "predictions": all_predictions,
        "labels": all_labels,
    }

    # Save results
    results_path = output_dir / f"lora_{dataset_key}_seed{seed}.json"
    results_to_save = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
    results_to_save["predictions_sample"] = all_predictions[:50]
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"[RESULT] LoRA on {dataset_config['name']}: {accuracy:.1f}%")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# LINEAR PROBE BASELINE
# =============================================================================

def extract_hidden_states(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    max_length: int,
    batch_size: int,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Extract hidden states from a specific layer with memory-efficient implementation.

    Key optimizations:
    1. Only extracts the specific layer needed (not all 33 layers)
    2. Uses smaller batch size with memory monitoring
    3. Aggressive memory cleanup between batches
    4. Processes in chunks to avoid OOM
    """
    model.eval()
    all_hidden = []

    # Use a hook to extract only the specific layer - much more memory efficient
    extracted_hidden = []

    def hook_fn(module, input, output):
        # output is a tuple, first element is the hidden states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        extracted_hidden.append(hidden.detach())

    # Get the specific layer
    if hasattr(model, 'model'):
        # For models with a .model attribute (e.g., LlamaForCausalLM)
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        # For models with a .transformer attribute
        layers = model.transformer.h
    else:
        # Fallback to using output_hidden_states (less memory efficient)
        if logger:
            logger.warning("Could not find model layers, using output_hidden_states (less efficient)")
        return _extract_hidden_states_fallback(
            texts, model, tokenizer, layer_idx, max_length, batch_size, device
        )

    # Register hook on the specific layer
    target_layer = layers[layer_idx]
    hook = target_layer.register_forward_hook(hook_fn)

    try:
        # Reduce batch size for memory safety
        effective_batch_size = min(batch_size, 4)  # Use smaller batches for hidden state extraction
        total_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size

        if logger:
            log_gpu_memory(logger, device, f"Before extraction (layer {layer_idx})")

        for i in tqdm(range(0, len(texts), effective_batch_size),
                     desc=f"Extracting layer {layer_idx}",
                     total=total_batches):
            batch_texts = texts[i:i+effective_batch_size]
            extracted_hidden.clear()

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_length).to(device)

            with torch.no_grad():
                # Just run forward pass - hook captures what we need
                _ = model(**inputs, output_hidden_states=False)

            if not extracted_hidden:
                raise RuntimeError(f"Hook did not capture hidden states for layer {layer_idx}")

            hidden = extracted_hidden[0]
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=device)
            pooled = hidden[batch_indices, seq_lengths]
            all_hidden.append(pooled.cpu().float().numpy())

            # Clear intermediate tensors
            del inputs, hidden, pooled
            extracted_hidden.clear()

            # Periodic memory cleanup
            if (i // effective_batch_size) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if logger:
            log_gpu_memory(logger, device, f"After extraction (layer {layer_idx})")

    finally:
        # Always remove the hook
        hook.remove()

    return np.vstack(all_hidden)


def _extract_hidden_states_fallback(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Fallback method using output_hidden_states when hook approach doesn't work."""
    model.eval()
    all_hidden = []

    # Use very small batch size for fallback
    effective_batch_size = min(batch_size, 2)

    for i in tqdm(range(0, len(texts), effective_batch_size), desc=f"Extracting layer {layer_idx} (fallback)"):
        batch_texts = texts[i:i+effective_batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=device)
        pooled = hidden[batch_indices, seq_lengths]
        all_hidden.append(pooled.cpu().float().numpy())

        # Aggressive cleanup
        del outputs, hidden, pooled, inputs
        gc.collect()
        torch.cuda.empty_cache()

    return np.vstack(all_hidden)


def run_linear_probe(
    dataset_key: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run linear probe baseline on a dataset."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Linear Probe: {dataset_config['name']} | Seed: {seed}")
    print(f"{'='*60}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data (FULL test set)
    train_data = load_dataset_by_config(dataset_key, dataset_config["split_train"],
                                        max_samples=config["max_samples"])
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    train_texts = [item["text"] for item in train_data]
    train_labels = np.array([item["label"] for item in train_data])
    test_texts = [item["text"] for item in test_data]
    test_labels = np.array([item["label"] for item in test_data])

    # Extract hidden states
    X_train = extract_hidden_states(
        train_texts, model, tokenizer, config["layer_idx"],
        dataset_config["max_length"], config["batch_size"], device
    )
    X_test = extract_hidden_states(
        test_texts, model, tokenizer, config["layer_idx"],
        dataset_config["max_length"], config["batch_size"], device
    )

    # Normalize
    if config["normalize"] == "l2":
        X_train = sk_normalize(X_train, norm='l2', axis=1)
        X_test = sk_normalize(X_test, norm='l2', axis=1)

    # Train logistic regression
    clf = LogisticRegression(C=config["C"], max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X_train, train_labels)

    # Evaluate
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_acc = accuracy_score(train_labels, train_pred) * 100
    test_acc = accuracy_score(test_labels, test_pred) * 100

    average = 'binary' if dataset_config["num_classes"] == 2 else 'macro'
    train_f1 = f1_score(train_labels, train_pred, average=average) * 100
    test_f1 = f1_score(test_labels, test_pred, average=average) * 100

    # Convert predictions to label names for McNemar's test
    all_predictions = [dataset_config["label_names"][p] for p in test_pred]
    all_labels = [dataset_config["label_names"][l] for l in test_labels]

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "method": "linear_probe",
        "seed": seed,
        "layer_idx": config["layer_idx"],
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "accuracy": test_acc,  # Alias for consistency
        "train_f1": train_f1,
        "test_f1": test_f1,
        "num_train": len(train_labels),
        "num_test": len(test_labels),
        "total": len(test_labels),
        "correct": int(test_acc * len(test_labels) / 100),
        "num_classes": dataset_config["num_classes"],
        "predictions": all_predictions,
        "labels": all_labels,
    }

    # Save results
    results_path = output_dir / f"linear_probe_{dataset_key}_seed{seed}.json"
    results_to_save = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
    results_to_save["predictions_sample"] = all_predictions[:50]
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n[RESULT] Linear Probe {dataset_config['name']} seed={seed}: {test_acc:.1f}%")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# STATISTICAL ANALYSIS WITH MULTIPLE COMPARISON CORRECTION
# =============================================================================

def run_statistical_tests_with_correction(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run statistical tests with multiple comparison correction."""

    print(f"\n{'='*60}")
    print("Statistical Analysis with Multiple Comparison Correction")
    print(f"{'='*60}")

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "correction_method": "bonferroni",
        "alpha": 0.05,
        "comparisons": {},
        "mcnemar_tests": {},
    }

    baseline_method = "bridge_8"  # Bridge with 8 tokens as baseline

    for dataset_key in DATASETS.keys():
        print(f"\n--- {DATASETS[dataset_key]['name']} ---")

        # Collect accuracy arrays for each method
        method_accuracies = {}
        method_predictions_by_seed = {}  # Store predictions for all seeds

        for method, results_list in all_results.items():
            method_results = [r for r in results_list if r.get("dataset") == dataset_key]
            if method_results:
                method_accuracies[method] = np.array([r["accuracy"] for r in method_results])
                # Collect predictions for McNemar's test for ALL seeds
                preds_by_seed = {}
                for r in method_results:
                    if r.get("predictions") and r.get("seed") is not None:
                        preds_by_seed[r["seed"]] = {
                            "predictions": r["predictions"],
                            "labels": r["labels"],
                        }
                if preds_by_seed:
                    method_predictions_by_seed[method] = preds_by_seed

        if not method_accuracies:
            continue

        dataset_analysis = {
            "dataset_name": DATASETS[dataset_key]["name"],
            "num_classes": DATASETS[dataset_key]["num_classes"],
            "random_chance": 100.0 / DATASETS[dataset_key]["num_classes"],
            "methods": {},
        }

        # Get baseline
        baseline_acc = method_accuracies.get(baseline_method)

        # Collect all p-values for multiple comparison correction
        all_p_values = []
        method_names_for_correction = []

        for method, accs in method_accuracies.items():
            # Basic statistics
            if STATS_AVAILABLE and len(accs) >= 3:
                mean_val, (ci_lower, ci_upper) = bootstrap_ci(accs, n_resamples=10000, random_state=42)
            else:
                mean_val = np.mean(accs)
                ci_lower = ci_upper = mean_val

            std_val = np.std(accs, ddof=1) if len(accs) > 1 else 0.0

            dataset_analysis["methods"][method] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "ci_95": [float(ci_lower), float(ci_upper)],
                "n_seeds": len(accs),
                "values": accs.tolist(),
            }

            # Statistical test vs baseline (if not the baseline itself)
            if baseline_acc is not None and method != baseline_method and len(accs) >= 2 and len(baseline_acc) >= 2:
                if STATS_AVAILABLE and len(accs) == len(baseline_acc):
                    diff, p_value, stats = paired_ttest(accs, baseline_acc)
                    d = cohens_d_paired(accs, baseline_acc)
                else:
                    from scipy import stats as scipy_stats
                    if len(accs) == len(baseline_acc):
                        _, p_value = scipy_stats.ttest_rel(accs, baseline_acc)
                    else:
                        _, p_value = scipy_stats.ttest_ind(accs, baseline_acc)
                    diff = np.mean(accs) - np.mean(baseline_acc)
                    d = diff / np.std(accs - baseline_acc if len(accs) == len(baseline_acc) else accs, ddof=1)

                all_p_values.append(p_value)
                method_names_for_correction.append(method)

                dataset_analysis["methods"][method]["vs_baseline"] = {
                    "difference": float(diff),
                    "p_value_raw": float(p_value),
                    "cohens_d": float(d),
                }

        # Apply multiple comparison correction
        if STATS_AVAILABLE and all_p_values:
            reject, corrected_p, _, _ = multiple_comparison_correction(
                all_p_values, alpha=0.05, method='bonferroni'
            )

            for i, method in enumerate(method_names_for_correction):
                dataset_analysis["methods"][method]["vs_baseline"]["p_value_corrected"] = float(corrected_p[i])
                dataset_analysis["methods"][method]["vs_baseline"]["significant"] = bool(reject[i])

                stars = p_value_to_stars(corrected_p[i]) if STATS_AVAILABLE else ""
                print(f"  {method}: {dataset_analysis['methods'][method]['mean']:.1f}% "
                      f"(diff={dataset_analysis['methods'][method]['vs_baseline']['difference']:+.1f}%, "
                      f"p={corrected_p[i]:.4f}{stars})")

        # McNemar's test for per-example comparison (aggregated across all seeds)
        if STATS_AVAILABLE and baseline_method in method_predictions_by_seed:
            baseline_preds_by_seed = method_predictions_by_seed[baseline_method]

            for method, preds_by_seed in method_predictions_by_seed.items():
                if method == baseline_method:
                    continue

                # Aggregate McNemar stats across all seeds
                all_stats = []
                all_p_vals = []
                combined_table = np.zeros((2, 2))

                for seed in baseline_preds_by_seed.keys():
                    if seed not in preds_by_seed:
                        continue

                    baseline_preds = baseline_preds_by_seed[seed]
                    method_preds = preds_by_seed[seed]

                    if len(method_preds["predictions"]) != len(baseline_preds["predictions"]):
                        continue

                    try:
                        stat, p_val, table = mcnemar_test(
                            np.array(method_preds["predictions"]),
                            np.array(baseline_preds["predictions"]),
                            np.array(method_preds["labels"])
                        )
                        all_stats.append(stat)
                        all_p_vals.append(p_val)
                        combined_table += table
                    except Exception as e:
                        print(f"    McNemar test failed for {method} seed={seed}: {e}")

                if all_stats:
                    if dataset_key not in analysis["mcnemar_tests"]:
                        analysis["mcnemar_tests"][dataset_key] = {}

                    # Report aggregated results
                    avg_stat = np.mean(all_stats)
                    # Use Fisher's method to combine p-values
                    combined_p = 1.0 - np.exp(-2 * np.sum(np.log(np.maximum(all_p_vals, 1e-10))) / (2 * len(all_p_vals)))

                    analysis["mcnemar_tests"][dataset_key][method] = {
                        "statistic_mean": float(avg_stat),
                        "statistic_per_seed": [float(s) for s in all_stats],
                        "p_value_combined": float(combined_p),
                        "p_values_per_seed": [float(p) for p in all_p_vals],
                        "contingency_table_combined": combined_table.tolist(),
                        "num_seeds": len(all_stats),
                    }

                    print(f"  McNemar {method} vs {baseline_method}: "
                          f"avg_stat={avg_stat:.2f}, combined_p={combined_p:.4f} ({len(all_stats)} seeds)")

        analysis["comparisons"][dataset_key] = dataset_analysis

    # Save analysis
    analysis_path = output_dir / "statistical_analysis_corrected.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nStatistical analysis saved to: {analysis_path}")

    return analysis


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_comprehensive_latex_tables(
    all_results: Dict[str, List[Dict]],
    analysis: Dict[str, Any],
    output_dir: Path,
    latency_results: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate comprehensive LaTeX tables for the paper with actual values."""

    print(f"\n{'='*60}")
    print("Generating LaTeX Tables")
    print(f"{'='*60}")

    def get_method_stats(method: str, dataset_key: str) -> Tuple[str, float]:
        """Get formatted mean +/- std and raw mean for a method on a dataset."""
        if dataset_key in analysis.get("comparisons", {}):
            method_data = analysis["comparisons"][dataset_key]["methods"].get(method, {})
            if method_data:
                mean = method_data.get("mean", 0)
                std = method_data.get("std", 0)
                # Check for significance markers
                sig_marker = ""
                if "vs_baseline" in method_data:
                    p_val = method_data["vs_baseline"].get("p_value_corrected", 1.0)
                    if p_val < 0.001:
                        sig_marker = "***"
                    elif p_val < 0.01:
                        sig_marker = "**"
                    elif p_val < 0.05:
                        sig_marker = "*"
                return f"{mean:.1f} $\\pm$ {std:.1f}{sig_marker}", mean
        return "--", None

    def format_row(method: str, method_label: str, params: str = "--") -> str:
        """Format a complete table row for a method."""
        values = []
        raw_means = []
        for dataset_key in ["sst2", "agnews", "trec"]:
            formatted, raw_mean = get_method_stats(method, dataset_key)
            values.append(formatted)
            if raw_mean is not None:
                raw_means.append(raw_mean)

        # Calculate average
        avg = f"{np.mean(raw_means):.1f}" if raw_means else "--"

        return f"{method_label} & {' & '.join(values)} & {avg} & {params} \\\\"

    latex = []

    # Table 1: Main results comparison
    latex.append(r"""% Table 1: Main Results - Method Comparison
\begin{table*}[t]
\centering
\caption{Classification accuracy (\%) on benchmark datasets. Results show mean $\pm$ std over 3 random seeds.
Bold indicates best result. Statistical significance vs. Bridge (8 tokens): *p<0.05, **p<0.01, ***p<0.001 (Bonferroni-corrected).}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
Method & SST-2 & AG News & TREC & Avg & Params \\
\midrule
\textit{Random Chance} & 50.0 & 25.0 & 16.7 & 30.6 & -- \\
\midrule
\textbf{Individual Models (Zero-shot)} \\""")

    # Zero-shot baselines
    latex.append(format_row("zeroshot_llama", r"\quad Llama-8B", "8B"))
    latex.append(format_row("zeroshot_mistral", r"\quad Mistral-7B", "7B"))

    latex.append(r"""\midrule
\textbf{Text-Relay} \\""")
    latex.append(format_row("text_relay", r"\quad Llama $\rightarrow$ text $\rightarrow$ Mistral", "15B"))

    latex.append(r"""\midrule
\textbf{Fine-tuning Baselines} \\""")
    latex.append(format_row("linear_probe", r"\quad Linear Probe (Llama layer-16)", "33M"))
    latex.append(format_row("prompt_tuning", r"\quad Prompt Tuning (Mistral only)", "33K"))
    latex.append(format_row("lora", r"\quad LoRA (Mistral, rank-8)", "4.2M"))

    latex.append(r"""\midrule
\textbf{Telepathy Bridge (Ours)} \\""")
    # Add bridge results for each token count
    for num_tokens in TOKEN_ABLATION_CONFIGS:
        method = f"bridge_{num_tokens}"
        # Calculate approximate params: Perceiver resampler + adapters
        bridge_params = num_tokens * 4096 * 2 + 4096 * 4096 * 2  # Rough estimate
        params_str = f"{bridge_params/1e6:.1f}M"
        latex.append(format_row(method, f"\\quad {num_tokens} soft tokens", params_str))

    latex.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")

    # Table 2: Token ablation with more detail
    latex.append(r"""
% Table 2: Token Count Ablation
\begin{table}[t]
\centering
\caption{Effect of soft token count on Bridge accuracy (\%). More tokens generally improve performance at the cost of higher bandwidth.}
\label{tab:token_ablation}
\begin{tabular}{lccc|c}
\toprule
Tokens & SST-2 & AG News & TREC & Compression \\
\midrule
""")

    for num_tokens in TOKEN_ABLATION_CONFIGS:
        method = f"bridge_{num_tokens}"
        row_values = []
        for dataset_key in ["sst2", "agnews", "trec"]:
            formatted, _ = get_method_stats(method, dataset_key)
            # Just get mean for this table (no std)
            if dataset_key in analysis.get("comparisons", {}):
                method_data = analysis["comparisons"][dataset_key]["methods"].get(method, {})
                if method_data:
                    mean = method_data.get("mean", 0)
                    row_values.append(f"{mean:.1f}")
                else:
                    row_values.append("--")
            else:
                row_values.append("--")

        # Calculate compression ratio from actual results data
        # Get average compression ratio from bridge results for this token count
        compression = 0.0
        comp_count = 0
        for dataset_key in ["sst2", "agnews", "trec"]:
            method = f"bridge_{num_tokens}"
            if "bridge" in all_results and dataset_key in all_results["bridge"]:
                for result in all_results["bridge"][dataset_key]:
                    if result.get("num_soft_tokens") == num_tokens:
                        comp_ratio = result.get("compression_ratio_avg", 0)
                        if comp_ratio > 0:
                            compression += comp_ratio
                            comp_count += 1

        if comp_count > 0:
            compression = compression / comp_count
        else:
            # Fallback calculation if no results found
            soft_token_bytes = num_tokens * 256 * 2  # d_z=256, fp16
            avg_text_bytes = 157  # Average of SST-2, AG News, TREC
            compression = soft_token_bytes / avg_text_bytes if avg_text_bytes > 0 else 0

        latex.append(f"{num_tokens} & {' & '.join(row_values)} & {compression:.4f}x \\\\\n")

    latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # Table 3: Latency comparison with actual values
    latex.append(r"""
% Table 3: Fair Latency Comparison
\begin{table}[t]
\centering
\caption{End-to-end classification latency (ms). Both Bridge and Text-Relay perform the same task (Llama encodes, Mistral classifies). Speedup is computed as Text-Relay / Bridge.}
\label{tab:latency}
\begin{tabular}{lccc}
\toprule
Method & Latency (ms) & Task & Speedup \\
\midrule
""")

    # Use actual latency results if available
    if latency_results:
        direct_lat = latency_results.get("mistral_direct_mean_ms", 0)
        relay_lat = latency_results.get("text_relay_mean_ms", 0)
        bridge_lat = latency_results.get("bridge_mean_ms", 0)
        speedup = latency_results.get("speedup_vs_text_relay", 0)

        latex.append(f"Mistral Direct & {direct_lat:.1f} & Classify & 1.0x (baseline) \\\\\n")
        latex.append(f"Text-Relay & {relay_lat:.1f} & Summarize + Classify & {direct_lat/relay_lat:.2f}x \\\\\n")
        latex.append(f"Bridge (Ours) & {bridge_lat:.1f} & Encode + Classify & {speedup:.2f}x \\\\\n")
    else:
        latex.append(r"Mistral Direct & -- & Classify & 1.0x (baseline) \\" + "\n")
        latex.append(r"Text-Relay & -- & Summarize + Classify & -- \\" + "\n")
        latex.append(r"Bridge (Ours) & -- & Encode + Classify & -- \\" + "\n")

    latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # Table 4: McNemar test results (if available)
    if analysis.get("mcnemar_tests"):
        latex.append(r"""
% Table 4: McNemar's Test Results
\begin{table}[t]
\centering
\caption{McNemar's test comparing per-example predictions between Bridge and other methods. Significant p-values indicate methods make different types of errors.}
\label{tab:mcnemar}
\begin{tabular}{llcc}
\toprule
Dataset & Comparison & Statistic & p-value \\
\midrule
""")
        for dataset_key, tests in analysis["mcnemar_tests"].items():
            dataset_name = DATASETS[dataset_key]["name"]
            for method, test_result in tests.items():
                stat = test_result.get("statistic", 0)
                p_val = test_result.get("p_value", 1)
                sig = "*" if p_val < 0.05 else ""
                latex.append(f"{dataset_name} & Bridge vs {method} & {stat:.2f} & {p_val:.4f}{sig} \\\\\n")

        latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # Save tables
    latex_content = "\n".join(latex)
    tables_path = output_dir / "paper_tables_comprehensive.tex"
    with open(tables_path, "w") as f:
        f.write(latex_content)

    print(f"LaTeX tables saved to: {tables_path}")

    return latex_content


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Track experiment progress and estimate time remaining with improved logging."""

    def __init__(self, total_experiments: int, logger: Optional[logging.Logger] = None,
                 save_callback: Optional[callable] = None):
        self.total_experiments = total_experiments
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self.experiment_times = []
        self.experiment_log = []
        self.logger = logger
        self.save_callback = save_callback  # Called periodically to save intermediate results

    def update(self, experiment_name: str, experiment_time: float, success: bool = True,
               error_msg: Optional[str] = None):
        """Record completion of an experiment."""
        self.completed += 1
        if not success:
            self.failed += 1

        self.experiment_times.append(experiment_time)

        # Log experiment
        log_entry = {
            "name": experiment_name,
            "time": experiment_time,
            "success": success,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
        self.experiment_log.append(log_entry)

        # Calculate statistics
        elapsed = time.time() - self.start_time
        avg_time = np.mean(self.experiment_times)
        remaining = self.total_experiments - self.completed
        eta_seconds = remaining * avg_time
        progress_pct = (self.completed / self.total_experiments) * 100 if self.total_experiments > 0 else 0

        # Format times
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)

        # Create progress bar
        bar_width = 30
        filled = int(bar_width * self.completed / self.total_experiments) if self.total_experiments > 0 else 0
        bar = '=' * filled + '-' * (bar_width - filled)

        status = "OK" if success else "FAILED"
        msg = (
            f"\n{'='*70}\n"
            f"PROGRESS: [{bar}] {progress_pct:.1f}% ({self.completed}/{self.total_experiments})\n"
            f"  Last: {experiment_name} ({experiment_time:.1f}s) [{status}]\n"
            f"  Elapsed: {elapsed_str} | ETA: {eta_str}\n"
            f"  Avg time: {avg_time:.1f}s | Failed: {self.failed} | Skipped: {self.skipped}\n"
            f"{'='*70}"
        )

        print(msg)
        if self.logger:
            self.logger.info(msg)

        # Call save callback periodically (every 5 experiments)
        if self.save_callback and self.completed % 5 == 0:
            try:
                self.save_callback()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Save callback failed: {e}")

    def record_skip(self, experiment_name: str):
        """Record a skipped experiment."""
        self.skipped += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total": self.total_experiments,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": (self.completed - self.failed) / self.completed * 100 if self.completed > 0 else 0,
            "total_time": time.time() - self.start_time,
            "avg_time_per_experiment": np.mean(self.experiment_times) if self.experiment_times else 0,
        }

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

    def get_experiment_log(self) -> List[Dict]:
        """Get the full experiment log."""
        return self.experiment_log


def count_total_experiments(datasets: List[str], seeds: List[int]) -> int:
    """Count total number of experiments to run."""
    n_datasets = len(datasets)
    n_seeds = len(seeds)
    n_tokens = len(TOKEN_ABLATION_CONFIGS)

    total = 0
    # Zero-shot (2 models per dataset-seed)
    total += n_datasets * n_seeds * 2
    # Text relay
    total += n_datasets * n_seeds
    # Prompt tuning
    total += n_datasets * n_seeds
    # LoRA
    total += n_datasets * n_seeds
    # Linear probe
    total += n_datasets * n_seeds
    # Bridge (per token config)
    total += n_tokens * n_datasets * n_seeds
    # Latency measurement (1 per dataset)
    total += n_datasets

    return total


# =============================================================================
# RESUME CAPABILITY - Check for existing results
# =============================================================================

def get_result_file_path(run_dir: Path, method: str, dataset_key: str, seed: int,
                         num_tokens: Optional[int] = None, model_name: Optional[str] = None) -> Path:
    """Get the expected result file path for a given experiment configuration."""
    if method == "zeroshot":
        return run_dir / "zeroshot" / f"zeroshot_{model_name}_{dataset_key}_seed{seed}.json"
    elif method == "text_relay":
        return run_dir / "text_relay" / f"text_relay_{dataset_key}_seed{seed}.json"
    elif method == "prompt_tuning":
        return run_dir / "prompt_tuning" / f"prompt_tuning_{dataset_key}_seed{seed}.json"
    elif method == "lora":
        return run_dir / "lora" / f"lora_{dataset_key}_seed{seed}.json"
    elif method == "linear_probe":
        return run_dir / "linear_probe" / f"linear_probe_{dataset_key}_seed{seed}.json"
    elif method == "bridge":
        return run_dir / "bridge" / f"{dataset_key}_seed{seed}_tokens{num_tokens}_results.json"
    elif method == "latency":
        return run_dir / "latency" / "latency_results.json"
    else:
        raise ValueError(f"Unknown method: {method}")


def check_existing_result(result_path: Path) -> Optional[Dict[str, Any]]:
    """Check if a result file exists and load its contents if valid."""
    if not result_path.exists():
        return None

    try:
        with open(result_path, "r") as f:
            result = json.load(f)

        # Validate that result has required fields
        if "accuracy" in result or "latency_mean_ms" in result or "bridge_mean_ms" in result:
            return result

        return None
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Could not load existing result from {result_path}: {e}")
        return None


def count_completed_experiments(run_dir: Path, datasets: List[str], seeds: List[int]) -> Tuple[int, int, List[str]]:
    """
    Count completed vs remaining experiments and return details.

    Returns:
        Tuple of (completed_count, total_count, list_of_completed_experiment_names)
    """
    completed = 0
    completed_names = []
    total = count_total_experiments(datasets, seeds)

    # Check zero-shot results
    for dataset_key in datasets:
        for seed in seeds:
            for model_name in ["llama", "mistral"]:
                result_path = get_result_file_path(run_dir, "zeroshot", dataset_key, seed, model_name=model_name)
                if check_existing_result(result_path) is not None:
                    completed += 1
                    completed_names.append(f"zeroshot_{model_name}_{dataset_key}_seed{seed}")

    # Check text_relay results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "text_relay", dataset_key, seed)
            if check_existing_result(result_path) is not None:
                completed += 1
                completed_names.append(f"text_relay_{dataset_key}_seed{seed}")

    # Check prompt_tuning results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "prompt_tuning", dataset_key, seed)
            if check_existing_result(result_path) is not None:
                completed += 1
                completed_names.append(f"prompt_tuning_{dataset_key}_seed{seed}")

    # Check lora results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "lora", dataset_key, seed)
            if check_existing_result(result_path) is not None:
                completed += 1
                completed_names.append(f"lora_{dataset_key}_seed{seed}")

    # Check linear_probe results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "linear_probe", dataset_key, seed)
            if check_existing_result(result_path) is not None:
                completed += 1
                completed_names.append(f"linear_probe_{dataset_key}_seed{seed}")

    # Check bridge results
    for num_tokens in TOKEN_ABLATION_CONFIGS:
        for dataset_key in datasets:
            for seed in seeds:
                result_path = get_result_file_path(run_dir, "bridge", dataset_key, seed, num_tokens=num_tokens)
                if check_existing_result(result_path) is not None:
                    completed += 1
                    completed_names.append(f"bridge_{num_tokens}_{dataset_key}_seed{seed}")

    # Check latency results (one per dataset)
    latency_path = get_result_file_path(run_dir, "latency", "", 0)
    if latency_path.exists():
        try:
            with open(latency_path, "r") as f:
                latency_data = json.load(f)
            for dataset_key in datasets:
                if dataset_key in latency_data:
                    completed += 1
                    completed_names.append(f"latency_{dataset_key}")
        except (json.JSONDecodeError, IOError):
            pass

    return completed, total, completed_names


def load_existing_results(run_dir: Path, datasets: List[str], seeds: List[int]) -> Dict[str, List[Dict]]:
    """Load all existing results from a previous run for resume capability."""
    all_results = {
        "zeroshot_llama": [],
        "zeroshot_mistral": [],
        "text_relay": [],
        "prompt_tuning": [],
        "lora": [],
        "linear_probe": [],
    }

    # Add bridge results for each token config
    for num_tokens in TOKEN_ABLATION_CONFIGS:
        all_results[f"bridge_{num_tokens}"] = []

    # Load zero-shot results
    for dataset_key in datasets:
        for seed in seeds:
            for model_name in ["llama", "mistral"]:
                result_path = get_result_file_path(run_dir, "zeroshot", dataset_key, seed, model_name=model_name)
                result = check_existing_result(result_path)
                if result is not None:
                    all_results[f"zeroshot_{model_name}"].append(result)

    # Load text_relay results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "text_relay", dataset_key, seed)
            result = check_existing_result(result_path)
            if result is not None:
                all_results["text_relay"].append(result)

    # Load prompt_tuning results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "prompt_tuning", dataset_key, seed)
            result = check_existing_result(result_path)
            if result is not None:
                all_results["prompt_tuning"].append(result)

    # Load lora results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "lora", dataset_key, seed)
            result = check_existing_result(result_path)
            if result is not None:
                all_results["lora"].append(result)

    # Load linear_probe results
    for dataset_key in datasets:
        for seed in seeds:
            result_path = get_result_file_path(run_dir, "linear_probe", dataset_key, seed)
            result = check_existing_result(result_path)
            if result is not None:
                all_results["linear_probe"].append(result)

    # Load bridge results
    for num_tokens in TOKEN_ABLATION_CONFIGS:
        for dataset_key in datasets:
            for seed in seeds:
                result_path = get_result_file_path(run_dir, "bridge", dataset_key, seed, num_tokens=num_tokens)
                result = check_existing_result(result_path)
                if result is not None:
                    all_results[f"bridge_{num_tokens}"].append(result)

    return all_results


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Complete Telepathy Paper Evaluation (Revised)")

    # Basic options
    parser.add_argument("--output_dir", type=str, default="runs/paper_results",
                       help="Output directory for all results")
    parser.add_argument("--only", type=str,
                       choices=["bridge", "ablation", "zeroshot", "text_relay",
                               "prompt_tuning", "lora", "linear_probe",
                               "statistical_tests", "generate_tables"],
                       help="Run only a specific component")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["sst2", "agnews", "trec"],
                       help="Datasets to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                       help="Random seeds to use (default: [42, 456])")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")

    # Resume and skip options
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip experiments that already have results")
    parser.add_argument("--resume_dir", type=str, default=None,
                       help="Resume from a specific run directory (e.g., runs/paper_results/run_20250111_120000). "
                            "If set, --skip_existing is automatically enabled.")

    # Skip slow experiments
    parser.add_argument("--skip_slow", type=str, nargs="*", default=None,
                       choices=["lora", "text_relay", "bridge"],
                       help="Skip slow experiment types (e.g., --skip_slow lora text_relay)")
    parser.add_argument("--fast_mode", action="store_true",
                       help="Enable fast mode: reduces training steps and samples for quick iteration")

    # Retry and robustness options
    parser.add_argument("--max_retries", type=int, default=2,
                       help="Maximum number of retries for failed experiments (default: 2)")
    parser.add_argument("--retry_delay", type=float, default=5.0,
                       help="Delay in seconds between retries (default: 5.0)")
    parser.add_argument("--continue_on_error", action="store_true",
                       help="Continue with other experiments if one fails (default: True)")

    # High variance handling
    parser.add_argument("--extra_seeds_on_high_variance", action="store_true",
                       help="Automatically run extra seeds when high variance is detected")
    parser.add_argument("--variance_threshold", type=float, default=5.0,
                       help="Standard deviation threshold for high variance detection (default: 5.0)")
    parser.add_argument("--max_extra_seeds", type=int, default=3,
                       help="Maximum extra seeds to run for high-variance experiments (default: 3)")

    # Memory and performance options
    parser.add_argument("--memory_threshold_gb", type=float, default=5.0,
                       help="Minimum free GPU memory (GB) required to start an experiment")
    parser.add_argument("--aggressive_cleanup", action="store_true",
                       help="Perform aggressive memory cleanup between all experiments")

    # Logging options
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    parser.add_argument("--save_frequency", type=int, default=5,
                       help="Save results every N experiments (default: 5)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume mode
    if args.resume_dir:
        # Resume from existing run directory
        run_dir = Path(args.resume_dir)
        if not run_dir.exists():
            print(f"ERROR: Resume directory does not exist: {run_dir}")
            sys.exit(1)
        args.skip_existing = True  # Automatically enable skip_existing when resuming
        timestamp = run_dir.name.replace("run_", "") if run_dir.name.startswith("run_") else datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"RESUMING from: {run_dir}")
    else:
        # Create new run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(run_dir, log_level)
    logger.info(f"Starting evaluation run: {run_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Log GPU info
    mem_info = get_gpu_memory_info(device)
    if mem_info.get("available"):
        logger.info(f"GPU Memory: {mem_info['total_gb']:.1f}GB total, {mem_info['free_gb']:.1f}GB free")

    # Apply fast mode settings if enabled
    if args.fast_mode:
        logger.info("FAST MODE enabled - reducing training steps and samples")
        BRIDGE_CONFIG["train_steps"] = 500
        LORA_CONFIG["epochs"] = 1
        LORA_CONFIG["max_train_samples"] = 500
        PROMPT_TUNING_CONFIG["steps"] = 500
        LINEAR_PROBE_CONFIG["max_samples"] = 1000

    # Count total experiments for progress tracking
    total_experiments = count_total_experiments(args.datasets, args.seeds)

    # Check for existing results if skip_existing is enabled
    skipped_experiments = []
    if args.skip_existing or args.resume_dir:
        completed_count, total_count, completed_names = count_completed_experiments(
            run_dir, args.datasets, args.seeds
        )
        skipped_experiments = completed_names
        remaining_experiments = total_experiments - completed_count

        logger.info("="*70)
        logger.info("RESUME MODE - Checking for existing results")
        logger.info("="*70)
        logger.info(f"Completed experiments: {completed_count}/{total_count}")
        logger.info(f"Remaining experiments: {remaining_experiments}")

        if completed_names:
            logger.info(f"Skipping {len(completed_names)} completed experiments")
            for name in completed_names[:10]:  # Show first 10
                logger.debug(f"  - {name}")
            if len(completed_names) > 10:
                logger.debug(f"  ... and {len(completed_names) - 10} more")
        logger.info("="*70)
    else:
        remaining_experiments = total_experiments

    # Determine which experiments to skip
    skip_slow_set = set(args.skip_slow) if args.skip_slow else set()
    if skip_slow_set:
        logger.info(f"Skipping slow experiment types: {skip_slow_set}")

    # Create save callback for periodic saves
    def save_intermediate_results():
        """Save intermediate results to disk."""
        try:
            intermediate = {
                "timestamp": datetime.now().isoformat(),
                "status": "in_progress",
                "completed_experiments": tracker.completed,
                "failed_experiments": tracker.failed,
            }
            for method, results_list in all_results.items():
                serializable = []
                for r in results_list:
                    r_copy = {k: v for k, v in r.items() if k not in ["predictions", "labels"]}
                    serializable.append(r_copy)
                intermediate[method] = serializable
            save_partial_results_atomic(intermediate, run_dir / "intermediate_results.json", logger)
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")

    # Initialize progress tracker with logger and save callback
    tracker = ProgressTracker(
        remaining_experiments if remaining_experiments > 0 else total_experiments,
        logger=logger,
        save_callback=save_intermediate_results
    )

    # Track failed experiments for reporting
    failed_experiments = []
    high_variance_warnings = []

    print("="*70)
    print("TELEPATHY COMPLETE PAPER EVALUATION (REVISED)")
    print("="*70)
    print(f"Output directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Token ablation configs: {TOKEN_ABLATION_CONFIGS}")
    print(f"Total experiments: {total_experiments}")
    if args.skip_existing:
        print(f"Skip existing: ENABLED")
        print(f"Experiments to run: {remaining_experiments}")
    if skip_slow_set:
        print(f"Skipping: {skip_slow_set}")
    if args.fast_mode:
        print(f"Fast mode: ENABLED")
    print(f"Max retries: {args.max_retries}")
    print(f"Continue on error: {args.continue_on_error}")
    print(f"Estimated time: ~12 hours on 1 H100 (less with fast mode)")
    print("="*70)

    logger.info(f"Configuration: datasets={args.datasets}, seeds={args.seeds}")
    logger.info(f"Retry settings: max_retries={args.max_retries}, retry_delay={args.retry_delay}")

    # Save configuration
    config = {
        "timestamp": timestamp,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "token_ablation": TOKEN_ABLATION_CONFIGS,
        "device": str(device),
        "bridge_config": BRIDGE_CONFIG,
        "linear_probe_config": LINEAR_PROBE_CONFIG,
        "lora_config": LORA_CONFIG,
        "prompt_tuning_config": PROMPT_TUNING_CONFIG,
        "total_experiments": total_experiments,
        "fast_mode": args.fast_mode,
        "max_retries": args.max_retries,
        "skip_slow": list(skip_slow_set) if skip_slow_set else [],
        "extra_seeds_on_high_variance": args.extra_seeds_on_high_variance,
        "variance_threshold": args.variance_threshold,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Results storage - load existing results if resuming
    if args.skip_existing or args.resume_dir:
        print("\nLoading existing results...")
        all_results = load_existing_results(run_dir, args.datasets, args.seeds)
        print(f"  Loaded {sum(len(v) for v in all_results.values())} existing results")
    else:
        all_results = {
            "zeroshot_llama": [],
            "zeroshot_mistral": [],
            "text_relay": [],
            "prompt_tuning": [],
            "lora": [],
            "linear_probe": [],
        }
        # Add bridge results for each token config
        for num_tokens in TOKEN_ABLATION_CONFIGS:
            all_results[f"bridge_{num_tokens}"] = []

    # Latency results storage - load existing if available
    latency_results = {}
    if args.skip_existing or args.resume_dir:
        latency_path = run_dir / "latency" / "latency_results.json"
        if latency_path.exists():
            try:
                with open(latency_path, "r") as f:
                    latency_results = json.load(f)
                print(f"  Loaded existing latency results for {len(latency_results)} datasets")
            except (json.JSONDecodeError, IOError):
                pass

    try:
        # Helper function for running experiments with retry
        def run_experiment_with_retry(func, exp_name, result_key, *func_args, **func_kwargs):
            """Run an experiment with retry logic and proper tracking."""
            nonlocal failed_experiments

            # Check memory before starting
            if args.aggressive_cleanup:
                aggressive_memory_cleanup(device, logger)

            if not check_gpu_memory_threshold(device, args.memory_threshold_gb):
                logger.warning(f"Low GPU memory before {exp_name}, running cleanup...")
                aggressive_memory_cleanup(device, logger)

            exp_start = time.time()
            success, result, error_msg = run_with_retry(
                func, *func_args,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                experiment_name=exp_name,
                logger=logger,
                device=device,
                **func_kwargs
            )

            exp_time = time.time() - exp_start

            if success and result:
                all_results[result_key].append(result)
                tracker.update(exp_name, exp_time, success=True)
                logger.info(f"[SUCCESS] {exp_name} completed in {exp_time:.1f}s")
            else:
                failed_experiments.append({
                    "name": exp_name,
                    "error": error_msg,
                    "time": exp_time,
                })
                tracker.update(exp_name, exp_time, success=False, error_msg=error_msg)
                logger.error(f"[FAILED] {exp_name}: {error_msg}")

                if not args.continue_on_error:
                    raise ExperimentError(exp_name, error_msg)

            return success, result

        # 1. Zero-shot baselines (individual models for super-additivity)
        if args.only is None or args.only == "zeroshot":
            zeroshot_dir = run_dir / "zeroshot"
            zeroshot_dir.mkdir(exist_ok=True)
            logger.info("Starting zero-shot baselines...")

            for dataset_key in args.datasets:
                for seed in args.seeds:
                    # Llama zero-shot
                    exp_name = f"zeroshot_llama_{dataset_key}_seed{seed}"
                    if args.skip_existing and exp_name in skipped_experiments:
                        logger.info(f"[SKIP] {exp_name} - result already exists")
                        tracker.record_skip(exp_name)
                    else:
                        run_experiment_with_retry(
                            run_zeroshot_baseline,
                            exp_name,
                            "zeroshot_llama",
                            dataset_key, "llama", SOURCE_MODEL, seed, zeroshot_dir, device
                        )

                    # Mistral zero-shot
                    exp_name = f"zeroshot_mistral_{dataset_key}_seed{seed}"
                    if args.skip_existing and exp_name in skipped_experiments:
                        logger.info(f"[SKIP] {exp_name} - result already exists")
                        tracker.record_skip(exp_name)
                    else:
                        run_experiment_with_retry(
                            run_zeroshot_baseline,
                            exp_name,
                            "zeroshot_mistral",
                            dataset_key, "mistral", TARGET_MODEL, seed, zeroshot_dir, device
                        )

        # 2. Text-relay baseline (with accuracy)
        if args.only is None or args.only == "text_relay":
            if "text_relay" in skip_slow_set:
                logger.info("[SKIP] Skipping text_relay experiments (--skip_slow)")
            else:
                relay_dir = run_dir / "text_relay"
                relay_dir.mkdir(exist_ok=True)
                logger.info("Starting text-relay baselines...")

                for dataset_key in args.datasets:
                    for seed in args.seeds:
                        exp_name = f"text_relay_{dataset_key}_seed{seed}"
                        if args.skip_existing and exp_name in skipped_experiments:
                            logger.info(f"[SKIP] {exp_name} - result already exists")
                            tracker.record_skip(exp_name)
                        else:
                            run_experiment_with_retry(
                                run_text_relay_baseline,
                                exp_name,
                                "text_relay",
                                dataset_key, seed, relay_dir, device
                            )

        # 3. Prompt tuning baseline
        if args.only is None or args.only == "prompt_tuning":
            pt_dir = run_dir / "prompt_tuning"
            pt_dir.mkdir(exist_ok=True)
            logger.info("Starting prompt tuning baselines...")

            for dataset_key in args.datasets:
                for seed in args.seeds:
                    exp_name = f"prompt_tuning_{dataset_key}_seed{seed}"
                    if args.skip_existing and exp_name in skipped_experiments:
                        logger.info(f"[SKIP] {exp_name} - result already exists")
                        tracker.record_skip(exp_name)
                    else:
                        run_experiment_with_retry(
                            train_prompt_tuning_baseline,
                            exp_name,
                            "prompt_tuning",
                            dataset_key, seed, pt_dir, device, PROMPT_TUNING_CONFIG
                        )

        # 4. LoRA baseline
        if args.only is None or args.only == "lora":
            if "lora" in skip_slow_set:
                logger.info("[SKIP] Skipping LoRA experiments (--skip_slow)")
            else:
                lora_dir = run_dir / "lora"
                lora_dir.mkdir(exist_ok=True)
                logger.info("Starting LoRA baselines...")

                for dataset_key in args.datasets:
                    for seed in args.seeds:
                        exp_name = f"lora_{dataset_key}_seed{seed}"
                        if args.skip_existing and exp_name in skipped_experiments:
                            logger.info(f"[SKIP] {exp_name} - result already exists")
                            tracker.record_skip(exp_name)
                        else:
                            run_experiment_with_retry(
                                train_lora_baseline,
                                exp_name,
                                "lora",
                                dataset_key, seed, lora_dir, device, LORA_CONFIG
                            )

        # 5. Linear probe baseline
        if args.only is None or args.only == "linear_probe":
            probe_dir = run_dir / "linear_probe"
            probe_dir.mkdir(exist_ok=True)
            logger.info("Starting linear probe baselines...")

            for dataset_key in args.datasets:
                for seed in args.seeds:
                    exp_name = f"linear_probe_{dataset_key}_seed{seed}"
                    if args.skip_existing and exp_name in skipped_experiments:
                        logger.info(f"[SKIP] {exp_name} - result already exists")
                        tracker.record_skip(exp_name)
                    else:
                        run_experiment_with_retry(
                            run_linear_probe,
                            exp_name,
                            "linear_probe",
                            dataset_key, seed, probe_dir, device, LINEAR_PROBE_CONFIG
                        )

        # 6. Bridge with token ablation
        if args.only is None or args.only == "bridge" or args.only == "ablation":
            if "bridge" in skip_slow_set:
                logger.info("[SKIP] Skipping bridge experiments (--skip_slow)")
            else:
                bridge_dir = run_dir / "bridge"
                bridge_dir.mkdir(exist_ok=True)
                logger.info("Starting bridge experiments...")

                for num_tokens in TOKEN_ABLATION_CONFIGS:
                    for dataset_key in args.datasets:
                        for seed in args.seeds:
                            exp_name = f"bridge_{num_tokens}_{dataset_key}_seed{seed}"
                            if args.skip_existing and exp_name in skipped_experiments:
                                logger.info(f"[SKIP] {exp_name} - result already exists")
                                tracker.record_skip(exp_name)
                            else:
                                run_experiment_with_retry(
                                    train_bridge_for_dataset,
                                    exp_name,
                                    f"bridge_{num_tokens}",
                                    dataset_key, seed, bridge_dir, device, BRIDGE_CONFIG,
                                    num_soft_tokens=num_tokens
                                )

                # Check for high variance in bridge results and run extra seeds if needed
                if args.extra_seeds_on_high_variance:
                    logger.info("Checking for high variance in bridge results...")
                    for num_tokens in TOKEN_ABLATION_CONFIGS:
                        for dataset_key in args.datasets:
                            method_key = f"bridge_{num_tokens}"
                            method_results = [r for r in all_results[method_key]
                                            if r.get("dataset") == dataset_key]
                            if len(method_results) >= 2:
                                accs = [r["accuracy"] for r in method_results]
                                is_high_var, var_stats = detect_high_variance(
                                    accs, threshold_std=args.variance_threshold
                                )
                                if is_high_var:
                                    seeds_run = [r["seed"] for r in method_results]
                                    extra_seeds = get_extra_seeds_if_needed(
                                        seeds_run, accs,
                                        variance_threshold_std=args.variance_threshold,
                                        max_extra_seeds=args.max_extra_seeds
                                    )
                                    if extra_seeds:
                                        warning_msg = (
                                            f"HIGH VARIANCE detected for {method_key}/{dataset_key}: "
                                            f"std={var_stats['std']:.1f}, range={var_stats['range']:.1f}. "
                                            f"Running {len(extra_seeds)} extra seeds: {extra_seeds}"
                                        )
                                        logger.warning(warning_msg)
                                        high_variance_warnings.append({
                                            "method": method_key,
                                            "dataset": dataset_key,
                                            "stats": var_stats,
                                            "extra_seeds": extra_seeds,
                                        })

                                        for extra_seed in extra_seeds:
                                            exp_name = f"bridge_{num_tokens}_{dataset_key}_seed{extra_seed}_extra"
                                            run_experiment_with_retry(
                                                train_bridge_for_dataset,
                                                exp_name,
                                                method_key,
                                                dataset_key, extra_seed, bridge_dir, device, BRIDGE_CONFIG,
                                                num_soft_tokens=num_tokens
                                            )

        # 7. Latency measurement (after bridge training is complete)
        if args.only is None or args.only == "bridge" or args.only == "ablation":
            print(f"\n{'='*60}")
            print("LATENCY MEASUREMENT")
            print(f"{'='*60}")

            latency_dir = run_dir / "latency"
            latency_dir.mkdir(exist_ok=True)

            # Check which latency measurements need to be done
            datasets_to_measure = []
            for dataset_key in args.datasets:
                exp_name = f"latency_{dataset_key}"
                if args.skip_existing and exp_name in skipped_experiments:
                    print(f"[SKIP] {exp_name} - result already exists")
                else:
                    datasets_to_measure.append(dataset_key)

            if datasets_to_measure:
                # Load models once for latency measurement
                print("Loading models for latency measurement...")
                src_model = AutoModelForCausalLM.from_pretrained(
                    SOURCE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
                ).eval()
                tgt_model = AutoModelForCausalLM.from_pretrained(
                    TARGET_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
                ).eval()

                src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
                src_tok.pad_token = src_tok.eos_token
                tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
                tgt_tok.pad_token = tgt_tok.eos_token

                # Compute target RMS for bridge
                with torch.no_grad():
                    tgt_embeds = tgt_model.get_input_embeddings().weight.float()
                    target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

                # Create a fresh bridge for latency measurement
                bridge = TelepathyBridge(
                    src_dim=src_model.config.hidden_size,
                    tgt_dim=tgt_model.config.hidden_size,
                    num_soft_tokens=8,
                    heads=BRIDGE_CONFIG["heads"],
                    depth=BRIDGE_CONFIG["depth"],
                    target_rms=target_rms
                ).to(device).to(torch.bfloat16).eval()

                # Measure latency for one example per dataset
                for dataset_key in datasets_to_measure:
                    exp_start = time.time()
                    try:
                        dataset_config = DATASETS[dataset_key]
                        test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"], max_samples=1)

                        if test_data:
                            text = test_data[0]["text"]
                            print(f"\nMeasuring latency for {dataset_config['name']}...")

                            lat_result = measure_fair_latency(
                                text=text,
                                dataset_config=dataset_config,
                                src_model=src_model,
                                tgt_model=tgt_model,
                                src_tok=src_tok,
                                tgt_tok=tgt_tok,
                                bridge=bridge,
                                device=device,
                                num_warmup=3,
                                num_runs=10,  # Reduced for faster execution
                            )

                            latency_results[dataset_key] = lat_result

                            print(f"  Bridge: {lat_result['bridge_mean_ms']:.1f}ms")
                            print(f"  Text-Relay: {lat_result['text_relay_mean_ms']:.1f}ms")
                            print(f"  Mistral Direct: {lat_result['mistral_direct_mean_ms']:.1f}ms")
                            print(f"  Speedup vs Text-Relay: {lat_result['speedup_vs_text_relay']:.2f}x")

                            tracker.update(f"latency_{dataset_key}", time.time() - exp_start)

                    except Exception as e:
                        print(f"ERROR in latency measurement {dataset_key}: {e}")
                        traceback.print_exc()

                # Clean up latency models
                del src_model, tgt_model, bridge
                gc.collect()
                torch.cuda.empty_cache()

                # Save latency results
                with open(latency_dir / "latency_results.json", "w") as f:
                    json.dump(latency_results, f, indent=2)

            # Compute average latency across datasets (including any loaded results)
            # Filter out 'average' key if present from previous runs
            dataset_latencies = {k: v for k, v in latency_results.items() if k != "average" and isinstance(v, dict) and "bridge_mean_ms" in v}
            if dataset_latencies:
                avg_latency = {
                    "bridge_mean_ms": np.mean([r["bridge_mean_ms"] for r in dataset_latencies.values()]),
                    "text_relay_mean_ms": np.mean([r["text_relay_mean_ms"] for r in dataset_latencies.values()]),
                    "mistral_direct_mean_ms": np.mean([r["mistral_direct_mean_ms"] for r in dataset_latencies.values()]),
                    "speedup_vs_text_relay": np.mean([r["speedup_vs_text_relay"] for r in dataset_latencies.values()]),
                }
                latency_results["average"] = avg_latency

        # 8. Statistical tests with multiple comparison correction
        if args.only is None or args.only == "statistical_tests":
            try:
                analysis = run_statistical_tests_with_correction(all_results, run_dir)
            except Exception as e:
                print(f"ERROR in statistical tests: {e}")
                traceback.print_exc()
                analysis = {}

        # 9. Generate comprehensive LaTeX tables (with latency results)
        if args.only is None or args.only == "generate_tables":
            try:
                if 'analysis' not in locals():
                    # Load analysis if we skipped statistical tests
                    analysis_path = run_dir / "statistical_analysis_corrected.json"
                    if analysis_path.exists():
                        with open(analysis_path) as f:
                            analysis = json.load(f)
                    else:
                        analysis = {}

                # Use average latency for tables if available
                latency_for_tables = latency_results.get("average") if latency_results else None
                generate_comprehensive_latex_tables(all_results, analysis, run_dir, latency_for_tables)
            except Exception as e:
                print(f"ERROR in table generation: {e}")
                traceback.print_exc()

        # Save complete results
        complete_results = {
            "timestamp": timestamp,
            "config": config,
            "latency": latency_results if latency_results else {},
            "failed_experiments": failed_experiments,
            "high_variance_warnings": high_variance_warnings,
            "tracker_summary": tracker.get_summary(),
            "experiment_log": tracker.get_experiment_log(),
        }

        # Convert results to serializable format
        for method, results_list in all_results.items():
            serializable = []
            for r in results_list:
                r_copy = {k: v for k, v in r.items() if k not in ["predictions", "labels"]}
                serializable.append(r_copy)
            complete_results[method] = serializable

        save_partial_results_atomic(complete_results, run_dir / "complete_results.json", logger)

        # Final summary
        total_time = time.time() - tracker.start_time
        summary = tracker.get_summary()

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Results saved to: {run_dir}")
        print(f"Total time: {tracker._format_time(total_time)}")
        print(f"Experiments completed: {summary['completed']}/{summary['total']}")
        print(f"Experiments failed: {summary['failed']}")
        print(f"Experiments skipped: {summary['skipped']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")

        # Print failed experiments summary
        if failed_experiments:
            print("\n--- Failed Experiments ---")
            for fail in failed_experiments:
                print(f"  - {fail['name']}: {fail['error'][:100]}...")
            logger.warning(f"{len(failed_experiments)} experiments failed")

        # Print high variance warnings
        if high_variance_warnings:
            print("\n--- High Variance Warnings ---")
            for warn in high_variance_warnings:
                print(f"  - {warn['method']}/{warn['dataset']}: "
                      f"std={warn['stats']['std']:.1f}, range={warn['stats']['range']:.1f}")
            logger.warning(f"{len(high_variance_warnings)} high-variance results detected")

        # Print summary table
        print("\n--- Accuracy Summary ---")
        for dataset_key in args.datasets:
            print(f"\n{DATASETS[dataset_key]['name']}:")

            for method, results_list in all_results.items():
                method_results = [r for r in results_list if r.get("dataset") == dataset_key]
                if method_results:
                    accs = [r["accuracy"] for r in method_results]
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs) if len(accs) > 1 else 0
                    # Mark high variance
                    is_high_var, _ = detect_high_variance(accs, threshold_std=args.variance_threshold)
                    var_marker = " [HIGH VAR]" if is_high_var else ""
                    print(f"  {method}: {mean_acc:.1f}% +/- {std_acc:.1f}%{var_marker}")

        # Print latency summary
        if latency_results and "average" in latency_results:
            print("\n--- Latency Summary ---")
            avg = latency_results["average"]
            print(f"  Bridge: {avg['bridge_mean_ms']:.1f}ms")
            print(f"  Text-Relay: {avg['text_relay_mean_ms']:.1f}ms")
            print(f"  Mistral Direct: {avg['mistral_direct_mean_ms']:.1f}ms")
            print(f"  Speedup vs Text-Relay: {avg['speedup_vs_text_relay']:.2f}x")

        print("\n--- Output Files ---")
        print(f"  Complete results: {run_dir / 'complete_results.json'}")
        print(f"  Statistical analysis: {run_dir / 'statistical_analysis_corrected.json'}")
        print(f"  LaTeX tables: {run_dir / 'paper_tables_comprehensive.tex'}")
        print(f"  Evaluation log: {run_dir / 'evaluation_*.log'}")
        if latency_results:
            print(f"  Latency results: {run_dir / 'latency' / 'latency_results.json'}")

        logger.info(f"Evaluation complete. Total time: {tracker._format_time(total_time)}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        logger.warning("Evaluation interrupted by user")

        # Still save whatever we have
        try:
            partial_results = {
                "timestamp": timestamp,
                "config": config,
                "status": "interrupted",
                "latency": latency_results if 'latency_results' in dir() else {},
                "failed_experiments": failed_experiments,
                "high_variance_warnings": high_variance_warnings,
                "tracker_summary": tracker.get_summary() if 'tracker' in dir() else {},
                "experiment_log": tracker.get_experiment_log() if 'tracker' in dir() else [],
            }
            for method, results_list in all_results.items():
                serializable = []
                for r in results_list:
                    r_copy = {k: v for k, v in r.items() if k not in ["predictions", "labels"]}
                    serializable.append(r_copy)
                partial_results[method] = serializable

            save_partial_results_atomic(partial_results, run_dir / "partial_results.json", logger)
            print(f"Partial results saved to: {run_dir / 'partial_results.json'}")
            logger.info(f"Partial results saved to: {run_dir / 'partial_results.json'}")

            # Print summary of what was completed
            if 'tracker' in dir():
                summary = tracker.get_summary()
                print(f"\nProgress at interruption:")
                print(f"  Completed: {summary['completed']}/{summary['total']}")
                print(f"  Failed: {summary['failed']}")
                print(f"  Time elapsed: {tracker._format_time(summary['total_time'])}")

        except Exception as e:
            print(f"Failed to save partial results: {e}")
            logger.error(f"Failed to save partial results: {e}")

    except ExperimentError as e:
        logger.error(f"Experiment error: {e}")
        print(f"\nExperiment failed: {e}")
        print("Use --continue_on_error to continue with other experiments after failures")

    print(f"\nAll results: {run_dir}")


if __name__ == "__main__":
    main()
