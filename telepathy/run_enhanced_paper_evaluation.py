#!/usr/bin/env python3
"""
Enhanced Telepathy Paper Evaluation Script

Builds on run_complete_evaluation.py with these enhancements:
1. 5 seeds (42, 123, 456, 789, 1011) for robust statistical analysis
2. 4 token configs (8, 16, 24, 32) for comprehensive ablation
3. 3 classification datasets: SST-2, AG News, TREC
4. 3 reasoning benchmarks: BoolQ, ARC-Easy, GSM8K
5. Improved text-relay baseline with task-specific prompting
6. Enhanced resume capability with atomic checkpointing
7. Comprehensive memory management for 12-hour H100 execution

Constraints:
- Runs on 1 H100 GPU in ~12 hours
- Uses 5 seeds [42, 123, 456, 789, 1011]
- Tests on 6 datasets: SST-2, AG News, TREC, BoolQ, ARC-Easy, GSM8K

Usage:
    python telepathy/run_enhanced_paper_evaluation.py --output_dir runs/enhanced_results

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
    log_file = log_dir / f"enhanced_evaluation_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("telepathy_enhanced_eval")
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


class AtomicCheckpointer:
    """Manages atomic checkpointing for long-running experiments with resume capability."""

    def __init__(self, checkpoint_dir: Path, experiment_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.checkpoint_path = checkpoint_dir / f"{experiment_name}_checkpoint.json"
        self.lock_path = checkpoint_dir / f"{experiment_name}_checkpoint.lock"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        self.completed_experiments = set()

    def mark_completed(self, exp_id: str, result: Dict):
        """Mark an experiment as completed and save checkpoint atomically."""
        self.completed_experiments.add(exp_id)
        self.data[exp_id] = result
        self._save_atomic()

    def is_completed(self, exp_id: str) -> bool:
        """Check if an experiment has already been completed."""
        return exp_id in self.completed_experiments

    def get_result(self, exp_id: str) -> Optional[Dict]:
        """Get the result for a completed experiment."""
        return self.data.get(exp_id)

    def _save_atomic(self):
        """Save checkpoint atomically using temp file + rename."""
        checkpoint = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "completed_experiments": list(self.completed_experiments),
            "data": self.data,
        }
        save_partial_results_atomic(checkpoint, self.checkpoint_path)

    def load(self) -> bool:
        """Load checkpoint if it exists. Returns True if loaded successfully."""
        if not self.checkpoint_path.exists():
            return False
        try:
            with open(self.checkpoint_path) as f:
                checkpoint = json.load(f)
            self.completed_experiments = set(checkpoint.get("completed_experiments", []))
            self.data = checkpoint.get("data", {})
            return True
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return False

    def clear(self):
        """Clear checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        if self.lock_path.exists():
            self.lock_path.unlink()


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def detect_high_variance(
    accuracies: List[float],
    threshold_std: float = 5.0,
    threshold_range: float = 15.0,
) -> Tuple[bool, Dict[str, float]]:
    """Detect if experiment results have high variance."""
    if len(accuracies) < 2:
        return False, {"std": 0, "range": 0, "mean": accuracies[0] if accuracies else 0}

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)
    range_acc = max(accuracies) - min(accuracies)
    cv = (std_acc / mean_acc * 100) if mean_acc > 0 else 0

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


# Import from statistical_testing.py if available
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
try:
    from statistical_testing import (
        bootstrap_ci,
        paired_ttest,
        mcnemar_test,
        multiple_comparison_correction,
        cohens_d_paired,
        p_value_to_stars,
    )
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: statistical_testing.py not found, using basic statistics")


# =============================================================================
# CONFIGURATION - ENHANCED
# =============================================================================

# Classification datasets (3)
CLASSIFICATION_DATASETS = {
    "sst2": {
        "name": "SST-2",
        "type": "classification",
        "num_classes": 2,
        "split_train": "train",
        "split_test": "validation",
        "load_fn": "glue:sst2",
        "text_field": "sentence",
        "label_field": "label",
        "label_names": ["negative", "positive"],
        "prompt_template": "Review: {text}\nSentiment (positive or negative):",
        "text_relay_prompt": "Summarize this review while preserving its sentiment:\n\n{text}\n\nSummary:",
        "primer": "Sentiment:",
        "max_length": 128,
    },
    "agnews": {
        "name": "AG News",
        "type": "classification",
        "num_classes": 4,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "fancyzhx/ag_news",
        "text_field": "text",
        "label_field": "label",
        "label_names": ["world", "sports", "business", "science"],
        "prompt_template": "Article: {text}\nTopic (world, sports, business, or science):",
        "text_relay_prompt": "Summarize this news article, preserving the main topic:\n\n{text}\n\nSummary:",
        "primer": "Topic:",
        "max_length": 256,
    },
    "trec": {
        "name": "TREC",
        "type": "classification",
        "num_classes": 6,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "trec",
        "text_field": "text",
        "label_field": "coarse_label",
        "label_names": ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
        "prompt_template": "Question: {text}\nCategory (ABBR, ENTY, DESC, HUM, LOC, or NUM):",
        "text_relay_prompt": "Rephrase this question clearly, preserving what type of answer is expected:\n\n{text}\n\nRephrased:",
        "primer": "Category:",
        "max_length": 128,
    },
}

# Reasoning benchmarks (3)
REASONING_DATASETS = {
    "boolq": {
        "name": "BoolQ",
        "type": "reasoning",
        "num_classes": 2,
        "split_train": "train",
        "split_test": "validation",
        "load_fn": "boolq",
        "text_field": "question",
        "context_field": "passage",
        "label_field": "answer",
        "label_names": ["no", "yes"],
        "prompt_template": "Passage: {context}\n\nQuestion: {text}\n\nAnswer (yes or no):",
        "text_relay_prompt": "Summarize this passage and question, preserving key facts needed to answer:\n\nPassage: {context}\n\nQuestion: {text}\n\nSummary:",
        "primer": "Answer:",
        "max_length": 384,
    },
    "arc_easy": {
        "name": "ARC-Easy",
        "type": "reasoning",
        "num_classes": 4,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "ai2_arc:ARC-Easy",
        "text_field": "question",
        "choices_field": "choices",
        "label_field": "answerKey",
        "label_names": ["A", "B", "C", "D"],
        "prompt_template": "Question: {text}\n\nChoices:\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}\nD) {choice_d}\n\nAnswer (A, B, C, or D):",
        "text_relay_prompt": "Rephrase this science question and choices clearly:\n\nQuestion: {text}\n\nChoices:\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}\nD) {choice_d}\n\nRephrased question and choices:",
        "primer": "Answer:",
        "max_length": 256,
    },
    "gsm8k": {
        "name": "GSM8K",
        "type": "reasoning",
        "num_classes": None,  # Free-form numeric answer
        "split_train": "train",
        "split_test": "test",
        "load_fn": "gsm8k:main",
        "text_field": "question",
        "label_field": "answer",
        "label_names": None,
        "prompt_template": "Problem: {text}\n\nSolve step by step and give the final numerical answer:",
        "text_relay_prompt": "Simplify this math problem while preserving all numerical information:\n\n{text}\n\nSimplified problem:",
        "primer": "Solution:",
        "max_length": 384,
    },
}

# Combined datasets
DATASETS = {**CLASSIFICATION_DATASETS, **REASONING_DATASETS}

# Random seeds - 5 for robust statistical analysis
SEEDS = [42, 123, 456, 789, 1011]

# Model configurations
SOURCE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Token ablation configurations - 4 levels
TOKEN_ABLATION_CONFIGS = [8, 16, 24, 32]

# Bridge configurations (optimized for 12-hour budget with 5 seeds x 4 tokens x 6 datasets)
BRIDGE_CONFIG = {
    "soft_tokens": 8,
    "depth": 2,
    "heads": 8,
    "source_layer": 31,
    "train_steps": 1000,  # Reduced from 1500 for time budget
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
    "max_samples": 3000,  # Reduced for time budget
}

# LoRA configurations
LORA_CONFIG = {
    "rank": 8,
    "alpha": 16,
    "epochs": 1,  # Reduced for time budget
    "batch_size": 4,
    "lr": 1e-4,
    "max_train_samples": 1500,
}

# Prompt tuning configurations
PROMPT_TUNING_CONFIG = {
    "soft_tokens": 8,
    "lr": 2e-4,
    "steps": 1000,  # Reduced for time budget
    "batch_size": 16,
    "grad_accum": 2,
}


# =============================================================================
# DATA LOADING - ENHANCED FOR REASONING BENCHMARKS
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

    # Extract examples
    examples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        example = {"idx": idx}

        # Handle text field
        example["text"] = item[config["text_field"]]

        # Handle context field for BoolQ
        if "context_field" in config and config["context_field"]:
            example["context"] = item.get(config["context_field"], "")

        # Handle choices for ARC
        if "choices_field" in config and config["choices_field"]:
            choices = item[config["choices_field"]]
            if isinstance(choices, dict):
                # ARC format: {"text": [...], "label": [...]}
                example["choices"] = choices.get("text", [])
                example["choice_labels"] = choices.get("label", [])
            else:
                example["choices"] = choices

        # Handle label
        if config["label_field"]:
            label = item[config["label_field"]]
            if dataset_key == "boolq":
                # BoolQ has boolean labels
                example["label"] = 1 if label else 0
            elif dataset_key == "arc_easy":
                # ARC has letter labels (A, B, C, D)
                example["label"] = label
            elif dataset_key == "gsm8k":
                # GSM8K has free-form answers
                example["answer_text"] = label
                # Extract final numeric answer
                example["label"] = extract_gsm8k_answer(label)
            else:
                example["label"] = label

        examples.append(example)

    return examples


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K answer text."""
    import re
    # GSM8K answers end with "#### <number>"
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) > 1:
            return parts[-1].strip().replace(",", "")
    # Fallback: try to extract last number
    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    return numbers[-1] if numbers else ""


def format_prompt_for_example(example: Dict, config: Dict) -> str:
    """Format the prompt for a given example based on dataset config."""
    template = config["prompt_template"]

    if config.get("type") == "reasoning":
        if "context" in example:
            # BoolQ
            return template.format(text=example["text"], context=example["context"])
        elif "choices" in example:
            # ARC
            choices = example.get("choices", ["", "", "", ""])
            # Pad choices if needed
            while len(choices) < 4:
                choices.append("")
            return template.format(
                text=example["text"],
                choice_a=choices[0] if len(choices) > 0 else "",
                choice_b=choices[1] if len(choices) > 1 else "",
                choice_c=choices[2] if len(choices) > 2 else "",
                choice_d=choices[3] if len(choices) > 3 else "",
            )
        else:
            # GSM8K
            return template.format(text=example["text"])
    else:
        # Classification
        return template.format(text=example["text"][:config["max_length"]])


def format_text_relay_prompt(example: Dict, config: Dict) -> str:
    """Format the text-relay prompt for task-specific summarization."""
    template = config.get("text_relay_prompt", config["prompt_template"])

    if config.get("type") == "reasoning":
        if "context" in example:
            # BoolQ
            return template.format(text=example["text"], context=example["context"][:256])
        elif "choices" in example:
            # ARC
            choices = example.get("choices", ["", "", "", ""])
            while len(choices) < 4:
                choices.append("")
            return template.format(
                text=example["text"],
                choice_a=choices[0] if len(choices) > 0 else "",
                choice_b=choices[1] if len(choices) > 1 else "",
                choice_c=choices[2] if len(choices) > 2 else "",
                choice_d=choices[3] if len(choices) > 3 else "",
            )
        else:
            # GSM8K
            return template.format(text=example["text"][:256])
    else:
        # Classification
        return template.format(text=example["text"][:256])


# =============================================================================
# COMPRESSION RATIO CALCULATION
# =============================================================================

def calculate_compression_ratio(
    text: str,
    num_soft_tokens: int,
    token_dim: int = 4096,
    quantization_bits: int = 16
) -> Dict[str, float]:
    """Calculate compression ratio between text and soft tokens."""
    text_bytes = len(text.encode('utf-8'))
    soft_token_bytes = num_soft_tokens * token_dim * quantization_bits / 8
    d_z = 256
    compressed_soft_token_bytes = num_soft_tokens * d_z * quantization_bits / 8

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


class SoftPromptTuning(torch.nn.Module):
    """Learnable soft prompts for a frozen LLM."""

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
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

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

        B = len(batch)

        # Format prompts based on dataset type
        src_texts = [format_prompt_for_example(item, dataset_config) for item in batch]

        # Get labels
        if dataset_key == "gsm8k":
            labels = [str(item.get("label", "")) for item in batch]
        else:
            labels = [dataset_config["label_names"][item["label"]]
                     if dataset_config["label_names"] else str(item["label"])
                     for item in batch]

        # Source encoding
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
                            truncation=True, max_length=32, add_special_tokens=False).to(device)
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

    # Evaluation
    bridge.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    compression_ratios = []

    print(f"\nEvaluating on {len(test_data)} test samples...")

    for item in tqdm(test_data, desc="Evaluating", leave=False):
        # Get ground truth label
        if dataset_key == "gsm8k":
            label = str(item.get("label", ""))
        else:
            label = dataset_config["label_names"][item["label"]] if dataset_config["label_names"] else str(item["label"])

        # Format prompt
        prompt = format_prompt_for_example(item, dataset_config)
        comp_ratio = calculate_compression_ratio(prompt, soft_tokens, tgt_model.config.hidden_size)
        compression_ratios.append(comp_ratio)

        src_enc = src_tok(prompt, return_tensors="pt", truncation=True,
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

            max_new = 64 if dataset_key == "gsm8k" else 10
            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        # Check prediction
        if dataset_key == "gsm8k":
            # Extract numeric answer
            pred_label = extract_gsm8k_answer(output)
            is_correct = pred_label == label
        else:
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

    accuracy = 100.0 * correct / total if total > 0 else 0.0
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
        "predictions": all_predictions,
        "labels": all_labels,
    }

    # Save results
    results_path = output_dir / f"{dataset_key}_seed{seed}_tokens{soft_tokens}_results.json"
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
# ZERO-SHOT BASELINES
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

    # Load test data
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    print(f"Evaluating on {len(test_data)} test samples...")

    for item in tqdm(test_data, desc=f"Evaluating {model_name}", leave=False):
        # Get ground truth
        if dataset_key == "gsm8k":
            label = str(item.get("label", ""))
        else:
            label = dataset_config["label_names"][item["label"]] if dataset_config["label_names"] else str(item["label"])

        prompt = format_prompt_for_example(item, dataset_config)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            max_new = 64 if dataset_key == "gsm8k" else 20
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        # Match prediction
        if dataset_key == "gsm8k":
            pred_label = extract_gsm8k_answer(response)
            is_correct = pred_label == label
        else:
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
# IMPROVED TEXT-RELAY BASELINE WITH TASK-SPECIFIC PROMPTING
# =============================================================================

def run_text_relay_baseline(
    dataset_key: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run improved text-relay baseline with task-specific prompting.

    Key improvements:
    1. Task-specific summarization prompts that preserve relevant information
    2. Explicit classification guidance for the target model
    3. Better handling of reasoning benchmarks
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Text-Relay Baseline (Improved): {dataset_config['name']} | Seed: {seed}")
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

    # Load test data
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    latencies = []

    print(f"Evaluating on {len(test_data)} test samples...")

    for item in tqdm(test_data, desc="Text-Relay", leave=False):
        # Get ground truth
        if dataset_key == "gsm8k":
            label = str(item.get("label", ""))
        else:
            label = dataset_config["label_names"][item["label"]] if dataset_config["label_names"] else str(item["label"])

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Step 1: Llama summarizes with task-specific prompt
        summary_prompt = format_text_relay_prompt(item, dataset_config)
        src_enc = src_tok(summary_prompt, return_tensors="pt", truncation=True, max_length=400).to(device)

        with torch.no_grad():
            summary_ids = src_model.generate(
                **src_enc,
                max_new_tokens=100 if dataset_key == "gsm8k" else 60,
                do_sample=False,
                pad_token_id=src_tok.eos_token_id
            )
            summary = src_tok.decode(summary_ids[0][src_enc.input_ids.shape[1]:], skip_special_tokens=True)

        # Step 2: Mistral classifies from summary with explicit guidance
        if dataset_key == "gsm8k":
            classify_prompt = f"Based on this problem summary, provide the final numerical answer:\n\n{summary}\n\nFinal answer (just the number):"
        elif dataset_config["label_names"]:
            label_list = ", ".join(dataset_config["label_names"])
            classify_prompt = f"{dataset_config['primer']}\n\nSummary: {summary}\n\nClassify as one of [{label_list}]. Answer:"
        else:
            classify_prompt = f"{dataset_config['primer']}\n\nSummary: {summary}\n\nAnswer:"

        tgt_enc = tgt_tok(classify_prompt, return_tensors="pt", truncation=True, max_length=300).to(device)

        with torch.no_grad():
            max_new = 64 if dataset_key == "gsm8k" else 10
            outputs = tgt_model.generate(
                **tgt_enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

        response = tgt_tok.decode(outputs[0][tgt_enc.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        # Match prediction
        if dataset_key == "gsm8k":
            pred_label = extract_gsm8k_answer(response)
            is_correct = pred_label == label
        else:
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
        "method": "text_relay_improved",
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
# LINEAR PROBE BASELINE
# =============================================================================

def extract_hidden_states(
    examples: List[Dict],
    model: torch.nn.Module,
    tokenizer,
    layer_idx: int,
    dataset_config: Dict,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract hidden states from a specific layer with memory-efficient implementation."""
    model.eval()
    all_hidden = []

    extracted_hidden = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        extracted_hidden.append(hidden.detach())

    # Get the specific layer
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        layers = model.transformer.h
    else:
        raise RuntimeError("Could not find model layers")

    target_layer = layers[layer_idx]
    hook = target_layer.register_forward_hook(hook_fn)

    try:
        effective_batch_size = min(batch_size, 4)
        total_batches = (len(examples) + effective_batch_size - 1) // effective_batch_size

        for i in tqdm(range(0, len(examples), effective_batch_size),
                     desc=f"Extracting layer {layer_idx}",
                     total=total_batches):
            batch_examples = examples[i:i+effective_batch_size]
            extracted_hidden.clear()

            texts = [format_prompt_for_example(ex, dataset_config) for ex in batch_examples]
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=dataset_config["max_length"]).to(device)

            with torch.no_grad():
                _ = model(**inputs, output_hidden_states=False)

            if not extracted_hidden:
                raise RuntimeError(f"Hook did not capture hidden states for layer {layer_idx}")

            hidden = extracted_hidden[0]
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=device)
            pooled = hidden[batch_indices, seq_lengths]
            all_hidden.append(pooled.cpu().float().numpy())

            del inputs, hidden, pooled
            extracted_hidden.clear()

            if (i // effective_batch_size) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    finally:
        hook.remove()

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

    # Skip GSM8K for linear probe (not classification)
    if dataset_key == "gsm8k":
        print(f"Skipping linear probe for GSM8K (not a classification task)")
        return {"dataset": dataset_key, "skipped": True, "reason": "not_classification"}

    print(f"\n{'='*60}")
    print(f"Linear Probe: {dataset_config['name']} | Seed: {seed}")
    print(f"{'='*60}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_data = load_dataset_by_config(dataset_key, dataset_config["split_train"],
                                        max_samples=config["max_samples"])
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"])

    train_labels = np.array([item["label"] for item in train_data])
    test_labels = np.array([item["label"] for item in test_data])

    # Extract hidden states
    X_train = extract_hidden_states(
        train_data, model, tokenizer, config["layer_idx"],
        dataset_config, config["batch_size"], device
    )
    X_test = extract_hidden_states(
        test_data, model, tokenizer, config["layer_idx"],
        dataset_config, config["batch_size"], device
    )

    # Normalize
    if config["normalize"] == "l2":
        X_train = sk_normalize(X_train, norm='l2', axis=1)
        X_test = sk_normalize(X_test, norm='l2', axis=1)

    # Train logistic regression
    clf = LogisticRegression(C=config["C"], max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X_train, train_labels)

    # Evaluate
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(test_labels, test_pred) * 100

    # Convert predictions to label names
    all_predictions = [dataset_config["label_names"][p] for p in test_pred]
    all_labels = [dataset_config["label_names"][l] for l in test_labels]

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "method": "linear_probe",
        "seed": seed,
        "layer_idx": config["layer_idx"],
        "accuracy": test_acc,
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
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Track experiment progress and estimate time remaining."""

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
        self.save_callback = save_callback

    def update(self, experiment_name: str, experiment_time: float, success: bool = True,
               error_msg: Optional[str] = None):
        """Record completion of an experiment."""
        self.completed += 1
        if not success:
            self.failed += 1

        self.experiment_times.append(experiment_time)

        log_entry = {
            "name": experiment_name,
            "time": experiment_time,
            "success": success,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
        self.experiment_log.append(log_entry)

        elapsed = time.time() - self.start_time
        avg_time = np.mean(self.experiment_times)
        remaining = self.total_experiments - self.completed
        eta_seconds = remaining * avg_time
        progress_pct = (self.completed / self.total_experiments) * 100 if self.total_experiments > 0 else 0

        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)

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
    # Linear probe (skip GSM8K)
    classification_datasets = [d for d in datasets if d != "gsm8k"]
    total += len(classification_datasets) * n_seeds
    # Bridge (per token config)
    total += n_tokens * n_datasets * n_seeds

    return total


# =============================================================================
# RESUME CAPABILITY
# =============================================================================

def get_result_file_path(run_dir: Path, method: str, dataset_key: str, seed: int,
                         num_tokens: Optional[int] = None, model_name: Optional[str] = None) -> Path:
    """Get the expected result file path for a given experiment configuration."""
    if method == "zeroshot":
        return run_dir / "zeroshot" / f"zeroshot_{model_name}_{dataset_key}_seed{seed}.json"
    elif method == "text_relay":
        return run_dir / "text_relay" / f"text_relay_{dataset_key}_seed{seed}.json"
    elif method == "linear_probe":
        return run_dir / "linear_probe" / f"linear_probe_{dataset_key}_seed{seed}.json"
    elif method == "bridge":
        return run_dir / "bridge" / f"{dataset_key}_seed{seed}_tokens{num_tokens}_results.json"
    else:
        raise ValueError(f"Unknown method: {method}")


def check_existing_result(result_path: Path) -> Optional[Dict[str, Any]]:
    """Check if a result file exists and load its contents if valid."""
    if not result_path.exists():
        return None

    try:
        with open(result_path, "r") as f:
            result = json.load(f)

        if "accuracy" in result or "skipped" in result:
            return result

        return None
    except (json.JSONDecodeError, IOError):
        return None


def load_existing_results(run_dir: Path, datasets: List[str], seeds: List[int]) -> Dict[str, List[Dict]]:
    """Load all existing results from a previous run for resume capability."""
    all_results = {
        "zeroshot_llama": [],
        "zeroshot_mistral": [],
        "text_relay": [],
        "linear_probe": [],
    }

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
    parser = argparse.ArgumentParser(description="Enhanced Telepathy Paper Evaluation")

    parser.add_argument("--output_dir", type=str, default="runs/enhanced_results",
                       help="Output directory for all results")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["sst2", "agnews", "trec", "boolq", "arc_easy", "gsm8k"],
                       help="Datasets to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                       help="Random seeds to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")

    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip experiments that already have results")
    parser.add_argument("--resume_dir", type=str, default=None,
                       help="Resume from a specific run directory")

    parser.add_argument("--skip_slow", type=str, nargs="*", default=None,
                       choices=["text_relay", "bridge", "zeroshot", "linear_probe"],
                       help="Skip slow experiment types")
    parser.add_argument("--fast_mode", action="store_true",
                       help="Enable fast mode with reduced training")

    parser.add_argument("--max_retries", type=int, default=2,
                       help="Maximum retries for failed experiments")
    parser.add_argument("--continue_on_error", action="store_true", default=True,
                       help="Continue with other experiments if one fails")

    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume mode
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        if not run_dir.exists():
            print(f"ERROR: Resume directory does not exist: {run_dir}")
            sys.exit(1)
        args.skip_existing = True
        timestamp = run_dir.name.replace("run_", "") if run_dir.name.startswith("run_") else datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"RESUMING from: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(run_dir, log_level)
    logger.info(f"Starting enhanced evaluation run: {run_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Apply fast mode settings
    if args.fast_mode:
        logger.info("FAST MODE enabled")
        BRIDGE_CONFIG["train_steps"] = 500
        LINEAR_PROBE_CONFIG["max_samples"] = 1000

    # Count experiments
    total_experiments = count_total_experiments(args.datasets, args.seeds)

    # Load existing results if resuming
    skipped_experiments = set()
    if args.skip_existing or args.resume_dir:
        print("\nLoading existing results...")
        all_results = load_existing_results(run_dir, args.datasets, args.seeds)
        # Build set of completed experiments
        for method, results_list in all_results.items():
            for r in results_list:
                ds = r.get("dataset", "")
                seed = r.get("seed", "")
                tokens = r.get("num_soft_tokens", "")
                model = r.get("model_name", "")
                if "bridge" in method:
                    skipped_experiments.add(f"bridge_{tokens}_{ds}_seed{seed}")
                elif "zeroshot" in method:
                    skipped_experiments.add(f"zeroshot_{model}_{ds}_seed{seed}")
                else:
                    skipped_experiments.add(f"{method}_{ds}_seed{seed}")
        print(f"  Found {len(skipped_experiments)} completed experiments")
    else:
        all_results = {
            "zeroshot_llama": [],
            "zeroshot_mistral": [],
            "text_relay": [],
            "linear_probe": [],
        }
        for num_tokens in TOKEN_ABLATION_CONFIGS:
            all_results[f"bridge_{num_tokens}"] = []

    # Determine which experiments to skip
    skip_slow_set = set(args.skip_slow) if args.skip_slow else set()

    # Initialize atomic checkpointer
    checkpointer = AtomicCheckpointer(run_dir, "enhanced_evaluation")
    checkpointer.load()

    # Save callback
    def save_intermediate_results():
        try:
            intermediate = {
                "timestamp": datetime.now().isoformat(),
                "status": "in_progress",
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

    # Initialize progress tracker
    tracker = ProgressTracker(total_experiments, logger=logger, save_callback=save_intermediate_results)
    failed_experiments = []

    print("="*70)
    print("ENHANCED TELEPATHY PAPER EVALUATION")
    print("="*70)
    print(f"Output directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Token configs: {TOKEN_ABLATION_CONFIGS}")
    print(f"Total experiments: {total_experiments}")
    print(f"Estimated time: ~12 hours on 1 H100")
    print("="*70)

    # Save configuration
    config = {
        "timestamp": timestamp,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "token_ablation": TOKEN_ABLATION_CONFIGS,
        "device": str(device),
        "bridge_config": BRIDGE_CONFIG,
        "linear_probe_config": LINEAR_PROBE_CONFIG,
        "total_experiments": total_experiments,
        "fast_mode": args.fast_mode,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    try:
        # Helper function for running experiments with retry
        def run_experiment_with_retry(func, exp_name, result_key, *func_args, **func_kwargs):
            nonlocal failed_experiments

            if args.skip_existing and exp_name in skipped_experiments:
                logger.info(f"[SKIP] {exp_name} - result already exists")
                tracker.record_skip(exp_name)
                return True, None

            aggressive_memory_cleanup(device, logger)

            exp_start = time.time()
            success, result, error_msg = run_with_retry(
                func, *func_args,
                max_retries=args.max_retries,
                retry_delay=5.0,
                experiment_name=exp_name,
                logger=logger,
                device=device,
                **func_kwargs
            )

            exp_time = time.time() - exp_start

            if success and result:
                all_results[result_key].append(result)
                checkpointer.mark_completed(exp_name, result)
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

        # 1. Zero-shot baselines
        if "zeroshot" not in skip_slow_set:
            zeroshot_dir = run_dir / "zeroshot"
            zeroshot_dir.mkdir(exist_ok=True)
            logger.info("Starting zero-shot baselines...")

            for dataset_key in args.datasets:
                for seed in args.seeds:
                    # Llama zero-shot
                    exp_name = f"zeroshot_llama_{dataset_key}_seed{seed}"
                    run_experiment_with_retry(
                        run_zeroshot_baseline,
                        exp_name,
                        "zeroshot_llama",
                        dataset_key, "llama", SOURCE_MODEL, seed, zeroshot_dir, device
                    )

                    # Mistral zero-shot
                    exp_name = f"zeroshot_mistral_{dataset_key}_seed{seed}"
                    run_experiment_with_retry(
                        run_zeroshot_baseline,
                        exp_name,
                        "zeroshot_mistral",
                        dataset_key, "mistral", TARGET_MODEL, seed, zeroshot_dir, device
                    )

        # 2. Text-relay baseline (improved)
        if "text_relay" not in skip_slow_set:
            relay_dir = run_dir / "text_relay"
            relay_dir.mkdir(exist_ok=True)
            logger.info("Starting improved text-relay baselines...")

            for dataset_key in args.datasets:
                for seed in args.seeds:
                    exp_name = f"text_relay_{dataset_key}_seed{seed}"
                    run_experiment_with_retry(
                        run_text_relay_baseline,
                        exp_name,
                        "text_relay",
                        dataset_key, seed, relay_dir, device
                    )

        # 3. Linear probe baseline
        if "linear_probe" not in skip_slow_set:
            probe_dir = run_dir / "linear_probe"
            probe_dir.mkdir(exist_ok=True)
            logger.info("Starting linear probe baselines...")

            for dataset_key in args.datasets:
                if dataset_key == "gsm8k":
                    continue  # Skip GSM8K for linear probe
                for seed in args.seeds:
                    exp_name = f"linear_probe_{dataset_key}_seed{seed}"
                    run_experiment_with_retry(
                        run_linear_probe,
                        exp_name,
                        "linear_probe",
                        dataset_key, seed, probe_dir, device, LINEAR_PROBE_CONFIG
                    )

        # 4. Bridge with token ablation
        if "bridge" not in skip_slow_set:
            bridge_dir = run_dir / "bridge"
            bridge_dir.mkdir(exist_ok=True)
            logger.info("Starting bridge experiments...")

            for num_tokens in TOKEN_ABLATION_CONFIGS:
                for dataset_key in args.datasets:
                    for seed in args.seeds:
                        exp_name = f"bridge_{num_tokens}_{dataset_key}_seed{seed}"
                        run_experiment_with_retry(
                            train_bridge_for_dataset,
                            exp_name,
                            f"bridge_{num_tokens}",
                            dataset_key, seed, bridge_dir, device, BRIDGE_CONFIG,
                            num_soft_tokens=num_tokens
                        )

        # Save complete results
        complete_results = {
            "timestamp": timestamp,
            "config": config,
            "failed_experiments": failed_experiments,
            "tracker_summary": tracker.get_summary(),
            "experiment_log": tracker.get_experiment_log(),
        }

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
        print("ENHANCED EVALUATION COMPLETE")
        print("="*70)
        print(f"Results saved to: {run_dir}")
        print(f"Total time: {tracker._format_time(total_time)}")
        print(f"Experiments completed: {summary['completed']}/{summary['total']}")
        print(f"Experiments failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")

        # Print accuracy summary
        print("\n--- Accuracy Summary ---")
        for dataset_key in args.datasets:
            print(f"\n{DATASETS[dataset_key]['name']}:")
            for method, results_list in all_results.items():
                method_results = [r for r in results_list if r.get("dataset") == dataset_key and not r.get("skipped")]
                if method_results:
                    accs = [r["accuracy"] for r in method_results]
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs) if len(accs) > 1 else 0
                    print(f"  {method}: {mean_acc:.1f}% +/- {std_acc:.1f}%")

        logger.info(f"Evaluation complete. Total time: {tracker._format_time(total_time)}")
        checkpointer.clear()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        logger.warning("Evaluation interrupted by user")

        try:
            partial_results = {
                "timestamp": timestamp,
                "config": config,
                "status": "interrupted",
                "failed_experiments": failed_experiments,
                "tracker_summary": tracker.get_summary(),
            }
            for method, results_list in all_results.items():
                serializable = []
                for r in results_list:
                    r_copy = {k: v for k, v in r.items() if k not in ["predictions", "labels"]}
                    serializable.append(r_copy)
                partial_results[method] = serializable

            save_partial_results_atomic(partial_results, run_dir / "partial_results.json", logger)
            print(f"Partial results saved to: {run_dir / 'partial_results.json'}")

        except Exception as e:
            print(f"Failed to save partial results: {e}")

    except ExperimentError as e:
        logger.error(f"Experiment error: {e}")
        print(f"\nExperiment failed: {e}")

    print(f"\nAll results: {run_dir}")


if __name__ == "__main__":
    main()
