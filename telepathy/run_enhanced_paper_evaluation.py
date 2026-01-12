#!/usr/bin/env python3
"""
Enhanced Telepathy Paper Evaluation Script

Builds on run_complete_evaluation.py with improvements:
1. 5 seeds (42, 123, 456, 789, 1011) for statistical power
2. Additional token ablations (16, 24) for granular analysis
3. Reasoning benchmarks (GSM8K, ARC-Easy, BoolQ) for broader coverage
4. Fixed text-relay implementation with improved prompting
5. Optimized for 12 hours on 1 H100 GPU

Experimental Design:
- Datasets: SST-2, AG News, TREC (classification), BoolQ, ARC-Easy, GSM8K (reasoning)
- Token Ablations: 8, 16, 24, 32 (4 configurations)
- Seeds: 42, 123, 456, 789, 1011 (5 seeds)
- Baselines: Bridge, Text-Relay, Llama 0-shot, Mistral 0-shot, Linear Probe

Time Budget (12 hours on 1 H100):
- Bridge training: ~6 hours (4 tokens × 5 seeds × 15 min)
- Classification eval: ~2 hours (3 datasets × 4 tokens × 5 seeds × 2 min)
- Reasoning eval: ~3 hours (3 datasets × 200 samples × 5 seeds × 3 min)
- Baselines & analysis: ~1 hour
Total: ~12 hours (with buffer)

Usage:
    python telepathy/run_enhanced_paper_evaluation.py --output_dir runs/enhanced_paper_eval

Author: Telepathy Project
Date: January 2025
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import GSM8K evaluation utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from latentwire.gsm8k_eval import (
    load_gsm8k_dataset,
    format_gsm8k_prompt,
    extract_numerical_answer,
    compute_gsm8k_accuracy,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Enhanced seed configuration - 5 seeds for statistical power
SEEDS = [42, 123, 456, 789, 1011]

# Enhanced token ablation - finer-grained analysis
TOKEN_ABLATION_CONFIGS = [8, 16, 24, 32]

# Model configurations
SOURCE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Classification datasets (reduced sample size for time budget)
CLASSIFICATION_DATASETS = {
    "sst2": {
        "name": "SST-2",
        "num_classes": 2,
        "split_train": "train",
        "split_test": "validation",
        "load_fn": "glue:sst2",
        "text_field": "sentence",
        "label_field": "label",
        "label_names": ["negative", "positive"],
        "prompt_template": "Review: {text}\nSentiment (positive or negative):",
        "primer": "Sentiment:",
        "max_length": 128,
        "eval_samples": 500,  # Reduced from full set for time
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
        "eval_samples": 500,
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
        "eval_samples": 500,
    },
}

# Reasoning benchmarks
REASONING_BENCHMARKS = {
    "boolq": {
        "name": "BoolQ",
        "num_choices": 2,
        "load_fn": "google/boolq",
        "split_test": "validation",
        "labels": ["No", "Yes"],
        "eval_samples": 200,
        "max_new_tokens": 10,
        "description": "Yes/No reading comprehension",
    },
    "arc_easy": {
        "name": "ARC-Easy",
        "num_choices": 4,
        "load_fn": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split_test": "test",
        "labels": ["A", "B", "C", "D"],
        "eval_samples": 200,
        "max_new_tokens": 10,
        "description": "Science QA (easy)",
    },
    "gsm8k": {
        "name": "GSM8K",
        "num_choices": None,  # Free-form numerical answer
        "load_fn": "openai/gsm8k",
        "config": "main",
        "split_test": "test",
        "labels": None,
        "eval_samples": 200,
        "max_new_tokens": 256,  # Need longer for CoT reasoning
        "description": "Grade school math",
    },
}

# Bridge configurations (optimized for 12h budget)
BRIDGE_CONFIG = {
    "depth": 2,
    "heads": 8,
    "source_layer": 31,
    "train_steps": 1000,  # Reduced from 1500 for time
    "batch_size": 16,
    "lr": 2e-4,
    "diversity_weight": 0.1,
    "target_rms": 0.03,
    "eval_interval": 200,
}

# Linear probe configurations
LINEAR_PROBE_CONFIG = {
    "layer_idx": 16,
    "normalize": "l2",
    "C": 1.0,
    "batch_size": 8,
    "max_samples": 3000,  # Reduced from 5000 for time
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logging with both file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"enhanced_evaluation_{timestamp}.log"

    logger = logging.getLogger("enhanced_eval")
    logger.setLevel(log_level)
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
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
        allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
        total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
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


def aggressive_memory_cleanup(device: torch.device, logger: Optional[logging.Logger] = None):
    """Perform aggressive memory cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    if logger:
        mem_info = get_gpu_memory_info(device)
        if mem_info.get("available"):
            logger.debug(
                f"After cleanup: {mem_info['allocated_gb']:.1f}GB allocated, "
                f"{mem_info['free_gb']:.1f}GB free"
            )


def save_checkpoint_atomic(data: Dict, path: Path, logger: Optional[logging.Logger] = None):
    """Save checkpoint atomically to prevent corruption."""
    temp_path = path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.replace(path)
        if logger:
            logger.debug(f"Saved checkpoint to {path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# =============================================================================
# BRIDGE COMPONENTS
# =============================================================================

class PerceiverResampler(nn.Module):
    """Perceiver-style cross-attention resampler."""

    def __init__(self, src_dim: int, tgt_dim: int, num_latents: int, heads: int, depth: int):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim

        self.latents = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)
        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
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


class TelepathyBridge(nn.Module):
    """Telepathy bridge for cross-model communication."""

    def __init__(self, src_dim: int, tgt_dim: int, num_soft_tokens: int,
                 heads: int, depth: int, target_rms: float):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.num_soft_tokens = num_soft_tokens
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_soft_tokens, heads, depth)
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward(self, src_hidden: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        compressed = self.resampler(src_hidden, src_mask)
        rms = torch.sqrt((compressed ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (compressed / rms) * self.output_scale
        z_variance = compressed.var(dim=[0, 1]).mean()
        return out, z_variance


# =============================================================================
# DATA LOADING
# =============================================================================

def load_classification_dataset(dataset_key: str, split: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load classification dataset."""
    config = CLASSIFICATION_DATASETS[dataset_key]

    load_fn = config["load_fn"]
    if ":" in load_fn:
        ds_name, ds_config = load_fn.split(":")
        dataset = load_dataset(ds_name, ds_config, split=split, trust_remote_code=True)
    else:
        dataset = load_dataset(load_fn, split=split, trust_remote_code=True)

    examples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        text = item[config["text_field"]]
        label = item[config["label_field"]]
        examples.append({"text": text, "label": label, "idx": idx})

    return examples


def load_reasoning_dataset(benchmark_key: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load reasoning benchmark dataset."""
    config = REASONING_BENCHMARKS[benchmark_key]

    if benchmark_key == "gsm8k":
        # Use specialized GSM8K loader
        return load_gsm8k_dataset(split="test", samples=max_samples)

    elif benchmark_key == "boolq":
        dataset = load_dataset(config["load_fn"], split=config["split_test"])
        examples = []
        for idx, item in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            text = f"Passage: {item['passage'][:500]}\n\nQuestion: {item['question']}"
            label = 1 if item["answer"] else 0
            examples.append({"text": text, "label": label, "idx": idx})
        return examples

    elif benchmark_key == "arc_easy":
        dataset = load_dataset(config["load_fn"], config["config"], split=config["split_test"])
        examples = []
        for idx, item in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            text = f"Question: {item['question']}\n\nChoices:\n"
            for label, choice in zip(item["choices"]["label"], item["choices"]["text"]):
                text += f"{label}. {choice}\n"
            label = item["choices"]["label"].index(item["answerKey"])
            examples.append({"text": text, "label": label, "idx": idx})
        return examples

    return []


# =============================================================================
# BRIDGE TRAINING
# =============================================================================

def train_bridge(
    dataset_key: str,
    num_soft_tokens: int,
    seed: int,
    src_model,
    tgt_model,
    src_tok,
    tgt_tok,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
) -> TelepathyBridge:
    """Train telepathy bridge for a specific dataset and token configuration."""

    logger.info(f"Training bridge: {dataset_key}, {num_soft_tokens} tokens, seed {seed}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load training data
    train_data = load_classification_dataset(
        dataset_key,
        CLASSIFICATION_DATASETS[dataset_key]["split_train"],
        max_samples=3000  # Limit training samples for time
    )

    # Initialize bridge
    src_dim = src_model.config.hidden_size
    tgt_dim = tgt_model.config.hidden_size

    bridge = TelepathyBridge(
        src_dim=src_dim,
        tgt_dim=tgt_dim,
        num_soft_tokens=num_soft_tokens,
        heads=BRIDGE_CONFIG["heads"],
        depth=BRIDGE_CONFIG["depth"],
        target_rms=BRIDGE_CONFIG["target_rms"],
    ).to(device)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=BRIDGE_CONFIG["lr"])

    # Training loop
    config = CLASSIFICATION_DATASETS[dataset_key]
    batch_size = BRIDGE_CONFIG["batch_size"]
    num_steps = BRIDGE_CONFIG["train_steps"]

    bridge.train()
    src_model.eval()
    tgt_model.eval()

    losses = []
    start_time = time.time()

    for step in tqdm(range(num_steps), desc=f"Training {dataset_key}-{num_soft_tokens}tok-seed{seed}"):
        # Sample batch
        batch_indices = np.random.choice(len(train_data), size=batch_size, replace=False)
        batch = [train_data[i] for i in batch_indices]

        # Prepare source inputs
        src_texts = [config["prompt_template"].format(text=item["text"][:config["max_length"]])
                     for item in batch]
        src_enc = src_tok(src_texts, return_tensors="pt", truncation=True,
                         max_length=config["max_length"], padding=True).to(device)

        # Get source hidden states
        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_hidden = src_out.hidden_states[BRIDGE_CONFIG["source_layer"]]

        # Forward through bridge
        soft_tokens, z_variance = bridge(src_hidden, src_enc.attention_mask)

        # Prepare target inputs (classification with primer)
        labels = [item["label"] for item in batch]
        label_texts = [config["label_names"][label] for label in labels]

        # Get primer embeddings
        primer_enc = tgt_tok(config["primer"], return_tensors="pt", add_special_tokens=False).to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
        primer_embeds = primer_embeds.expand(batch_size, -1, -1)

        # Combine primer + soft tokens
        combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)

        # Prepare labels for classification
        label_enc = tgt_tok(label_texts, return_tensors="pt", add_special_tokens=False, padding=True).to(device)

        # Attention mask
        attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)
        full_attn_mask = torch.cat([attn_mask, torch.ones_like(label_enc.input_ids)], dim=1)

        # Forward through target model
        outputs = tgt_model(
            inputs_embeds=combined_embeds,
            attention_mask=attn_mask,
            labels=label_enc.input_ids,
        )

        # Loss: CE + diversity regularization
        ce_loss = outputs.loss
        diversity_loss = -z_variance  # Encourage diverse representations
        loss = ce_loss + BRIDGE_CONFIG["diversity_weight"] * diversity_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        # Log progress
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            logger.debug(f"Step {step+1}/{num_steps}, Loss: {avg_loss:.4f}")

    train_time = time.time() - start_time
    logger.info(f"Training complete in {train_time:.1f}s, Final loss: {np.mean(losses[-100:]):.4f}")

    # Save bridge checkpoint
    checkpoint_dir = output_dir / "bridges" / f"{dataset_key}_{num_soft_tokens}tok_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "bridge_state_dict": bridge.state_dict(),
        "config": {
            "dataset_key": dataset_key,
            "num_soft_tokens": num_soft_tokens,
            "seed": seed,
            "train_steps": num_steps,
            "final_loss": np.mean(losses[-100:]),
            "train_time_sec": train_time,
        }
    }, checkpoint_dir / "bridge.pt")

    bridge.eval()
    return bridge


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_bridge_classification(
    dataset_key: str,
    bridge: TelepathyBridge,
    src_model,
    tgt_model,
    src_tok,
    tgt_tok,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Evaluate bridge on classification task."""

    config = CLASSIFICATION_DATASETS[dataset_key]
    test_data = load_classification_dataset(dataset_key, config["split_test"],
                                           max_samples=config["eval_samples"])

    predictions = []
    true_labels = []

    bridge.eval()
    src_model.eval()
    tgt_model.eval()

    with torch.no_grad():
        for item in tqdm(test_data, desc=f"Eval {dataset_key}"):
            # Prepare source input
            src_text = config["prompt_template"].format(text=item["text"][:config["max_length"]])
            src_enc = src_tok(src_text, return_tensors="pt", truncation=True,
                            max_length=config["max_length"]).to(device)

            # Get source hidden states
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_hidden = src_out.hidden_states[BRIDGE_CONFIG["source_layer"]]

            # Forward through bridge
            soft_tokens, _ = bridge(src_hidden, src_enc.attention_mask)

            # Get primer embeddings
            primer_enc = tgt_tok(config["primer"], return_tensors="pt",
                               add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)

            # Combine
            combined = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined.shape[:2], device=device, dtype=torch.long)

            # Generate
            outputs = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )

            # Decode and match to label
            response = tgt_tok.decode(outputs[0], skip_special_tokens=True).strip().lower()

            # Find matching label
            pred_label = None
            for i, label_name in enumerate(config["label_names"]):
                if label_name.lower() in response:
                    pred_label = i
                    break

            if pred_label is None:
                pred_label = 0  # Default to first label

            predictions.append(pred_label)
            true_labels.append(item["label"])

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)

    return {
        "accuracy": float(accuracy),
        "num_samples": len(test_data),
        "predictions": predictions,
        "true_labels": true_labels,
    }


def evaluate_text_relay_classification(
    dataset_key: str,
    src_model,
    tgt_model,
    src_tok,
    tgt_tok,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Evaluate improved text-relay baseline."""

    config = CLASSIFICATION_DATASETS[dataset_key]
    test_data = load_classification_dataset(dataset_key, config["split_test"],
                                           max_samples=config["eval_samples"])

    predictions = []
    true_labels = []

    src_model.eval()
    tgt_model.eval()

    with torch.no_grad():
        for item in tqdm(test_data, desc=f"Text-Relay {dataset_key}"):
            text = item["text"][:config["max_length"]]

            # Step 1: Llama summarizes with task-specific prompt
            summary_prompt = f"Summarize the key information for classification:\n\n{text}\n\nKey points:"
            src_enc = src_tok(summary_prompt, return_tensors="pt", truncation=True,
                            max_length=config["max_length"]).to(device)

            summary_ids = src_model.generate(
                **src_enc,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=src_tok.eos_token_id,
            )

            summary = src_tok.decode(summary_ids[0][src_enc.input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()

            # Step 2: Mistral classifies from summary
            classify_prompt = f"{config['primer']}\n\nContext: {summary}\n\nAnswer:"
            tgt_enc = tgt_tok(classify_prompt, return_tensors="pt", truncation=True,
                            max_length=256).to(device)

            output_ids = tgt_model.generate(
                **tgt_enc,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )

            response = tgt_tok.decode(output_ids[0][tgt_enc.input_ids.shape[1]:],
                                     skip_special_tokens=True).strip().lower()

            # Match to label
            pred_label = None
            for i, label_name in enumerate(config["label_names"]):
                if label_name.lower() in response:
                    pred_label = i
                    break

            if pred_label is None:
                pred_label = 0

            predictions.append(pred_label)
            true_labels.append(item["label"])

    accuracy = accuracy_score(true_labels, predictions)

    return {
        "accuracy": float(accuracy),
        "num_samples": len(test_data),
        "predictions": predictions,
        "true_labels": true_labels,
    }


def evaluate_zeroshot_reasoning(
    benchmark_key: str,
    model,
    tokenizer,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Evaluate zero-shot reasoning performance."""

    config = REASONING_BENCHMARKS[benchmark_key]
    test_data = load_reasoning_dataset(benchmark_key, max_samples=config["eval_samples"])

    predictions = []
    true_labels = []

    model.eval()

    with torch.no_grad():
        for item in tqdm(test_data, desc=f"0-shot {benchmark_key}"):
            if benchmark_key == "gsm8k":
                # GSM8K: Use 8-shot CoT prompting
                prompt = format_gsm8k_prompt(item["question"], few_shot=True)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=1024).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True)

                pred_answer = extract_numerical_answer(response)
                true_answer = item["numerical_answer"]

                predictions.append(pred_answer)
                true_labels.append(true_answer)

            elif benchmark_key == "boolq":
                prompt = f"{item['text']}\n\nAnswer with Yes or No:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=512).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True).strip().lower()

                # Match to Yes/No
                if "yes" in response:
                    pred_label = 1
                elif "no" in response:
                    pred_label = 0
                else:
                    pred_label = 0  # Default

                predictions.append(pred_label)
                true_labels.append(item["label"])

            elif benchmark_key == "arc_easy":
                prompt = f"{item['text']}\n\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=512).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True).strip().upper()

                # Match to A/B/C/D
                pred_label = None
                for i, label in enumerate(config["labels"]):
                    if label in response:
                        pred_label = i
                        break

                if pred_label is None:
                    pred_label = 0

                predictions.append(pred_label)
                true_labels.append(item["label"])

    # Compute accuracy
    if benchmark_key == "gsm8k":
        accuracy = compute_gsm8k_accuracy(predictions, true_labels)
    else:
        accuracy = accuracy_score(true_labels, predictions)

    return {
        "accuracy": float(accuracy),
        "num_samples": len(test_data),
        "predictions": predictions,
        "true_labels": true_labels,
    }


# =============================================================================
# MAIN EXPERIMENT ORCHESTRATION
# =============================================================================

def run_classification_experiments(
    src_model,
    tgt_model,
    src_tok,
    tgt_tok,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run all classification experiments."""

    results = {
        "bridge": defaultdict(lambda: defaultdict(list)),
        "text_relay": defaultdict(list),
        "llama_zeroshot": defaultdict(list),
        "mistral_zeroshot": defaultdict(list),
    }

    # 1. Bridge experiments: 3 datasets × 4 token configs × 5 seeds
    logger.info("=" * 80)
    logger.info("BRIDGE EXPERIMENTS")
    logger.info("=" * 80)

    for dataset_key in CLASSIFICATION_DATASETS.keys():
        for num_tokens in TOKEN_ABLATION_CONFIGS:
            for seed in SEEDS:
                # Train bridge
                bridge = train_bridge(
                    dataset_key, num_tokens, seed,
                    src_model, tgt_model, src_tok, tgt_tok,
                    device, output_dir, logger
                )

                # Evaluate bridge
                eval_result = evaluate_bridge_classification(
                    dataset_key, bridge, src_model, tgt_model,
                    src_tok, tgt_tok, device, logger
                )

                results["bridge"][dataset_key][num_tokens].append({
                    "seed": seed,
                    "accuracy": eval_result["accuracy"],
                    "num_samples": eval_result["num_samples"],
                })

                logger.info(f"{dataset_key}, {num_tokens}tok, seed{seed}: {eval_result['accuracy']:.3f}")

                # Clean up bridge
                del bridge
                aggressive_memory_cleanup(device, logger)

    # 2. Text-relay baseline: 3 datasets × 5 seeds
    logger.info("=" * 80)
    logger.info("TEXT-RELAY BASELINE")
    logger.info("=" * 80)

    for dataset_key in CLASSIFICATION_DATASETS.keys():
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            eval_result = evaluate_text_relay_classification(
                dataset_key, src_model, tgt_model,
                src_tok, tgt_tok, device, logger
            )

            results["text_relay"][dataset_key].append({
                "seed": seed,
                "accuracy": eval_result["accuracy"],
                "num_samples": eval_result["num_samples"],
            })

            logger.info(f"Text-Relay {dataset_key}, seed{seed}: {eval_result['accuracy']:.3f}")

    # 3. Zero-shot baselines: 3 datasets × 2 models × 5 seeds
    logger.info("=" * 80)
    logger.info("ZERO-SHOT BASELINES")
    logger.info("=" * 80)

    for dataset_key in CLASSIFICATION_DATASETS.keys():
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Llama 0-shot
            eval_result = evaluate_bridge_classification(
                dataset_key,
                None,  # No bridge - direct text
                src_model, src_model,  # Use Llama for both
                src_tok, src_tok,
                device, logger
            )
            results["llama_zeroshot"][dataset_key].append({
                "seed": seed,
                "accuracy": eval_result["accuracy"],
            })

            # Mistral 0-shot
            eval_result = evaluate_bridge_classification(
                dataset_key,
                None,
                tgt_model, tgt_model,  # Use Mistral for both
                tgt_tok, tgt_tok,
                device, logger
            )
            results["mistral_zeroshot"][dataset_key].append({
                "seed": seed,
                "accuracy": eval_result["accuracy"],
            })

    return results


def run_reasoning_experiments(
    src_model,
    tgt_model,
    src_tok,
    tgt_tok,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run reasoning benchmark experiments."""

    results = {
        "llama": defaultdict(list),
        "mistral": defaultdict(list),
    }

    logger.info("=" * 80)
    logger.info("REASONING BENCHMARKS")
    logger.info("=" * 80)

    for benchmark_key in REASONING_BENCHMARKS.keys():
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Llama
            eval_result = evaluate_zeroshot_reasoning(
                benchmark_key, src_model, src_tok, device, logger
            )
            results["llama"][benchmark_key].append({
                "seed": seed,
                "accuracy": eval_result["accuracy"],
                "num_samples": eval_result["num_samples"],
            })
            logger.info(f"Llama {benchmark_key}, seed{seed}: {eval_result['accuracy']:.3f}")

            # Mistral
            eval_result = evaluate_zeroshot_reasoning(
                benchmark_key, tgt_model, tgt_tok, device, logger
            )
            results["mistral"][benchmark_key].append({
                "seed": seed,
                "accuracy": eval_result["accuracy"],
                "num_samples": eval_result["num_samples"],
            })
            logger.info(f"Mistral {benchmark_key}, seed{seed}: {eval_result['accuracy']:.3f}")

    return results


def compute_aggregate_statistics(results: Dict) -> Dict:
    """Compute mean, std, and confidence intervals across seeds."""

    aggregated = {}

    for method, method_results in results.items():
        aggregated[method] = {}

        for dataset_or_config, config_results in method_results.items():
            if isinstance(config_results, dict):
                # Bridge results: nested by token config
                aggregated[method][dataset_or_config] = {}
                for token_config, seed_results in config_results.items():
                    accuracies = [r["accuracy"] for r in seed_results]
                    aggregated[method][dataset_or_config][token_config] = {
                        "mean": float(np.mean(accuracies)),
                        "std": float(np.std(accuracies, ddof=1)),
                        "min": float(np.min(accuracies)),
                        "max": float(np.max(accuracies)),
                        "n_seeds": len(accuracies),
                    }
            else:
                # Other baselines: flat list
                accuracies = [r["accuracy"] for r in config_results]
                aggregated[method][dataset_or_config] = {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies, ddof=1)),
                    "min": float(np.min(accuracies)),
                    "max": float(np.max(accuracies)),
                    "n_seeds": len(accuracies),
                }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Enhanced Paper Evaluation")
    parser.add_argument("--output_dir", type=str, default="runs/enhanced_paper_eval")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with reduced samples")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("ENHANCED PAPER EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Token configs: {TOKEN_ABLATION_CONFIGS}")
    logger.info(f"Classification datasets: {list(CLASSIFICATION_DATASETS.keys())}")
    logger.info(f"Reasoning benchmarks: {list(REASONING_BENCHMARKS.keys())}")

    # Device setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info(device)
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"Total memory: {mem_info['total_gb']:.1f}GB")

    # Load models
    logger.info("Loading models...")
    start_time = time.time()

    src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    src_tok.pad_token = src_tok.eos_token
    src_model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    src_model.eval()
    logger.info(f"Loaded {SOURCE_MODEL}")

    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tgt_tok.pad_token = tgt_tok.eos_token
    tgt_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tgt_model.eval()
    logger.info(f"Loaded {TARGET_MODEL}")
    logger.info(f"Model loading took {time.time() - start_time:.1f}s")

    # Run experiments
    total_start = time.time()

    # 1. Classification experiments
    classification_results = run_classification_experiments(
        src_model, tgt_model, src_tok, tgt_tok,
        device, output_dir, logger
    )

    # 2. Reasoning experiments
    reasoning_results = run_reasoning_experiments(
        src_model, tgt_model, src_tok, tgt_tok,
        device, output_dir, logger
    )

    total_time = time.time() - total_start

    # Compute aggregated statistics
    logger.info("=" * 80)
    logger.info("COMPUTING AGGREGATE STATISTICS")
    logger.info("=" * 80)

    classification_stats = compute_aggregate_statistics(classification_results)
    reasoning_stats = compute_aggregate_statistics(reasoning_results)

    # Save all results
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seeds": SEEDS,
            "token_configs": TOKEN_ABLATION_CONFIGS,
            "total_time_hours": total_time / 3600,
            "source_model": SOURCE_MODEL,
            "target_model": TARGET_MODEL,
        },
        "classification": {
            "raw": classification_results,
            "aggregated": classification_stats,
        },
        "reasoning": {
            "raw": reasoning_results,
            "aggregated": reasoning_stats,
        },
    }

    results_path = output_dir / "enhanced_evaluation_results.json"
    save_checkpoint_atomic(all_results, results_path, logger)

    # Print summary
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info("")
    logger.info("Classification Results (mean ± std):")
    for dataset_key in CLASSIFICATION_DATASETS.keys():
        logger.info(f"\n{dataset_key.upper()}:")
        for num_tokens in TOKEN_ABLATION_CONFIGS:
            stats = classification_stats["bridge"][dataset_key][num_tokens]
            logger.info(f"  Bridge ({num_tokens}tok): {stats['mean']:.3f} ± {stats['std']:.3f}")

        relay_stats = classification_stats["text_relay"][dataset_key]
        logger.info(f"  Text-Relay: {relay_stats['mean']:.3f} ± {relay_stats['std']:.3f}")

    logger.info("\nReasoning Results (mean ± std):")
    for benchmark_key in REASONING_BENCHMARKS.keys():
        logger.info(f"\n{benchmark_key.upper()}:")
        llama_stats = reasoning_stats["llama"][benchmark_key]
        mistral_stats = reasoning_stats["mistral"][benchmark_key]
        logger.info(f"  Llama: {llama_stats['mean']:.3f} ± {llama_stats['std']:.3f}")
        logger.info(f"  Mistral: {mistral_stats['mean']:.3f} ± {mistral_stats['std']:.3f}")

    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
