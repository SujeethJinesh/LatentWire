#!/usr/bin/env python3
"""
Complete Telepathy Paper Evaluation Script

This orchestration script runs all experiments needed for the Telepathy paper:
1. Telepathy bridge evaluation on 7 datasets with 5 seeds
2. Linear probe baseline on all datasets
3. Production metrics (throughput, latency, memory)
4. Statistical tests comparing all methods
5. Paper-ready tables and figures

Usage:
    # Full evaluation
    python telepathy/run_complete_evaluation.py --output_dir runs/paper_results

    # Specific components
    python telepathy/run_complete_evaluation.py --only bridge --datasets sst2 agnews
    python telepathy/run_complete_evaluation.py --only linear_probe
    python telepathy/run_complete_evaluation.py --only production_metrics
    python telepathy/run_complete_evaluation.py --only statistical_tests
    python telepathy/run_complete_evaluation.py --only generate_tables

Author: Telepathy Project
Date: January 2025
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# CONFIGURATION
# =============================================================================

# Datasets to evaluate
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
    "imdb": {
        "name": "IMDB",
        "num_classes": 2,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "imdb",
        "text_field": "text",
        "label_field": "label",
        "label_names": ["negative", "positive"],
        "prompt_template": "Movie Review: {text}\nSentiment (positive or negative):",
        "primer": "Sentiment:",
        "max_length": 512,
    },
    "mnli": {
        "name": "MNLI",
        "num_classes": 3,
        "split_train": "train",
        "split_test": "validation_matched",
        "load_fn": "glue:mnli",
        "text_field": "premise",  # Will combine with hypothesis
        "label_field": "label",
        "label_names": ["entailment", "neutral", "contradiction"],
        "prompt_template": "Premise: {premise}\nHypothesis: {hypothesis}\nRelation (entailment, neutral, or contradiction):",
        "primer": "Relation:",
        "max_length": 256,
    },
    "20newsgroups": {
        "name": "20 Newsgroups",
        "num_classes": 20,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "SetFit/20_newsgroups",
        "text_field": "text",
        "label_field": "label",
        "label_names": None,  # Will be populated from dataset
        "prompt_template": "Article: {text}\nTopic:",
        "primer": "Topic:",
        "max_length": 512,
    },
    "banking77": {
        "name": "Banking77",
        "num_classes": 77,
        "split_train": "train",
        "split_test": "test",
        "load_fn": "PolyAI/banking77",
        "text_field": "text",
        "label_field": "label",
        "label_names": None,  # Will be populated from dataset
        "prompt_template": "Query: {text}\nIntent:",
        "primer": "Intent:",
        "max_length": 256,
    },
}

# Random seeds for reproducibility
SEEDS = [42, 123, 456, 789, 2024]

# Model configurations
SOURCE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Bridge configurations
BRIDGE_CONFIG = {
    "soft_tokens": 8,
    "depth": 2,
    "heads": 8,
    "source_layer": 31,
    "train_steps": 2000,
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
    "max_samples": 5000,  # For faster extraction
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

    # Extract examples
    examples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        # Handle different text field formats
        if dataset_key == "mnli":
            text = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}"
        else:
            text = item[config["text_field"]]

        label = item[config["label_field"]]
        examples.append({"text": text, "label": label, "idx": idx})

    return examples


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
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_soft_tokens, heads, depth)
        self.output_scale = torch.nn.Parameter(torch.tensor(target_rms))

    def forward(self, src_hidden: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        compressed = self.resampler(src_hidden, src_mask)
        rms = torch.sqrt((compressed ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (compressed / rms) * self.output_scale
        z_variance = compressed.var(dim=[0, 1]).mean()
        return out, z_variance


# =============================================================================
# BRIDGE TRAINING AND EVALUATION
# =============================================================================

def train_bridge_for_dataset(
    dataset_key: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train and evaluate bridge on a single dataset with a single seed."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Training Bridge: {dataset_config['name']} | Seed: {seed}")
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
        num_soft_tokens=config["soft_tokens"],
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

    training_log = []
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
        soft_tokens, z_var = bridge(src_h, src_enc.attention_mask)

        # Diversity loss
        flat_tokens = soft_tokens.reshape(B, -1).float()
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

        inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

        K = soft_tokens.shape[1]
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
    predictions = []

    test_subset = test_data[:min(500, len(test_data))]  # Eval on up to 500 samples

    for item in tqdm(test_subset, desc="Evaluating", leave=False):
        text = item["text"]
        label = dataset_config["label_names"][item["label"]]

        src_input = dataset_config["prompt_template"].format(text=text[:dataset_config["max_length"]])
        src_enc = src_tok(src_input, return_tensors="pt", truncation=True,
                         max_length=dataset_config["max_length"]).to(device)

        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[config["source_layer"]]
            soft_tokens, _ = bridge(src_h, src_enc.attention_mask)

            primer = dataset_config["primer"]
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)

            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
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
        is_correct = label.lower() in output
        if is_correct:
            correct += 1
        total += 1
        predictions.append({"label": label, "output": output, "correct": is_correct})

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Save checkpoint
    checkpoint_path = output_dir / f"{dataset_key}_seed{seed}_bridge.pt"
    torch.save(bridge.state_dict(), checkpoint_path)

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_classes": dataset_config["num_classes"],
        "config": config,
        "checkpoint_path": str(checkpoint_path),
        "predictions": predictions[:20],  # Save first 20 for inspection
    }

    # Save results
    results_path = output_dir / f"{dataset_key}_seed{seed}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[RESULT] {dataset_config['name']} seed={seed}: {accuracy:.1f}% ({correct}/{total})")

    # Cleanup
    del src_model, tgt_model, bridge
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
) -> np.ndarray:
    """Extract hidden states from a specific layer."""
    model.eval()
    all_hidden = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting layer {layer_idx}"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=device)
        pooled = hidden[batch_indices, seq_lengths]
        all_hidden.append(pooled.cpu().numpy())

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

    # Load data
    train_data = load_dataset_by_config(dataset_key, dataset_config["split_train"],
                                        max_samples=config["max_samples"])
    test_data = load_dataset_by_config(dataset_key, dataset_config["split_test"],
                                       max_samples=config["max_samples"])

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

    results = {
        "dataset": dataset_key,
        "dataset_name": dataset_config["name"],
        "seed": seed,
        "layer_idx": config["layer_idx"],
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "num_train": len(train_labels),
        "num_test": len(test_labels),
        "num_classes": dataset_config["num_classes"],
    }

    # Save results
    results_path = output_dir / f"{dataset_key}_seed{seed}_linear_probe.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[RESULT] Linear Probe {dataset_config['name']} seed={seed}: {test_acc:.1f}%")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# PRODUCTION METRICS
# =============================================================================

def run_production_metrics(output_dir: Path, device: torch.device) -> Dict[str, Any]:
    """Measure throughput, latency, and memory usage."""

    print(f"\n{'='*60}")
    print("Production Metrics: Throughput, Latency, Memory")
    print(f"{'='*60}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "metrics": {}
    }

    # Memory measurement
    print("\n--- Memory Analysis ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated() / 1024**2

    # Load source model
    src_model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    mem_after_src = torch.cuda.memory_allocated() / 1024**2

    # Load target model
    tgt_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    mem_after_tgt = torch.cuda.memory_allocated() / 1024**2

    # Load bridge
    bridge = TelepathyBridge(
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        num_soft_tokens=8,
        heads=8,
        depth=2,
        target_rms=0.03
    ).to(device).to(torch.bfloat16)
    mem_after_bridge = torch.cuda.memory_allocated() / 1024**2

    results["metrics"]["memory"] = {
        "source_model_mb": mem_after_src - mem_before,
        "target_model_mb": mem_after_tgt - mem_after_src,
        "bridge_mb": mem_after_bridge - mem_after_tgt,
        "total_mb": mem_after_bridge - mem_before,
    }

    print(f"  Source model: {results['metrics']['memory']['source_model_mb']:.0f} MB")
    print(f"  Target model: {results['metrics']['memory']['target_model_mb']:.0f} MB")
    print(f"  Bridge: {results['metrics']['memory']['bridge_mb']:.0f} MB")
    print(f"  Total: {results['metrics']['memory']['total_mb']:.0f} MB")

    # Latency measurement
    print("\n--- Latency Analysis ---")
    src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Warmup
    test_text = "This is a test sentence for latency measurement."
    for _ in range(3):
        src_enc = src_tok(test_text, return_tensors="pt").to(device)
        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[31]
            soft_tokens, _ = bridge(src_h, src_enc.attention_mask)

    # Measure
    latencies = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()

        src_enc = src_tok(test_text, return_tensors="pt").to(device)
        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[31]
            soft_tokens, _ = bridge(src_h, src_enc.attention_mask)

            primer_enc = tgt_tok("Answer:", return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            combined = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined.shape[:2], device=device)

            _ = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id
            )

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    results["metrics"]["latency"] = {
        "mean_ms": np.mean(latencies) * 1000,
        "std_ms": np.std(latencies) * 1000,
        "p50_ms": np.percentile(latencies, 50) * 1000,
        "p95_ms": np.percentile(latencies, 95) * 1000,
        "p99_ms": np.percentile(latencies, 99) * 1000,
    }

    print(f"  Mean: {results['metrics']['latency']['mean_ms']:.1f} ms")
    print(f"  P50: {results['metrics']['latency']['p50_ms']:.1f} ms")
    print(f"  P95: {results['metrics']['latency']['p95_ms']:.1f} ms")

    # Throughput (batch processing)
    print("\n--- Throughput Analysis ---")
    batch_sizes = [1, 2, 4, 8, 16]
    throughput_results = []

    for bs in batch_sizes:
        test_texts = [test_text] * bs

        torch.cuda.synchronize()
        start = time.perf_counter()

        src_enc = src_tok(test_texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[31]
            soft_tokens, _ = bridge(src_h, src_enc.attention_mask)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        throughput = bs / elapsed
        throughput_results.append({"batch_size": bs, "throughput_samples_per_s": throughput})
        print(f"  Batch {bs}: {throughput:.1f} samples/s")

    results["metrics"]["throughput"] = throughput_results

    # Count parameters
    bridge_params = sum(p.numel() for p in bridge.parameters())
    src_params = sum(p.numel() for p in src_model.parameters())
    tgt_params = sum(p.numel() for p in tgt_model.parameters())

    results["metrics"]["parameters"] = {
        "bridge": bridge_params,
        "source_model": src_params,
        "target_model": tgt_params,
        "bridge_percent_of_target": 100.0 * bridge_params / tgt_params,
    }

    print(f"\n--- Parameters ---")
    print(f"  Bridge: {bridge_params:,} ({results['metrics']['parameters']['bridge_percent_of_target']:.2f}% of target)")

    # Save results
    results_path = output_dir / "production_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Cleanup
    del src_model, tgt_model, bridge
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# STATISTICAL TESTING
# =============================================================================

def bootstrap_ci(scores: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(scores) < 3:
        # For small samples, use t-distribution based CI
        mean = np.mean(scores)
        if len(scores) == 1:
            return float(mean), float(mean)
        se = np.std(scores, ddof=1) / np.sqrt(len(scores))
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        margin = t_val * se
        return float(mean - margin), float(mean + margin)

    bootstrapped = np.random.choice(scores, size=(n_bootstrap, len(scores)), replace=True)
    means = bootstrapped.mean(axis=1)
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return lower, upper


def run_statistical_tests(results_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run statistical tests comparing methods."""

    print(f"\n{'='*60}")
    print("Statistical Analysis")
    print(f"{'='*60}")

    # Collect all results
    bridge_results = {}
    linear_probe_results = {}

    for json_file in results_dir.glob("*_results.json"):
        with open(json_file) as f:
            data = json.load(f)

        if "bridge" in json_file.name:
            key = (data["dataset"], data["seed"])
            bridge_results[key] = data["accuracy"]
        elif "linear_probe" in json_file.name:
            key = (data["dataset"], data["seed"])
            linear_probe_results[key] = data["test_accuracy"]

    # Aggregate by dataset
    analysis = {"timestamp": datetime.now().isoformat(), "comparisons": {}}

    for dataset_key in DATASETS.keys():
        bridge_scores = [v for (d, s), v in bridge_results.items() if d == dataset_key]
        probe_scores = [v for (d, s), v in linear_probe_results.items() if d == dataset_key]

        if not bridge_scores or not probe_scores:
            continue

        bridge_arr = np.array(bridge_scores)
        probe_arr = np.array(probe_scores)

        # Basic statistics
        bridge_mean, bridge_std = np.mean(bridge_arr), np.std(bridge_arr, ddof=1)
        probe_mean, probe_std = np.mean(probe_arr), np.std(probe_arr, ddof=1)

        # Bootstrap CI
        bridge_ci = bootstrap_ci(bridge_arr)
        probe_ci = bootstrap_ci(probe_arr)

        # Statistical test (paired t-test if same seeds)
        if len(bridge_arr) >= 3 and len(probe_arr) >= 3:
            if len(bridge_arr) == len(probe_arr):
                t_stat, p_value = stats.ttest_rel(bridge_arr, probe_arr)
            else:
                t_stat, p_value = stats.ttest_ind(bridge_arr, probe_arr)
        else:
            t_stat, p_value = np.nan, np.nan

        # Effect size (Cohen's d)
        if bridge_std > 0 or probe_std > 0:
            pooled_std = np.sqrt((bridge_std**2 + probe_std**2) / 2)
            cohens_d = (bridge_mean - probe_mean) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0

        analysis["comparisons"][dataset_key] = {
            "dataset_name": DATASETS[dataset_key]["name"],
            "num_classes": DATASETS[dataset_key]["num_classes"],
            "bridge": {
                "mean": bridge_mean,
                "std": bridge_std,
                "ci_95": bridge_ci,
                "n_seeds": len(bridge_arr),
            },
            "linear_probe": {
                "mean": probe_mean,
                "std": probe_std,
                "ci_95": probe_ci,
                "n_seeds": len(probe_arr),
            },
            "comparison": {
                "difference": bridge_mean - probe_mean,
                "relative_improvement": (bridge_mean - probe_mean) / probe_mean * 100 if probe_mean > 0 else 0,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05 if not np.isnan(p_value) else False,
                "cohens_d": cohens_d,
            }
        }

        print(f"\n{DATASETS[dataset_key]['name']}:")
        print(f"  Bridge: {bridge_mean:.1f}% +/- {bridge_std:.1f}%")
        print(f"  Linear Probe: {probe_mean:.1f}% +/- {probe_std:.1f}%")
        print(f"  Difference: {bridge_mean - probe_mean:+.1f}%")
        if not np.isnan(p_value):
            sig = "*" if p_value < 0.05 else ""
            print(f"  p-value: {p_value:.4f}{sig}")

    # Save results
    results_path = output_dir / "statistical_analysis.json"
    with open(results_path, "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_latex_tables(results_dir: Path, output_dir: Path) -> str:
    """Generate LaTeX tables for the paper."""

    print(f"\n{'='*60}")
    print("Generating LaTeX Tables")
    print(f"{'='*60}")

    # Load statistical analysis
    stats_path = results_dir / "statistical_analysis.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {"comparisons": {}}

    # Main results table
    latex = r"""
% Main Results Table
\begin{table}[t]
\centering
\caption{Classification accuracy (\%) on benchmark datasets. Results show mean $\pm$ std over 5 random seeds. Bold indicates best result. Statistical significance vs. Linear Probe: *p<0.05, **p<0.01, ***p<0.001.}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
Dataset & Classes & Random & Linear Probe & Telepathy (Ours) \\
\midrule
"""

    for dataset_key in ["sst2", "agnews", "trec", "imdb", "mnli", "20newsgroups", "banking77"]:
        if dataset_key not in stats["comparisons"]:
            continue

        data = stats["comparisons"][dataset_key]
        random_baseline = 100.0 / data["num_classes"]

        probe_str = f"{data['linear_probe']['mean']:.1f} $\\pm$ {data['linear_probe']['std']:.1f}"
        bridge_str = f"{data['bridge']['mean']:.1f} $\\pm$ {data['bridge']['std']:.1f}"

        # Add significance stars
        p = data["comparison"]["p_value"]
        if not np.isnan(p):
            if p < 0.001:
                bridge_str += "***"
            elif p < 0.01:
                bridge_str += "**"
            elif p < 0.05:
                bridge_str += "*"

        # Bold best result
        if data["bridge"]["mean"] > data["linear_probe"]["mean"]:
            bridge_str = f"\\textbf{{{bridge_str}}}"
        else:
            probe_str = f"\\textbf{{{probe_str}}}"

        latex += f"{data['dataset_name']} & {data['num_classes']} & {random_baseline:.1f} & {probe_str} & {bridge_str} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Production metrics table
    prod_path = results_dir / "production_metrics.json"
    if prod_path.exists():
        with open(prod_path) as f:
            prod = json.load(f)

        latex += r"""

% Production Metrics Table
\begin{table}[t]
\centering
\caption{Production metrics for Telepathy bridge on """ + prod["gpu_name"] + r""".}
\label{tab:production}
\begin{tabular}{lc}
\toprule
Metric & Value \\
\midrule
"""

        if "memory" in prod["metrics"]:
            latex += f"Total Memory & {prod['metrics']['memory']['total_mb']:.0f} MB \\\\\n"
            latex += f"Bridge Memory & {prod['metrics']['memory']['bridge_mb']:.0f} MB \\\\\n"

        if "latency" in prod["metrics"]:
            latex += f"Latency (P50) & {prod['metrics']['latency']['p50_ms']:.1f} ms \\\\\n"
            latex += f"Latency (P95) & {prod['metrics']['latency']['p95_ms']:.1f} ms \\\\\n"

        if "parameters" in prod["metrics"]:
            latex += f"Bridge Parameters & {prod['metrics']['parameters']['bridge']:,} \\\\\n"
            latex += f"Parameter Overhead & {prod['metrics']['parameters']['bridge_percent_of_target']:.2f}\\% \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Save tables
    tables_path = output_dir / "paper_tables.tex"
    with open(tables_path, "w") as f:
        f.write(latex)

    print(f"Tables saved to: {tables_path}")
    print(latex)

    return latex


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Complete Telepathy Paper Evaluation")

    parser.add_argument("--output_dir", type=str, default="runs/paper_results",
                       help="Output directory for all results")
    parser.add_argument("--only", type=str, choices=["bridge", "linear_probe", "production_metrics",
                                                     "statistical_tests", "generate_tables"],
                       help="Run only a specific component")
    parser.add_argument("--datasets", type=str, nargs="+",
                       default=["sst2", "agnews", "trec", "imdb", "mnli", "20newsgroups", "banking77"],
                       help="Datasets to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                       help="Random seeds to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip experiments that already have results")

    return parser.parse_args()


def estimate_runtime(datasets: List[str], seeds: List[int]) -> float:
    """Estimate total runtime in hours."""
    # Approximate times per dataset per seed (in minutes)
    dataset_times = {
        "sst2": 20,
        "agnews": 25,
        "trec": 15,
        "imdb": 40,
        "mnli": 45,
        "20newsgroups": 35,
        "banking77": 30,
    }

    bridge_time = sum(dataset_times.get(d, 30) for d in datasets) * len(seeds)
    linear_probe_time = len(datasets) * len(seeds) * 10  # ~10 min per dataset/seed
    production_time = 15  # ~15 minutes
    stats_time = 5  # ~5 minutes

    total_minutes = bridge_time + linear_probe_time + production_time + stats_time
    return total_minutes / 60


def main():
    args = parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("TELEPATHY COMPLETE PAPER EVALUATION")
    print("="*70)
    print(f"Output directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")

    estimated_hours = estimate_runtime(args.datasets, args.seeds)
    print(f"Estimated runtime: {estimated_hours:.1f} hours")
    print("="*70)

    # Save configuration
    config = {
        "timestamp": timestamp,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "device": str(device),
        "bridge_config": BRIDGE_CONFIG,
        "linear_probe_config": LINEAR_PROBE_CONFIG,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_results = {"bridge": [], "linear_probe": [], "production": None, "stats": None}

    try:
        # 1. Bridge evaluation
        if args.only is None or args.only == "bridge":
            bridge_dir = run_dir / "bridge"
            bridge_dir.mkdir(exist_ok=True)

            for dataset_key in args.datasets:
                for seed in args.seeds:
                    results_path = bridge_dir / f"{dataset_key}_seed{seed}_results.json"

                    if args.skip_existing and results_path.exists():
                        print(f"Skipping {dataset_key} seed={seed} (exists)")
                        with open(results_path) as f:
                            all_results["bridge"].append(json.load(f))
                        continue

                    try:
                        results = train_bridge_for_dataset(
                            dataset_key, seed, bridge_dir, device, BRIDGE_CONFIG
                        )
                        all_results["bridge"].append(results)
                    except Exception as e:
                        print(f"ERROR in bridge {dataset_key} seed={seed}: {e}")
                        traceback.print_exc()

        # 2. Linear probe baseline
        if args.only is None or args.only == "linear_probe":
            probe_dir = run_dir / "linear_probe"
            probe_dir.mkdir(exist_ok=True)

            for dataset_key in args.datasets:
                for seed in args.seeds:
                    results_path = probe_dir / f"{dataset_key}_seed{seed}_linear_probe.json"

                    if args.skip_existing and results_path.exists():
                        print(f"Skipping linear probe {dataset_key} seed={seed} (exists)")
                        with open(results_path) as f:
                            all_results["linear_probe"].append(json.load(f))
                        continue

                    try:
                        results = run_linear_probe(
                            dataset_key, seed, probe_dir, device, LINEAR_PROBE_CONFIG
                        )
                        all_results["linear_probe"].append(results)
                    except Exception as e:
                        print(f"ERROR in linear probe {dataset_key} seed={seed}: {e}")
                        traceback.print_exc()

        # 3. Production metrics
        if args.only is None or args.only == "production_metrics":
            try:
                all_results["production"] = run_production_metrics(run_dir, device)
            except Exception as e:
                print(f"ERROR in production metrics: {e}")
                traceback.print_exc()

        # 4. Statistical tests
        if args.only is None or args.only == "statistical_tests":
            try:
                all_results["stats"] = run_statistical_tests(run_dir, run_dir)
            except Exception as e:
                print(f"ERROR in statistical tests: {e}")
                traceback.print_exc()

        # 5. Generate tables
        if args.only is None or args.only == "generate_tables":
            try:
                generate_latex_tables(run_dir, run_dir)
            except Exception as e:
                print(f"ERROR in table generation: {e}")
                traceback.print_exc()

        # Save complete results
        with open(run_dir / "complete_results.json", "w") as f:
            # Convert non-serializable items
            serializable = {
                "timestamp": timestamp,
                "bridge_results": all_results["bridge"],
                "linear_probe_results": all_results["linear_probe"],
            }
            json.dump(serializable, f, indent=2)

        # Final summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Results saved to: {run_dir}")

        # Print summary table
        print("\n--- Summary ---")
        for dataset_key in args.datasets:
            bridge_accs = [r["accuracy"] for r in all_results["bridge"]
                         if r.get("dataset") == dataset_key]
            probe_accs = [r["test_accuracy"] for r in all_results["linear_probe"]
                        if r.get("dataset") == dataset_key]

            if bridge_accs:
                print(f"{DATASETS[dataset_key]['name']:15} Bridge: {np.mean(bridge_accs):.1f}% +/- {np.std(bridge_accs):.1f}%", end="")
            if probe_accs:
                print(f"  Probe: {np.mean(probe_accs):.1f}% +/- {np.std(probe_accs):.1f}%")
            else:
                print()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.")

    print(f"\nAll results: {run_dir}")


if __name__ == "__main__":
    main()
