#!/usr/bin/env python3
"""
Ablation Experiments for LatentWire Paper Revision

Three experiment types:
1. Source Layer Ablation: Test extraction from layers 8, 12, 16, 20, 24, 28
2. Multi-Model Pairs: Test Qwen→Mistral, Llama→Gemma to prove generality
3. Training-Free Baseline: Linear projection without bridge training

Usage:
    # Source layer ablation (Priority 1)
    python telepathy/run_ablation_experiments.py --experiment layer_ablation --dataset agnews

    # Multi-model pairs (Priority 2)
    python telepathy/run_ablation_experiments.py --experiment model_pairs --dataset agnews

    # Training-free baseline (Priority 3)
    python telepathy/run_ablation_experiments.py --experiment training_free --dataset agnews

    # All experiments
    python telepathy/run_ablation_experiments.py --experiment all --dataset agnews
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import time
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET CONFIGS (same as run_unified_comparison.py)
# =============================================================================

DATASET_CONFIGS = {
    "sst2": {
        "hf_name": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "task_prompt": "Is the sentiment positive or negative?",
        "random_chance": 50.0,
    },
    "agnews": {
        "hf_name": ("ag_news",),
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Classify the topic: World, Sports, Business, or Sci/Tech.",
        "random_chance": 25.0,
    },
    "trec": {
        "hf_name": ("trec",),
        "text_field": "text",
        "label_field": "coarse_label",
        "label_map": {
            0: "abbreviation", 1: "entity", 2: "description",
            3: "human", 4: "location", 5: "number"
        },
        "num_classes": 6,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Classify the question type: abbreviation, entity, description, human, location, or number.",
        "random_chance": 16.7,
    },
}


# =============================================================================
# MODEL PAIR CONFIGS
# =============================================================================

MODEL_PAIRS = {
    "llama_mistral": {
        "sender_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "receiver_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Llama-8B → Mistral-7B",
    },
    "qwen_mistral": {
        "sender_id": "Qwen/Qwen2.5-7B-Instruct",
        "receiver_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Qwen-7B → Mistral-7B",
    },
    "llama_gemma": {
        "sender_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "receiver_id": "google/gemma-2-9b-it",
        "name": "Llama-8B → Gemma-9B",
    },
    "qwen_llama": {
        "sender_id": "Qwen/Qwen2.5-7B-Instruct",
        "receiver_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "name": "Qwen-7B → Llama-8B",
    },
}


# =============================================================================
# BRIDGE ARCHITECTURE (same as run_unified_comparison.py)
# =============================================================================

class PerceiverResampler(nn.Module):
    """Cross-attention based compression."""
    def __init__(self, src_dim, tgt_dim, num_latents=8, heads=8, depth=2):
        super().__init__()
        self.num_latents = num_latents
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

    def forward(self, src_hidden, src_mask=None):
        B = src_hidden.shape[0]
        keys = self.input_proj(src_hidden)
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            x = x + layer["cross_attn"](layer["ln1"](x), keys, keys, key_padding_mask=key_padding_mask)[0]
            x = x + layer["self_attn"](layer["ln2"](x), layer["ln2"](x), layer["ln2"](x))[0]
            x = x + layer["ffn"](layer["ln3"](x))
        return x


class UnifiedBridge(nn.Module):
    """Full bridge: Sender → Perceiver → Receiver."""
    def __init__(self, sender_dim, receiver_dim, num_tokens=8, depth=2, target_rms=0.03):
        super().__init__()
        self.perceiver = PerceiverResampler(sender_dim, receiver_dim, num_tokens, depth=depth)
        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_tokens = num_tokens

    def forward(self, sender_hidden, attention_mask=None):
        soft_tokens = self.perceiver(sender_hidden, attention_mask)
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        return (soft_tokens / rms) * self.output_scale


# =============================================================================
# TRAINING-FREE BASELINE
# =============================================================================

class LinearProjectionBaseline(nn.Module):
    """Training-free linear projection from sender to receiver embedding space."""
    def __init__(self, sender_dim, receiver_dim, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        # Random orthogonal projection (better than purely random)
        self.projection = nn.Linear(sender_dim, receiver_dim, bias=False)
        # Initialize with truncated SVD-style initialization
        nn.init.orthogonal_(self.projection.weight)
        # Freeze - this is training-free
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sender_hidden, attention_mask=None):
        """Pool sender hidden states and project to receiver space."""
        # Mean pool over sequence (simple but effective)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(sender_hidden.dtype)
            pooled = (sender_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = sender_hidden.mean(dim=1)

        # Project to receiver dimension (ensure same dtype)
        projected = self.projection(pooled.to(self.projection.weight.dtype))  # [B, receiver_dim]

        # Expand to num_tokens (replicate the same vector)
        soft_tokens = projected.unsqueeze(1).expand(-1, self.num_tokens, -1)

        # RMS normalize
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        return (soft_tokens / rms) * 0.03  # Target RMS


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(dataset_name, split="train", max_samples=None):
    """Load dataset."""
    config = DATASET_CONFIGS[dataset_name]
    if len(config["hf_name"]) == 2:
        ds = load_dataset(config["hf_name"][0], config["hf_name"][1], split=split)
    else:
        ds = load_dataset(config["hf_name"][0], split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def get_label_tokens(tokenizer, dataset_name):
    """Get token IDs for each label."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = {}
    for idx, label in config["label_map"].items():
        tokens = tokenizer.encode(label, add_special_tokens=False)
        label_tokens[idx] = tokens[0]
    return label_tokens


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_bridge(
    bridge, sender, sender_tok, receiver, receiver_tok,
    train_ds, dataset_name, device,
    steps=2000, batch_size=16, lr=2e-4, source_layer=16
):
    """Train bridge on classification task."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(receiver_tok, dataset_name)

    # Disable diversity loss for binary classification
    diversity_weight = 0.0 if config["num_classes"] <= 2 else 0.1

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    bridge.train()

    def collate_fn(batch):
        texts = [item[config["text_field"]] for item in batch]
        labels = [item[config["label_field"]] for item in batch]
        return texts, labels

    # Class-balanced sampling for binary classification
    if config["num_classes"] == 2:
        label_counts = {}
        for item in train_ds:
            lbl = item[config["label_field"]]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        weights = [1.0 / label_counts[item[config["label_field"]]] for item in train_ds]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               collate_fn=collate_fn, drop_last=True)
    else:
        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
    data_iter = iter(dataloader)

    losses = []
    pbar = tqdm(total=steps, desc=f"Training (layer={source_layer})")

    for step in range(steps):
        try:
            texts, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            texts, labels = next(data_iter)

        B = len(texts)

        # Format prompts
        if dataset_name == "sst2":
            src_texts = [f"Review: {t}\n\nClassify sentiment as positive or negative:" for t in texts]
        elif dataset_name == "agnews":
            src_texts = [f"Article: {t[:256]}\nTopic:" for t in texts]
        else:
            src_texts = [f"Question: {t}\nType:" for t in texts]

        sender_inputs = sender_tok(src_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=256)
        sender_inputs = {k: v.to(device) for k, v in sender_inputs.items()}

        with torch.no_grad():
            sender_out = sender(
                input_ids=sender_inputs["input_ids"],
                attention_mask=sender_inputs["attention_mask"],
                output_hidden_states=True
            )
            sender_hidden = sender_out.hidden_states[source_layer]

        soft_tokens = bridge(sender_hidden, sender_inputs["attention_mask"])

        # Build receiver prompts
        prompts = [f"\n{config['task_prompt']}\nAnswer:" for _ in range(B)]
        prompt_inputs = receiver_tok(prompts, return_tensors="pt", padding=True,
                                     add_special_tokens=False)
        prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"].to(device))

        inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)
        soft_mask = torch.ones(B, soft_tokens.shape[1], device=device)
        full_mask = torch.cat([soft_mask, prompt_inputs["attention_mask"].to(device)], dim=1)

        outputs = receiver(inputs_embeds=inputs_embeds, attention_mask=full_mask)
        logits = outputs.logits[:, -1, :]

        label_logits = torch.stack([logits[:, label_tokens[i]] for i in range(len(label_tokens))], dim=1)
        targets = torch.tensor(labels, device=device)
        ce_loss = F.cross_entropy(label_logits, targets)

        # Diversity loss (optional)
        if diversity_weight > 0:
            flat = soft_tokens.view(B, -1)
            flat_norm = F.normalize(flat, dim=1)
            similarity = (flat_norm @ flat_norm.T)
            off_diag = similarity - torch.eye(B, device=device)
            div_loss = (off_diag ** 2).mean()
            loss = ce_loss + diversity_weight * div_loss
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        pbar.update(1)

    pbar.close()
    return sum(losses[-100:]) / 100


def evaluate_bridge(
    bridge, sender, sender_tok, receiver, receiver_tok,
    eval_ds, dataset_name, device, source_layer=16, max_samples=200
):
    """Evaluate bridge accuracy."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(receiver_tok, dataset_name)

    bridge.eval()
    correct = 0
    total = 0

    eval_ds = eval_ds.select(range(min(max_samples, len(eval_ds))))

    with torch.no_grad():
        for item in tqdm(eval_ds, desc="Evaluating"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            if dataset_name == "sst2":
                src_text = f"Review: {text}\n\nClassify sentiment as positive or negative:"
            elif dataset_name == "agnews":
                src_text = f"Article: {text[:256]}\nTopic:"
            else:
                src_text = f"Question: {text}\nType:"

            sender_inputs = sender_tok(src_text, return_tensors="pt", truncation=True, max_length=256)
            sender_inputs = {k: v.to(device) for k, v in sender_inputs.items()}

            sender_out = sender(
                input_ids=sender_inputs["input_ids"],
                attention_mask=sender_inputs["attention_mask"],
                output_hidden_states=True
            )
            sender_hidden = sender_out.hidden_states[source_layer]

            soft_tokens = bridge(sender_hidden, sender_inputs["attention_mask"])

            prompt = f"\n{config['task_prompt']}\nAnswer:"
            prompt_inputs = receiver_tok(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"].to(device))

            inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)
            soft_mask = torch.ones(1, soft_tokens.shape[1], device=device)
            full_mask = torch.cat([soft_mask, prompt_inputs["attention_mask"].to(device)], dim=1)

            outputs = receiver(inputs_embeds=inputs_embeds, attention_mask=full_mask)
            logits = outputs.logits[:, -1, :]

            label_logits = torch.stack([logits[:, label_tokens[i]] for i in range(len(label_tokens))], dim=1)
            pred = label_logits.argmax(dim=1).item()

            if pred == label:
                correct += 1
            total += 1

    return 100.0 * correct / total


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_layer_ablation(args):
    """Run source layer ablation experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT: Source Layer Ablation")
    print("="*60)

    source_layers = [8, 12, 16, 20, 24, 28]
    datasets = args.datasets if args.datasets else ["agnews"]

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models once
    print("\nLoading models...")
    sender_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    receiver_id = "mistralai/Mistral-7B-Instruct-v0.3"

    sender_tok = AutoTokenizer.from_pretrained(sender_id)
    sender_tok.pad_token = sender_tok.eos_token
    sender = AutoModelForCausalLM.from_pretrained(
        sender_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sender.eval()

    receiver_tok = AutoTokenizer.from_pretrained(receiver_id)
    receiver_tok.pad_token = receiver_tok.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        receiver_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    receiver.eval()

    sender_dim = sender.config.hidden_size
    receiver_dim = receiver.config.hidden_size

    for dataset_name in datasets:
        print(f"\n--- Dataset: {dataset_name} ---")
        results[dataset_name] = {}

        train_ds = load_data(dataset_name, split=DATASET_CONFIGS[dataset_name]["train_split"])
        eval_ds = load_data(dataset_name, split=DATASET_CONFIGS[dataset_name]["eval_split"])

        for source_layer in source_layers:
            print(f"\n  Source Layer: {source_layer}")
            set_seed(args.seed)

            bridge = UnifiedBridge(sender_dim, receiver_dim, num_tokens=8)
            bridge = bridge.to(device=device, dtype=torch.bfloat16)

            train_bridge(
                bridge, sender, sender_tok, receiver, receiver_tok,
                train_ds, dataset_name, device,
                steps=args.steps, source_layer=source_layer
            )

            accuracy = evaluate_bridge(
                bridge, sender, sender_tok, receiver, receiver_tok,
                eval_ds, dataset_name, device, source_layer=source_layer
            )

            results[dataset_name][source_layer] = accuracy
            print(f"  Layer {source_layer}: {accuracy:.1f}%")

            # Save intermediate results
            with open(f"{args.output_dir}/layer_ablation_results.json", "w") as f:
                json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*60)
    print("LAYER ABLATION RESULTS")
    print("="*60)
    print(f"{'Layer':<8}", end="")
    for ds in datasets:
        print(f"{ds:<12}", end="")
    print()
    print("-" * (8 + 12 * len(datasets)))
    for layer in source_layers:
        print(f"{layer:<8}", end="")
        for ds in datasets:
            print(f"{results[ds][layer]:.1f}%{'':<7}", end="")
        print()

    return results


def run_model_pairs(args):
    """Run multi-model pairs experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT: Multi-Model Pairs")
    print("="*60)

    pairs_to_test = args.model_pairs if args.model_pairs else ["llama_mistral", "qwen_mistral"]
    datasets = args.datasets if args.datasets else ["agnews"]

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pair_name in pairs_to_test:
        if pair_name not in MODEL_PAIRS:
            print(f"Unknown model pair: {pair_name}, skipping")
            continue

        pair_config = MODEL_PAIRS[pair_name]
        print(f"\n--- Model Pair: {pair_config['name']} ---")
        results[pair_name] = {}

        # Load models
        print(f"  Loading sender: {pair_config['sender_id']}")
        sender_tok = AutoTokenizer.from_pretrained(pair_config["sender_id"])
        sender_tok.pad_token = sender_tok.eos_token
        sender = AutoModelForCausalLM.from_pretrained(
            pair_config["sender_id"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        sender.eval()

        print(f"  Loading receiver: {pair_config['receiver_id']}")
        receiver_tok = AutoTokenizer.from_pretrained(pair_config["receiver_id"])
        receiver_tok.pad_token = receiver_tok.eos_token
        receiver = AutoModelForCausalLM.from_pretrained(
            pair_config["receiver_id"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        receiver.eval()

        sender_dim = sender.config.hidden_size
        receiver_dim = receiver.config.hidden_size

        for dataset_name in datasets:
            print(f"\n  Dataset: {dataset_name}")
            set_seed(args.seed)

            train_ds = load_data(dataset_name, split=DATASET_CONFIGS[dataset_name]["train_split"])
            eval_ds = load_data(dataset_name, split=DATASET_CONFIGS[dataset_name]["eval_split"])

            bridge = UnifiedBridge(sender_dim, receiver_dim, num_tokens=8)
            bridge = bridge.to(device=device, dtype=torch.bfloat16)

            train_bridge(
                bridge, sender, sender_tok, receiver, receiver_tok,
                train_ds, dataset_name, device,
                steps=args.steps, source_layer=16
            )

            accuracy = evaluate_bridge(
                bridge, sender, sender_tok, receiver, receiver_tok,
                eval_ds, dataset_name, device, source_layer=16
            )

            results[pair_name][dataset_name] = accuracy
            print(f"  {pair_config['name']} on {dataset_name}: {accuracy:.1f}%")

        # Clear GPU memory before loading next pair
        del sender, receiver, sender_tok, receiver_tok
        torch.cuda.empty_cache()

        # Save intermediate results
        with open(f"{args.output_dir}/model_pairs_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_training_free(args):
    """Run training-free baseline experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT: Training-Free Baseline")
    print("="*60)

    datasets = args.datasets if args.datasets else ["agnews"]

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("\nLoading models...")
    sender_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    receiver_id = "mistralai/Mistral-7B-Instruct-v0.3"

    sender_tok = AutoTokenizer.from_pretrained(sender_id)
    sender_tok.pad_token = sender_tok.eos_token
    sender = AutoModelForCausalLM.from_pretrained(
        sender_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sender.eval()

    receiver_tok = AutoTokenizer.from_pretrained(receiver_id)
    receiver_tok.pad_token = receiver_tok.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(
        receiver_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    receiver.eval()

    sender_dim = sender.config.hidden_size
    receiver_dim = receiver.config.hidden_size

    # Create training-free baseline
    baseline = LinearProjectionBaseline(sender_dim, receiver_dim, num_tokens=8)
    baseline = baseline.to(device=device, dtype=torch.bfloat16)

    for dataset_name in datasets:
        print(f"\n--- Dataset: {dataset_name} ---")

        eval_ds = load_data(dataset_name, split=DATASET_CONFIGS[dataset_name]["eval_split"])

        # Evaluate with training-free baseline (no training!)
        accuracy = evaluate_bridge(
            baseline, sender, sender_tok, receiver, receiver_tok,
            eval_ds, dataset_name, device, source_layer=16
        )

        results[dataset_name] = {
            "training_free": accuracy,
            "random_chance": DATASET_CONFIGS[dataset_name]["random_chance"]
        }
        print(f"  Training-free: {accuracy:.1f}% (random chance: {DATASET_CONFIGS[dataset_name]['random_chance']:.1f}%)")

    # Save results
    with open(f"{args.output_dir}/training_free_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LatentWire Ablation Experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["layer_ablation", "model_pairs", "training_free", "all"],
                        help="Which experiment to run")
    parser.add_argument("--datasets", nargs="+", default=["agnews"],
                        choices=["sst2", "agnews", "trec"],
                        help="Datasets to evaluate on")
    parser.add_argument("--model_pairs", nargs="+", default=None,
                        choices=list(MODEL_PAIRS.keys()),
                        help="Model pairs to test (for model_pairs experiment)")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="runs/ablation",
                        help="Output directory for results")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLatentWire Ablation Experiments")
    print(f"Experiment: {args.experiment}")
    print(f"Datasets: {args.datasets}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")

    if args.experiment == "layer_ablation" or args.experiment == "all":
        run_layer_ablation(args)

    if args.experiment == "model_pairs" or args.experiment == "all":
        run_model_pairs(args)

    if args.experiment == "training_free" or args.experiment == "all":
        run_training_free(args)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
