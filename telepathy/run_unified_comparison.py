#!/usr/bin/env python3
"""
Unified Comparison Experiment Script with Multi-Seed Support

Runs ALL baselines needed for paper in ONE script:
1. Bridge (Llama→Mistral) - Main method
2. Prompt-Tuning (Mistral only) - Proves sender is essential
3. Text-Relay (Llama summarizes → Mistral classifies) - Latency baseline
4. Few-shot Prompting (5-shot) - In-context learning baseline
5. Zero-shot baselines (Llama, Mistral direct)

Features:
- Multi-seed experiments with automatic aggregation (mean, std, min, max)
- Backward compatible with single-seed mode
- Saves both per-seed and aggregated results to JSON

Output: JSON files with per-seed results and aggregated statistics

Usage:
    # Multi-seed mode (default: seeds=[42, 123, 456])
    python telepathy/run_unified_comparison.py --datasets sst2 agnews trec --output_dir runs/unified

    # Custom seeds
    python telepathy/run_unified_comparison.py --seeds 42 100 200 --datasets sst2

    # Single-seed mode (legacy)
    python telepathy/run_unified_comparison.py --seed 42 --datasets sst2
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import time
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Import Linear Probe baseline modules
from telepathy.linear_probe_baseline import LinearProbeBaseline, train_linear_probe, eval_linear_probe


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET CONFIGS
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
            0: "abbreviation",
            1: "entity",
            2: "description",
            3: "human",
            4: "location",
            5: "number"
        },
        "num_classes": 6,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Classify the question type: abbreviation, entity, description, human, location, or number.",
        "random_chance": 16.7,
    },
    # Task transfer datasets (same task as SST-2 for sentiment transfer experiments)
    "imdb": {
        "hf_name": ("imdb",),
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Is the sentiment positive or negative?",
        "random_chance": 50.0,
    },
    "yelp_polarity": {
        "hf_name": ("yelp_polarity",),
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Is the sentiment positive or negative?",
        "random_chance": 50.0,
    },
}


# =============================================================================
# BRIDGE ARCHITECTURE
# =============================================================================

class PerceiverResampler(nn.Module):
    """Cross-attention based compression (same as comprehensive_experiments.py)."""
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
        # RMS normalize and scale
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        return (soft_tokens / rms) * self.output_scale


class SoftPromptTuning(nn.Module):
    """Prompt tuning baseline - NO sender model."""
    def __init__(self, num_tokens, embed_dim, target_rms=0.03):
        super().__init__()
        self.num_tokens = num_tokens
        self.soft_prompts = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward(self, batch_size):
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        rms = torch.sqrt((prompts ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        return (prompts / rms) * self.output_scale


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
        # Get first token of the label
        tokens = tokenizer.encode(label, add_special_tokens=False)
        label_tokens[idx] = tokens[0]
    return label_tokens


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_bridge(
    bridge, sender, sender_tok, receiver, receiver_tok,
    train_ds, dataset_name, device,
    steps=2000, batch_size=16, lr=2e-4, source_layer=16, diversity_weight=None
):
    """Train bridge on classification task.

    FIXED BUGS:
    - Use sender_tok for sender (was using receiver_tok - CRITICAL BUG)
    - Proper batching with DataLoader (was batch_size=1 effectively)
    - Added diversity loss to prevent mode collapse (DISABLED for binary classification)
    - Added gradient clipping for stability
    - Increased LR from 1e-4 to 2e-4
    - Added source prompt context

    SST-2 FIX (Dec 2024):
    - Diversity loss is COUNTERPRODUCTIVE for binary classification
    - With only 2 output classes, forcing orthogonal soft tokens hurts learning
    - Now: diversity_weight = 0.0 for num_classes <= 2, else 0.1
    """
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(receiver_tok, dataset_name)

    # CRITICAL FIX: Disable diversity loss for binary classification
    # Binary tasks have low output dimensionality - diversity loss conflicts with learning
    if diversity_weight is None:
        diversity_weight = 0.0 if config["num_classes"] <= 2 else 0.1
    print(f"  [train_bridge] num_classes={config['num_classes']}, diversity_weight={diversity_weight}")

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    bridge.train()

    # Use DataLoader for proper batching
    def collate_fn(batch):
        texts = [item[config["text_field"]] for item in batch]
        labels = [item[config["label_field"]] for item in batch]
        return texts, labels

    # CLASS-BALANCED SAMPLING for binary classification (SST-2 FIX)
    # Ensures each batch has roughly equal class distribution
    if config["num_classes"] == 2:
        from torch.utils.data import WeightedRandomSampler
        # Count labels
        label_counts = {}
        for item in train_ds:
            lbl = item[config["label_field"]]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        # Compute weights (inverse frequency)
        weights = [1.0 / label_counts[item[config["label_field"]]] for item in train_ds]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               collate_fn=collate_fn, drop_last=True)
        print(f"  [train_bridge] Using class-balanced sampling for binary classification")
    else:
        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
    data_iter = iter(dataloader)

    losses = []
    step = 0
    pbar = tqdm(total=steps, desc=f"Training Bridge on {dataset_name}")

    while step < steps:
        try:
            texts, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            texts, labels = next(data_iter)

        B = len(texts)

        # Format sender prompts with task context (CRITICAL FIX)
        # SST-2 prompt improved based on analysis - clearer directive format
        if dataset_name == "sst2":
            src_texts = [f"Review: {t}\n\nClassify sentiment as positive or negative:" for t in texts]
        elif dataset_name == "agnews":
            src_texts = [f"Article: {t[:256]}\nTopic:" for t in texts]
        else:
            src_texts = [f"Question: {t}\nType:" for t in texts]

        # Use SENDER tokenizer for sender model (CRITICAL BUG FIX - was using receiver_tok)
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

        # Get soft tokens for batch
        soft_tokens = bridge(sender_hidden, sender_inputs["attention_mask"])  # [B, K, D]

        # Build receiver prompts
        prompts = [f"\n{config['task_prompt']}\nAnswer:" for _ in range(B)]
        prompt_inputs = receiver_tok(prompts, return_tensors="pt", padding=True,
                                     add_special_tokens=False)
        prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"].to(device))

        # Concatenate soft tokens + prompts
        inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)

        # Create attention mask
        soft_mask = torch.ones(B, soft_tokens.shape[1], device=device)
        full_mask = torch.cat([soft_mask, prompt_inputs["attention_mask"].to(device)], dim=1)

        # Forward through receiver
        outputs = receiver(inputs_embeds=inputs_embeds, attention_mask=full_mask)
        logits = outputs.logits[:, -1, :]  # [B, vocab_size]

        # Classification loss
        label_logits = torch.stack([logits[:, label_tokens[i]] for i in range(len(label_tokens))], dim=1)
        targets = torch.tensor(labels, device=device)
        ce_loss = F.cross_entropy(label_logits, targets)

        # Diversity loss (CRITICAL FIX - prevents mode collapse)
        div_loss = torch.tensor(0.0, device=device)
        if diversity_weight > 0 and B > 1:
            flat = soft_tokens.reshape(B, -1).float()
            flat_norm = F.normalize(flat, dim=1)
            sim = torch.mm(flat_norm, flat_norm.t())
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            div_loss = sim[mask].mean()

        loss = ce_loss + diversity_weight * div_loss

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (CRITICAL FIX - prevents instability)
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        losses.append(ce_loss.item())
        step += 1
        pbar.update(1)

        if step % 100 == 0:
            pbar.set_postfix({"loss": np.mean(losses[-100:]), "div": div_loss.item()})

    pbar.close()
    return {"final_loss": np.mean(losses[-100:])}


def train_prompt_tuning(
    prompt_module, receiver, receiver_tok,
    train_ds, dataset_name, device,
    steps=2000, batch_size=8, lr=2e-4
):
    """Train prompt tuning baseline (no sender)."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(receiver_tok, dataset_name)

    optimizer = torch.optim.AdamW(prompt_module.parameters(), lr=lr)
    prompt_module.train()

    indices = list(range(len(train_ds)))
    random.shuffle(indices)

    losses = []
    step = 0
    pbar = tqdm(total=steps, desc=f"Training Prompt-Tuning on {dataset_name}")

    while step < steps:
        for idx in indices:
            if step >= steps:
                break

            item = train_ds[idx]
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            # Get soft prompts
            soft_tokens = prompt_module(batch_size=1)

            # Build receiver prompt (include text since no sender)
            prompt = f"\nText: {text[:256]}\n{config['task_prompt']}\nAnswer:"
            prompt_inputs = receiver_tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=256)
            prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"].to(device))

            # Concatenate soft prompts + text prompt
            inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)

            outputs = receiver(inputs_embeds=inputs_embeds)
            logits = outputs.logits[0, -1]

            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            target = torch.tensor([label], device=device)

            loss = F.cross_entropy(label_logits.unsqueeze(0), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1
            pbar.update(1)

            if step % 100 == 0:
                pbar.set_postfix({"loss": np.mean(losses[-100:])})

        random.shuffle(indices)

    pbar.close()
    return {"final_loss": np.mean(losses[-100:])}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def eval_bridge(bridge, sender, sender_tok, receiver, receiver_tok, eval_ds, dataset_name, device, source_layer=16):
    """Evaluate trained bridge.

    FIXED: Use sender_tok for sender (was using receiver_tok - CRITICAL BUG)
    """
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(receiver_tok, dataset_name)
    inv_label_map = {v.lower(): k for k, v in config["label_map"].items()}

    bridge.eval()
    correct = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for item in tqdm(eval_ds, desc=f"Eval Bridge on {dataset_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            start = time.time()

            # Format sender prompt with task context
            # SST-2 prompt improved - clearer directive format (matches train_bridge)
            if dataset_name == "sst2":
                src_text = f"Review: {text}\n\nClassify sentiment as positive or negative:"
            elif dataset_name == "agnews":
                src_text = f"Article: {text[:256]}\nTopic:"
            else:
                src_text = f"Question: {text}\nType:"

            # Use SENDER tokenizer for sender model (CRITICAL BUG FIX)
            sender_inputs = sender_tok(src_text, return_tensors="pt", truncation=True, max_length=256)
            sender_inputs = {k: v.to(device) for k, v in sender_inputs.items()}
            sender_out = sender(
                input_ids=sender_inputs["input_ids"],
                attention_mask=sender_inputs["attention_mask"],
                output_hidden_states=True
            )
            sender_hidden = sender_out.hidden_states[source_layer]

            # Bridge
            soft_tokens = bridge(sender_hidden, sender_inputs["attention_mask"])

            # Receiver
            prompt = f"\n{config['task_prompt']}\nAnswer:"
            prompt_inputs = receiver_tok(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"].to(device))
            inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)

            # Create attention mask
            soft_mask = torch.ones(1, soft_tokens.shape[1], device=device)
            full_mask = torch.cat([soft_mask, prompt_inputs["attention_mask"].to(device)], dim=1)

            outputs = receiver(inputs_embeds=inputs_embeds, attention_mask=full_mask)
            logits = outputs.logits[0, -1]

            latencies.append(time.time() - start)

            # Predict
            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            pred = label_logits.argmax().item()

            if pred == label:
                correct += 1
            total += 1

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "latency_ms": np.mean(latencies) * 1000,
    }


def eval_prompt_tuning(prompt_module, receiver, receiver_tok, eval_ds, dataset_name, device):
    """Evaluate prompt tuning baseline."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(receiver_tok, dataset_name)

    prompt_module.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(eval_ds, desc=f"Eval Prompt-Tuning on {dataset_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            soft_tokens = prompt_module(batch_size=1)

            prompt = f"\nText: {text[:256]}\n{config['task_prompt']}\nAnswer:"
            prompt_inputs = receiver_tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=256)
            prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"].to(device))
            inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)

            outputs = receiver(inputs_embeds=inputs_embeds)
            logits = outputs.logits[0, -1]

            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            pred = label_logits.argmax().item()

            if pred == label:
                correct += 1
            total += 1

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
    }


def eval_text_relay(sender, sender_tok, receiver, receiver_tok, eval_ds, dataset_name, device):
    """Evaluate text-relay baseline: Llama summarizes → text → Mistral classifies."""
    config = DATASET_CONFIGS[dataset_name]
    inv_label_map = {v.lower(): k for k, v in config["label_map"].items()}

    correct = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for item in tqdm(eval_ds, desc=f"Eval Text-Relay on {dataset_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            start = time.time()

            # Sender generates summary
            summary_prompt = f"Summarize this in one sentence:\n\n{text[:256]}\n\nSummary:"
            inputs = sender_tok(summary_prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = sender.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=sender_tok.eos_token_id,
            )
            summary = sender_tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            # Receiver classifies from summary
            classify_prompt = f"{config['task_prompt']}\n\nText: {summary}\n\nAnswer:"
            inputs = receiver_tok(classify_prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = receiver.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=receiver_tok.eos_token_id,
            )
            response = receiver_tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()

            latencies.append(time.time() - start)

            # Parse prediction
            pred = None
            for lbl_name, lbl_idx in inv_label_map.items():
                if lbl_name in response:
                    pred = lbl_idx
                    break

            if pred == label:
                correct += 1
            total += 1

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "latency_ms": np.mean(latencies) * 1000,
    }


def eval_zeroshot(model, tokenizer, eval_ds, dataset_name, device, model_name="model"):
    """Evaluate zero-shot baseline."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(tokenizer, dataset_name)

    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(eval_ds, desc=f"Eval {model_name} Zero-shot on {dataset_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            prompt = f"Text: {text[:256]}\n\n{config['task_prompt']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[0, -1]

            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            pred = label_logits.argmax().item()

            if pred == label:
                correct += 1
            total += 1

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
    }


def eval_fewshot(model, tokenizer, train_ds, eval_ds, dataset_name, device, shots=5, model_name="model"):
    """Evaluate few-shot baseline."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(tokenizer, dataset_name)

    # Sample balanced examples
    examples_by_class = {i: [] for i in range(config["num_classes"])}
    for item in train_ds:
        lbl = item[config["label_field"]]
        if len(examples_by_class[lbl]) < shots // config["num_classes"] + 1:
            examples_by_class[lbl].append(item)

    examples = []
    for i in range(config["num_classes"]):
        examples.extend(examples_by_class[i][:shots // config["num_classes"]])
    random.shuffle(examples)

    # Build few-shot prefix
    prefix = f"{config['task_prompt']}\n\n"
    for ex in examples[:shots]:
        ex_text = ex[config["text_field"]][:128]
        ex_label = config["label_map"][ex[config["label_field"]]]
        prefix += f"Text: {ex_text}\nLabel: {ex_label}\n\n"

    correct = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(eval_ds, desc=f"Eval {model_name} {shots}-shot on {dataset_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            prompt = f"{prefix}Text: {text[:256]}\nLabel:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[0, -1]

            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            pred = label_logits.argmax().item()

            if pred == label:
                correct += 1
            total += 1

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
    }


def eval_ensemble(model1, tok1, model2, tok2, eval_ds, dataset_name, device,
                  model1_name="model1", model2_name="model2"):
    """
    Evaluate ensemble baseline: run both models and combine predictions.

    This addresses the reviewer concern: "Why not just run both models and combine?"
    If Bridge doesn't beat this trivial ensemble, the bridge is unnecessary.
    """
    config = DATASET_CONFIGS[dataset_name]
    label_tokens1 = get_label_tokens(tok1, dataset_name)
    label_tokens2 = get_label_tokens(tok2, dataset_name)

    correct_m1 = 0
    correct_m2 = 0
    correct_vote = 0
    correct_avg = 0
    total = 0

    with torch.no_grad():
        for item in tqdm(eval_ds, desc=f"Eval Ensemble on {dataset_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            prompt = f"Text: {text[:256]}\n\n{config['task_prompt']}\nAnswer:"

            # Model 1 prediction
            inputs1 = tok1(prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            outputs1 = model1(**inputs1)
            logits1 = outputs1.logits[0, -1]
            label_logits1 = torch.stack([logits1[label_tokens1[i]] for i in range(len(label_tokens1))])
            probs1 = F.softmax(label_logits1, dim=0)
            pred1 = label_logits1.argmax().item()

            # Model 2 prediction
            inputs2 = tok2(prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            outputs2 = model2(**inputs2)
            logits2 = outputs2.logits[0, -1]
            label_logits2 = torch.stack([logits2[label_tokens2[i]] for i in range(len(label_tokens2))])
            probs2 = F.softmax(label_logits2, dim=0)
            pred2 = label_logits2.argmax().item()

            # Voting: majority vote (if tie, use model1)
            if pred1 == pred2:
                vote_pred = pred1
            else:
                # Use confidence to break tie
                if probs1.max() >= probs2.max():
                    vote_pred = pred1
                else:
                    vote_pred = pred2

            # Average probabilities
            avg_probs = (probs1 + probs2) / 2
            avg_pred = avg_probs.argmax().item()

            if pred1 == label:
                correct_m1 += 1
            if pred2 == label:
                correct_m2 += 1
            if vote_pred == label:
                correct_vote += 1
            if avg_pred == label:
                correct_avg += 1
            total += 1

    return {
        f"{model1_name}_accuracy": 100.0 * correct_m1 / total,
        f"{model2_name}_accuracy": 100.0 * correct_m2 / total,
        "vote_accuracy": 100.0 * correct_vote / total,
        "avg_accuracy": 100.0 * correct_avg / total,
        "total": total,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sst2", "agnews", "trec"])
    parser.add_argument("--output_dir", default="runs/unified_comparison")
    parser.add_argument("--sender", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--receiver", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--fewshot_shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Single seed (legacy mode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Multiple seeds for robust evaluation (default: [42, 123, 456])")
    parser.add_argument("--skip_text_relay", action="store_true", help="Skip slow text-relay baseline")
    parser.add_argument("--skip_fewshot", action="store_true", help="Skip few-shot baseline")
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse direction: use Mistral as sender, Llama as receiver")
    parser.add_argument("--run_ensemble", action="store_true",
                        help="Run ensemble baseline (both models vote)")
    args = parser.parse_args()

    # Handle seed/seeds argument compatibility
    if args.seeds is None:
        # If --seeds not specified, check if --seed was explicitly set
        # Default to multi-seed unless user explicitly set --seed
        import sys
        if "--seed" in sys.argv:
            # User explicitly set --seed, use single-seed mode
            seeds = [args.seed]
        else:
            # Default multi-seed mode
            seeds = [42, 123, 456]
    else:
        seeds = args.seeds

    print(f"Running with seeds: {seeds}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle reverse direction (swap sender and receiver)
    if args.reverse:
        sender_model_id = args.receiver  # Mistral becomes sender
        receiver_model_id = args.sender  # Llama becomes receiver
        direction_str = "REVERSE (Mistral→Llama)"
    else:
        sender_model_id = args.sender
        receiver_model_id = args.receiver
        direction_str = "FORWARD (Llama→Mistral)"

    print("=" * 70)
    print("UNIFIED COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Direction: {direction_str}")
    print(f"Sender: {sender_model_id}")
    print(f"Receiver: {receiver_model_id}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {seeds} ({len(seeds)} runs per experiment)")
    print(f"Soft tokens: {args.soft_tokens}")
    print(f"Train steps: {args.train_steps}")
    print(f"Eval samples: {args.eval_samples}")
    print(f"Output: {args.output_dir}")
    if args.run_ensemble:
        print("Ensemble baseline: ENABLED")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("\nLoading models...")
    sender = AutoModelForCausalLM.from_pretrained(
        sender_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sender_tok = AutoTokenizer.from_pretrained(sender_model_id)
    sender_tok.pad_token = sender_tok.eos_token

    receiver = AutoModelForCausalLM.from_pretrained(
        receiver_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    receiver_tok = AutoTokenizer.from_pretrained(receiver_model_id)
    receiver_tok.pad_token = receiver_tok.eos_token

    sender.eval()
    receiver.eval()

    # Results container for multi-seed runs
    all_seeds_results = {
        "meta": {
            "timestamp": timestamp,
            "direction": direction_str,
            "sender": sender_model_id,
            "receiver": receiver_model_id,
            "soft_tokens": args.soft_tokens,
            "train_steps": args.train_steps,
            "eval_samples": args.eval_samples,
            "seeds": seeds,
            "ensemble_enabled": args.run_ensemble,
        },
        "per_seed_results": {},  # Will store results for each seed
        "aggregated_results": {},  # Will store mean and std across seeds
        "comparison_table": {},
    }

    sender_dim = sender.config.hidden_size
    receiver_dim = receiver.config.hidden_size

    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")

        config = DATASET_CONFIGS[dataset_name]

        # Load data
        train_ds = load_data(dataset_name, config["train_split"], max_samples=5000)
        eval_ds = load_data(dataset_name, config["eval_split"], max_samples=args.eval_samples)

        # Store results for each seed
        all_seeds_results["per_seed_results"][dataset_name] = {}
        seed_results_list = {
            "bridge": [],
            "prompt_tuning": [],
            "linear_probe": [],  # Added Linear Probe baseline
            "text_relay": [],
            "llama_zeroshot": [],
            "mistral_zeroshot": [],
            "mistral_fewshot": [],
            "ensemble": [],
        }

        # ADAPTIVE HYPERPARAMETERS for binary classification (SST-2 FIX)
        # Binary tasks need: fewer tokens (4 vs 8), higher LR (5e-4 vs 2e-4), more steps (4000 vs 2000)
        # Also use layer 24 for sentiment (deeper layers capture abstract polarity better)
        if config["num_classes"] <= 2:
            soft_tokens = 4  # Fewer tokens for binary (inverse scaling)
            train_lr = 5e-4  # Higher LR for binary (weaker gradients need stronger updates)
            train_steps = min(args.train_steps * 2, 4000)  # More steps for binary
            source_layer = 24  # Layer 24 captures sentiment better than 16 (+3pp in ablations)
            print(f"  [Binary task] Using adaptive hyperparams: tokens={soft_tokens}, lr={train_lr}, steps={train_steps}, layer={source_layer}")
        else:
            soft_tokens = args.soft_tokens
            train_lr = 2e-4
            train_steps = args.train_steps
            source_layer = 16  # Default for multi-class

        # Run experiments for each seed
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'-'*70}")
            print(f"SEED {seed} ({seed_idx + 1}/{len(seeds)})")
            print(f"{'-'*70}")

            set_seed(seed)

            dataset_results = {
                "random_chance": config["random_chance"],
            }

            # 1. BRIDGE
            print(f"\n[1/8] Training BRIDGE (seed={seed})...")
            bridge = UnifiedBridge(sender_dim, receiver_dim, soft_tokens).to(device=device, dtype=torch.bfloat16)
            train_info = train_bridge(
                bridge, sender, sender_tok, receiver, receiver_tok,
                train_ds, dataset_name, device,
                steps=train_steps, lr=train_lr, source_layer=source_layer
            )
            bridge_results = eval_bridge(
                bridge, sender, sender_tok, receiver, receiver_tok,
                eval_ds, dataset_name, device, source_layer=source_layer
            )
            bridge_results["train_info"] = train_info
            dataset_results["bridge"] = bridge_results
            seed_results_list["bridge"].append(bridge_results)
            print(f"  Bridge accuracy: {bridge_results['accuracy']:.1f}%")

            # Save bridge checkpoint
            torch.save(bridge.state_dict(), f"{args.output_dir}/bridge_{dataset_name}_seed{seed}.pt")

            # 2. PROMPT-TUNING
            print(f"\n[2/7] Training PROMPT-TUNING (no sender, seed={seed})...")
            prompt_module = SoftPromptTuning(args.soft_tokens, receiver_dim).to(device=device, dtype=torch.bfloat16)
            train_info = train_prompt_tuning(
                prompt_module, receiver, receiver_tok,
                train_ds, dataset_name, device,
                steps=args.train_steps
            )
            pt_results = eval_prompt_tuning(
                prompt_module, receiver, receiver_tok,
                eval_ds, dataset_name, device
            )
            pt_results["train_info"] = train_info
            dataset_results["prompt_tuning"] = pt_results
            seed_results_list["prompt_tuning"].append(pt_results)
            print(f"  Prompt-tuning accuracy: {pt_results['accuracy']:.1f}%")

            # 3. LINEAR PROBE BASELINE
            print(f"\n[3/8] Training LINEAR PROBE (sklearn LogisticRegression, seed={seed})...")

            # Use adaptive layer selection based on task
            probe_layer = 24 if config["num_classes"] <= 2 else 16

            linear_probe = LinearProbeBaseline(
                hidden_dim=sender_dim,  # e.g., 4096 for Llama-8B
                num_classes=config["num_classes"],
                layer_idx=probe_layer,  # Layer 24 for binary, 16 for multi-class
                pooling="mean",  # Mean pooling over sequence
                normalize=True,  # Standardize features
                C=1.0,  # Regularization strength
                max_iter=1000,
                n_jobs=-1,  # Use all CPU cores
                random_state=seed,
            )

            # Train with cross-validation
            lp_train_info = train_linear_probe(
                probe=linear_probe,
                model=sender,
                tokenizer=sender_tok,
                train_ds=train_ds,
                dataset_name=dataset_name,
                device=device,
                dataset_config=config,
                batch_size=4,  # Small batch to avoid OOM
                cv_folds=5,  # 5-fold cross-validation
            )

            # Evaluate
            lp_results = eval_linear_probe(
                probe=linear_probe,
                model=sender,
                tokenizer=sender_tok,
                eval_ds=eval_ds,
                dataset_name=dataset_name,
                device=device,
                dataset_config=config,
                batch_size=4,
            )

            # Add training info to results
            lp_results["train_info"] = lp_train_info
            dataset_results["linear_probe"] = lp_results
            seed_results_list["linear_probe"].append(lp_results)

            # Save probe weights for reproducibility
            probe_path = f"{args.output_dir}/linear_probe_{dataset_name}_seed{seed}"
            linear_probe.save(probe_path)

            print(f"  Linear probe accuracy: {lp_results['accuracy']:.1f}%")
            if "cv_mean" in lp_train_info:
                print(f"  CV accuracy: {lp_train_info['cv_mean']:.1f}% ± {lp_train_info['cv_std']:.1f}%")

            # 4. TEXT-RELAY
            if not args.skip_text_relay:
                print(f"\n[4/8] Evaluating TEXT-RELAY (seed={seed})...")
                relay_results = eval_text_relay(
                    sender, sender_tok, receiver, receiver_tok,
                    eval_ds, dataset_name, device
                )
                dataset_results["text_relay"] = relay_results
                seed_results_list["text_relay"].append(relay_results)
                print(f"  Text-relay accuracy: {relay_results['accuracy']:.1f}%, latency: {relay_results['latency_ms']:.0f}ms")
            else:
                dataset_results["text_relay"] = {"accuracy": None, "skipped": True}

            # 5. ZERO-SHOT LLAMA
            print(f"\n[5/8] Evaluating LLAMA ZERO-SHOT (seed={seed})...")
            llama_zs = eval_zeroshot(sender, sender_tok, eval_ds, dataset_name, device, "Llama")
            dataset_results["llama_zeroshot"] = llama_zs
            seed_results_list["llama_zeroshot"].append(llama_zs)
            print(f"  Llama zero-shot: {llama_zs['accuracy']:.1f}%")

            # 6. ZERO-SHOT MISTRAL
            print(f"\n[6/8] Evaluating MISTRAL ZERO-SHOT (seed={seed})...")
            mistral_zs = eval_zeroshot(receiver, receiver_tok, eval_ds, dataset_name, device, "Mistral")
            dataset_results["mistral_zeroshot"] = mistral_zs
            seed_results_list["mistral_zeroshot"].append(mistral_zs)
            print(f"  Mistral zero-shot: {mistral_zs['accuracy']:.1f}%")

            # 7. FEW-SHOT
            if not args.skip_fewshot:
                print(f"\n[7/8] Evaluating {args.fewshot_shots}-SHOT (seed={seed})...")
                mistral_fs = eval_fewshot(
                    receiver, receiver_tok, train_ds, eval_ds,
                    dataset_name, device, args.fewshot_shots, "Mistral"
                )
                dataset_results["mistral_fewshot"] = mistral_fs
                seed_results_list["mistral_fewshot"].append(mistral_fs)
                print(f"  Mistral {args.fewshot_shots}-shot: {mistral_fs['accuracy']:.1f}%")
            else:
                dataset_results["mistral_fewshot"] = {"accuracy": None, "skipped": True}

            # 8. ENSEMBLE (if enabled)
            if args.run_ensemble:
                print(f"\n[8/8] Evaluating ENSEMBLE baseline (seed={seed})...")
                ensemble_results = eval_ensemble(
                    sender, sender_tok, receiver, receiver_tok,
                    eval_ds, dataset_name, device,
                    "Sender", "Receiver"
                )
                dataset_results["ensemble"] = ensemble_results
                seed_results_list["ensemble"].append(ensemble_results)
                print(f"  Ensemble vote: {ensemble_results['vote_accuracy']:.1f}%")
                print(f"  Ensemble avg:  {ensemble_results['avg_accuracy']:.1f}%")

            # Save per-seed results
            all_seeds_results["per_seed_results"][dataset_name][seed] = dataset_results

        # Aggregate results across seeds
        aggregated = {
            "random_chance": config["random_chance"],
        }

        for method_name, method_results in seed_results_list.items():
            if len(method_results) == 0 or (len(method_results) > 0 and method_results[0].get("skipped", False)):
                aggregated[method_name] = {"accuracy": None, "skipped": True}
                continue

            # Extract accuracy values
            accuracies = [r["accuracy"] for r in method_results]

            # Compute statistics
            aggregated[method_name] = {
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
                "accuracy_min": np.min(accuracies),
                "accuracy_max": np.max(accuracies),
                "num_seeds": len(accuracies),
            }

            # Include latency if available
            if "latency_ms" in method_results[0]:
                latencies = [r["latency_ms"] for r in method_results]
                aggregated[method_name]["latency_ms_mean"] = np.mean(latencies)
                aggregated[method_name]["latency_ms_std"] = np.std(latencies, ddof=1) if len(latencies) > 1 else 0.0

        all_seeds_results["aggregated_results"][dataset_name] = aggregated

        # Build comparison table for this dataset (using aggregated results)
        all_seeds_results["comparison_table"][dataset_name] = {
            "Method": ["Random", "Linear Probe", "Prompt-Tuning", "Llama 0-shot", "Mistral 0-shot",
                       f"Mistral {args.fewshot_shots}-shot", "Text-Relay", "Bridge (ours)"],
            "Accuracy (Mean)": [
                config["random_chance"],
                aggregated["linear_probe"]["accuracy_mean"] if "accuracy_mean" in aggregated["linear_probe"] else "N/A",
                aggregated["prompt_tuning"]["accuracy_mean"] if "accuracy_mean" in aggregated["prompt_tuning"] else "N/A",
                aggregated["llama_zeroshot"]["accuracy_mean"] if "accuracy_mean" in aggregated["llama_zeroshot"] else "N/A",
                aggregated["mistral_zeroshot"]["accuracy_mean"] if "accuracy_mean" in aggregated["mistral_zeroshot"] else "N/A",
                aggregated["mistral_fewshot"]["accuracy_mean"] if "accuracy_mean" in aggregated["mistral_fewshot"] else "N/A",
                aggregated["text_relay"]["accuracy_mean"] if "accuracy_mean" in aggregated.get("text_relay", {}) else "N/A",
                aggregated["bridge"]["accuracy_mean"] if "accuracy_mean" in aggregated["bridge"] else "N/A",
            ],
            "Accuracy (Std)": [
                0.0,
                aggregated["linear_probe"]["accuracy_std"] if "accuracy_std" in aggregated["linear_probe"] else "N/A",
                aggregated["prompt_tuning"]["accuracy_std"] if "accuracy_std" in aggregated["prompt_tuning"] else "N/A",
                aggregated["llama_zeroshot"]["accuracy_std"] if "accuracy_std" in aggregated["llama_zeroshot"] else "N/A",
                aggregated["mistral_zeroshot"]["accuracy_std"] if "accuracy_std" in aggregated["mistral_zeroshot"] else "N/A",
                aggregated["mistral_fewshot"]["accuracy_std"] if "accuracy_std" in aggregated["mistral_fewshot"] else "N/A",
                aggregated["text_relay"]["accuracy_std"] if "accuracy_std" in aggregated.get("text_relay", {}) else "N/A",
                aggregated["bridge"]["accuracy_std"] if "accuracy_std" in aggregated["bridge"] else "N/A",
            ]
        }

        # Summary for this dataset (aggregated across seeds)
        print(f"\n{'='*70}")
        print(f"AGGREGATED SUMMARY: {dataset_name.upper()} (across {len(seeds)} seeds)")
        print(f"{'='*70}")
        print(f"  Random chance:    {config['random_chance']:.1f}%")
        if "accuracy_mean" in aggregated["prompt_tuning"]:
            print(f"  Prompt-Tuning:    {aggregated['prompt_tuning']['accuracy_mean']:.1f}% ± {aggregated['prompt_tuning']['accuracy_std']:.1f}")
        if "accuracy_mean" in aggregated["llama_zeroshot"]:
            print(f"  Llama 0-shot:     {aggregated['llama_zeroshot']['accuracy_mean']:.1f}% ± {aggregated['llama_zeroshot']['accuracy_std']:.1f}")
        if "accuracy_mean" in aggregated["mistral_zeroshot"]:
            print(f"  Mistral 0-shot:   {aggregated['mistral_zeroshot']['accuracy_mean']:.1f}% ± {aggregated['mistral_zeroshot']['accuracy_std']:.1f}")
        if "accuracy_mean" in aggregated["mistral_fewshot"]:
            print(f"  Mistral {args.fewshot_shots}-shot:    {aggregated['mistral_fewshot']['accuracy_mean']:.1f}% ± {aggregated['mistral_fewshot']['accuracy_std']:.1f}")
        if not args.skip_text_relay and "accuracy_mean" in aggregated.get("text_relay", {}):
            print(f"  Text-Relay:       {aggregated['text_relay']['accuracy_mean']:.1f}% ± {aggregated['text_relay']['accuracy_std']:.1f}")
        if "accuracy_mean" in aggregated["bridge"]:
            print(f"  Bridge (ours):    {aggregated['bridge']['accuracy_mean']:.1f}% ± {aggregated['bridge']['accuracy_std']:.1f}")
            if "latency_ms_mean" in aggregated["bridge"]:
                print(f"  Bridge latency:   {aggregated['bridge']['latency_ms_mean']:.1f}ms ± {aggregated['bridge']['latency_ms_std']:.1f}")

    # Save all results
    results_path = f"{args.output_dir}/unified_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_seeds_results, f, indent=2)

    # Also save a compact summary with just aggregated results
    summary_path = f"{args.output_dir}/unified_summary_{timestamp}.json"
    summary = {
        "meta": all_seeds_results["meta"],
        "aggregated_results": all_seeds_results["aggregated_results"],
        "comparison_table": all_seeds_results["comparison_table"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Full results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    # Final summary table with mean ± std
    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE (Mean ± Std across seeds)")
    print("="*70)
    print(f"{'Method':<25} ", end="")
    for ds in args.datasets:
        print(f"{ds.upper():<20}", end="")
    print()
    print("-"*70)

    methods = ["prompt_tuning", "llama_zeroshot", "mistral_zeroshot", "mistral_fewshot", "text_relay", "bridge"]
    method_names = ["Prompt-Tuning", "Llama 0-shot", "Mistral 0-shot", f"Mistral {args.fewshot_shots}-shot", "Text-Relay", "Bridge (ours)"]

    for method, name in zip(methods, method_names):
        print(f"{name:<25} ", end="")
        for ds in args.datasets:
            result = all_seeds_results["aggregated_results"][ds].get(method, {})
            if "accuracy_mean" in result:
                mean = result["accuracy_mean"]
                std = result["accuracy_std"]
                print(f"{mean:.1f}±{std:.1f}{'':>12}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()


if __name__ == "__main__":
    main()
