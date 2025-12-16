#!/usr/bin/env python3
"""
Unified Comparison Experiment Script

Runs ALL baselines needed for paper in ONE script:
1. Bridge (Llama→Mistral) - Main method
2. Prompt-Tuning (Mistral only) - Proves sender is essential
3. Text-Relay (Llama summarizes → Mistral classifies) - Latency baseline
4. Few-shot Prompting (5-shot) - In-context learning baseline
5. Zero-shot baselines (Llama, Mistral direct)

Output: Single JSON with all results for easy comparison

Usage:
    python telepathy/run_unified_comparison.py --datasets sst2 agnews trec --output_dir runs/unified
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_text_relay", action="store_true", help="Skip slow text-relay baseline")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("UNIFIED COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Datasets: {args.datasets}")
    print(f"Soft tokens: {args.soft_tokens}")
    print(f"Train steps: {args.train_steps}")
    print(f"Eval samples: {args.eval_samples}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("\nLoading models...")
    sender = AutoModelForCausalLM.from_pretrained(
        args.sender, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sender_tok = AutoTokenizer.from_pretrained(args.sender)
    sender_tok.pad_token = sender_tok.eos_token

    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver, torch_dtype=torch.bfloat16, device_map="auto"
    )
    receiver_tok = AutoTokenizer.from_pretrained(args.receiver)
    receiver_tok.pad_token = receiver_tok.eos_token

    sender.eval()
    receiver.eval()

    # Results container
    all_results = {
        "meta": {
            "timestamp": timestamp,
            "sender": args.sender,
            "receiver": args.receiver,
            "soft_tokens": args.soft_tokens,
            "train_steps": args.train_steps,
            "eval_samples": args.eval_samples,
            "seed": args.seed,
        },
        "results": {},
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

        dataset_results = {
            "random_chance": config["random_chance"],
        }

        # ADAPTIVE HYPERPARAMETERS for binary classification (SST-2 FIX)
        # Binary tasks need: fewer tokens (4 vs 8), higher LR (5e-4 vs 2e-4), more steps (4000 vs 2000)
        if config["num_classes"] <= 2:
            soft_tokens = 4  # Fewer tokens for binary (inverse scaling)
            train_lr = 5e-4  # Higher LR for binary (weaker gradients need stronger updates)
            train_steps = min(args.train_steps * 2, 4000)  # More steps for binary
            print(f"  [Binary task] Using adaptive hyperparams: tokens={soft_tokens}, lr={train_lr}, steps={train_steps}")
        else:
            soft_tokens = args.soft_tokens
            train_lr = 2e-4
            train_steps = args.train_steps

        # 1. BRIDGE
        print("\n[1/6] Training BRIDGE...")
        bridge = UnifiedBridge(sender_dim, receiver_dim, soft_tokens).to(device=device, dtype=torch.bfloat16)
        train_info = train_bridge(
            bridge, sender, sender_tok, receiver, receiver_tok,
            train_ds, dataset_name, device,
            steps=train_steps, lr=train_lr
        )
        bridge_results = eval_bridge(
            bridge, sender, sender_tok, receiver, receiver_tok,
            eval_ds, dataset_name, device
        )
        bridge_results["train_info"] = train_info
        dataset_results["bridge"] = bridge_results
        print(f"  Bridge accuracy: {bridge_results['accuracy']:.1f}%")

        # Save bridge checkpoint
        torch.save(bridge.state_dict(), f"{args.output_dir}/bridge_{dataset_name}.pt")

        # 2. PROMPT-TUNING
        print("\n[2/6] Training PROMPT-TUNING (no sender)...")
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
        print(f"  Prompt-tuning accuracy: {pt_results['accuracy']:.1f}%")

        # 3. TEXT-RELAY
        if not args.skip_text_relay:
            print("\n[3/6] Evaluating TEXT-RELAY...")
            relay_results = eval_text_relay(
                sender, sender_tok, receiver, receiver_tok,
                eval_ds, dataset_name, device
            )
            dataset_results["text_relay"] = relay_results
            print(f"  Text-relay accuracy: {relay_results['accuracy']:.1f}%, latency: {relay_results['latency_ms']:.0f}ms")
        else:
            dataset_results["text_relay"] = {"accuracy": None, "skipped": True}

        # 4. ZERO-SHOT LLAMA
        print("\n[4/6] Evaluating LLAMA ZERO-SHOT...")
        llama_zs = eval_zeroshot(sender, sender_tok, eval_ds, dataset_name, device, "Llama")
        dataset_results["llama_zeroshot"] = llama_zs
        print(f"  Llama zero-shot: {llama_zs['accuracy']:.1f}%")

        # 5. ZERO-SHOT MISTRAL
        print("\n[5/6] Evaluating MISTRAL ZERO-SHOT...")
        mistral_zs = eval_zeroshot(receiver, receiver_tok, eval_ds, dataset_name, device, "Mistral")
        dataset_results["mistral_zeroshot"] = mistral_zs
        print(f"  Mistral zero-shot: {mistral_zs['accuracy']:.1f}%")

        # 6. FEW-SHOT
        print(f"\n[6/6] Evaluating {args.fewshot_shots}-SHOT...")
        mistral_fs = eval_fewshot(
            receiver, receiver_tok, train_ds, eval_ds,
            dataset_name, device, args.fewshot_shots, "Mistral"
        )
        dataset_results["mistral_fewshot"] = mistral_fs
        print(f"  Mistral {args.fewshot_shots}-shot: {mistral_fs['accuracy']:.1f}%")

        all_results["results"][dataset_name] = dataset_results

        # Build comparison table for this dataset
        all_results["comparison_table"][dataset_name] = {
            "Method": ["Random", "Prompt-Tuning", "Llama 0-shot", "Mistral 0-shot",
                       f"Mistral {args.fewshot_shots}-shot", "Text-Relay", "Bridge (ours)"],
            "Accuracy": [
                config["random_chance"],
                pt_results["accuracy"],
                llama_zs["accuracy"],
                mistral_zs["accuracy"],
                mistral_fs["accuracy"],
                dataset_results["text_relay"]["accuracy"] if not args.skip_text_relay else "N/A",
                bridge_results["accuracy"],
            ]
        }

        # Summary for this dataset
        print(f"\n{'='*50}")
        print(f"SUMMARY: {dataset_name.upper()}")
        print(f"{'='*50}")
        print(f"  Random chance:    {config['random_chance']:.1f}%")
        print(f"  Prompt-Tuning:    {pt_results['accuracy']:.1f}%")
        print(f"  Llama 0-shot:     {llama_zs['accuracy']:.1f}%")
        print(f"  Mistral 0-shot:   {mistral_zs['accuracy']:.1f}%")
        print(f"  Mistral {args.fewshot_shots}-shot:    {mistral_fs['accuracy']:.1f}%")
        if not args.skip_text_relay:
            print(f"  Text-Relay:       {dataset_results['text_relay']['accuracy']:.1f}%")
        print(f"  Bridge (ours):    {bridge_results['accuracy']:.1f}%")
        print(f"  Bridge latency:   {bridge_results['latency_ms']:.1f}ms")

    # Save all results
    results_path = f"{args.output_dir}/unified_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_path}")

    # Final summary table
    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE")
    print("="*70)
    print(f"{'Method':<25} ", end="")
    for ds in args.datasets:
        print(f"{ds.upper():<12}", end="")
    print()
    print("-"*70)

    methods = ["prompt_tuning", "llama_zeroshot", "mistral_zeroshot", "mistral_fewshot", "text_relay", "bridge"]
    method_names = ["Prompt-Tuning", "Llama 0-shot", "Mistral 0-shot", f"Mistral {args.fewshot_shots}-shot", "Text-Relay", "Bridge (ours)"]

    for method, name in zip(methods, method_names):
        print(f"{name:<25} ", end="")
        for ds in args.datasets:
            result = all_results["results"][ds].get(method, {})
            acc = result.get("accuracy")
            if acc is not None:
                print(f"{acc:<12.1f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()


if __name__ == "__main__":
    main()
