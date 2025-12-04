#!/usr/bin/env python
# telepathy/comprehensive_experiments.py
"""
Comprehensive Telepathy Bridge Experiments

This script runs ALL experiments needed for a rigorous paper:

1. BRIDGE ARCHITECTURES
   - Continuous (Perceiver + RMS norm)
   - Diffusion Transformer
   - Linear projection
   - MLP bridge
   - Mean pooling
   - Identity (direct hidden state transfer)

2. LAYER ABLATIONS
   - Source layers: 0, 4, 8, 12, 16, 20, 24, 28, 31
   - Which layer contains the best semantic information?

3. COMPRESSION ABLATIONS
   - Soft tokens: 4, 8, 16, 32, 64, 128
   - How much compression is possible?

4. ARCHITECTURE DEPTH
   - Perceiver depth: 1, 2, 4, 6
   - How deep does the bridge need to be?

5. TRANSFER DIRECTIONS
   - Llama → Mistral (primary)
   - Mistral → Llama (reverse)
   - Llama → Llama (sanity check)

6. TRAINING ABLATIONS
   - With/without diversity loss
   - Diversity weights: 0, 0.05, 0.1, 0.2, 0.5

7. BASELINES
   - Random, majority, noise, text baselines for each config

All results saved to telepathy/preserved_data/exp_comprehensive/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os
import time
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import math


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# BRIDGE ARCHITECTURES
# =============================================================================

class PerceiverResampler(nn.Module):
    """Cross-attention based compression."""
    def __init__(self, src_dim, tgt_dim, num_latents=32, heads=8, depth=2):
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


class DiffusionBridge(nn.Module):
    """Diffusion Transformer bridge - generates soft tokens via denoising."""
    def __init__(self, src_dim, tgt_dim, num_latents=32, heads=8, depth=4, diffusion_steps=10):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim
        self.diffusion_steps = diffusion_steps

        self.input_proj = nn.Linear(src_dim, tgt_dim)
        self.time_embed = nn.Embedding(diffusion_steps, tgt_dim)

        # Cross-attention to source
        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln1": nn.LayerNorm(tgt_dim),
                "self_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln2": nn.LayerNorm(tgt_dim),
                "ffn": nn.Sequential(nn.Linear(tgt_dim, 4*tgt_dim), nn.GELU(), nn.Linear(4*tgt_dim, tgt_dim)),
                "ln3": nn.LayerNorm(tgt_dim)
            }) for _ in range(depth)
        ])

        self.output_proj = nn.Linear(tgt_dim, tgt_dim)

    def forward(self, src_hidden, src_mask=None):
        B = src_hidden.shape[0]
        device = src_hidden.device
        dtype = src_hidden.dtype

        # Project source
        src_proj = self.input_proj(src_hidden)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        # Start from noise (use fixed seed during eval for determinism)
        if self.training:
            x = torch.randn(B, self.num_latents, self.tgt_dim, device=device, dtype=dtype) * 0.1
        else:
            # Deterministic noise for evaluation
            generator = torch.Generator(device=device).manual_seed(42)
            x = torch.randn(B, self.num_latents, self.tgt_dim, device=device, dtype=dtype, generator=generator) * 0.1

        # Denoise
        for t in reversed(range(self.diffusion_steps)):
            t_emb = self.time_embed(torch.tensor([t], device=device)).unsqueeze(0).expand(B, self.num_latents, -1)
            x = x + t_emb.to(dtype)

            for layer in self.cross_layers:
                x = x + layer["cross_attn"](layer["ln1"](x), src_proj, src_proj, key_padding_mask=key_padding_mask)[0]
                x = x + layer["self_attn"](layer["ln2"](x), layer["ln2"](x), layer["ln2"](x))[0]
                x = x + layer["ffn"](layer["ln3"](x))

        return self.output_proj(x)


class MLPBridge(nn.Module):
    """Simple MLP bridge - pool then project."""
    def __init__(self, src_dim, tgt_dim, num_latents=32, hidden_dim=2048):
        super().__init__()
        self.num_latents = num_latents
        self.mlp = nn.Sequential(
            nn.Linear(src_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, tgt_dim * num_latents)
        )
        self.tgt_dim = tgt_dim

    def forward(self, src_hidden, src_mask=None):
        # Mean pool (preserve dtype)
        orig_dtype = src_hidden.dtype
        if src_mask is not None:
            mask = src_mask.unsqueeze(-1).float()
            pooled = (src_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled.to(orig_dtype)
        else:
            pooled = src_hidden.mean(dim=1)

        out = self.mlp(pooled)  # [B, tgt_dim * num_latents]
        return out.view(-1, self.num_latents, self.tgt_dim)


class LinearBridge(nn.Module):
    """Simple linear projection."""
    def __init__(self, src_dim, tgt_dim, num_latents=32):
        super().__init__()
        self.num_latents = num_latents
        self.proj = nn.Linear(src_dim, tgt_dim)
        self.tgt_dim = tgt_dim

    def forward(self, src_hidden, src_mask=None):
        # Mean pool (preserve dtype)
        orig_dtype = src_hidden.dtype
        if src_mask is not None:
            mask = src_mask.unsqueeze(-1).float()
            pooled = (src_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled.to(orig_dtype)
        else:
            pooled = src_hidden.mean(dim=1)

        projected = self.proj(pooled)  # [B, tgt_dim]
        return projected.unsqueeze(1).expand(-1, self.num_latents, -1)


class MeanPoolBridge(nn.Module):
    """Just mean pooling, no learned parameters except scale."""
    def __init__(self, src_dim, tgt_dim, num_latents=32):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim
        # Only project if dimensions differ
        self.proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

    def forward(self, src_hidden, src_mask=None):
        # Mean pool (preserve dtype)
        orig_dtype = src_hidden.dtype
        if src_mask is not None:
            mask = src_mask.unsqueeze(-1).float()
            pooled = (src_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled.to(orig_dtype)
        else:
            pooled = src_hidden.mean(dim=1)

        projected = self.proj(pooled)
        return projected.unsqueeze(1).expand(-1, self.num_latents, -1)


class IdentityBridge(nn.Module):
    """Direct transfer of last token hidden state."""
    def __init__(self, src_dim, tgt_dim, num_latents=32):
        super().__init__()
        self.num_latents = num_latents
        self.proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()

    def forward(self, src_hidden, src_mask=None):
        # Get last valid token
        if src_mask is not None:
            seq_lens = src_mask.sum(dim=1) - 1
            batch_idx = torch.arange(src_hidden.shape[0], device=src_hidden.device)
            last_hidden = src_hidden[batch_idx, seq_lens]
        else:
            last_hidden = src_hidden[:, -1]

        projected = self.proj(last_hidden)
        return projected.unsqueeze(1).expand(-1, self.num_latents, -1)


# =============================================================================
# UNIFIED BRIDGE WRAPPER
# =============================================================================

class UnifiedBridge(nn.Module):
    """Wrapper that handles all bridge types with RMS normalization."""

    BRIDGE_TYPES = ["continuous", "diffusion", "mlp", "linear", "meanpool", "identity"]

    def __init__(self, bridge_type, src_dim, tgt_dim, num_latents=32,
                 depth=2, heads=8, target_rms=0.003, diffusion_steps=10):
        super().__init__()
        self.bridge_type = bridge_type
        self.target_rms = nn.Parameter(torch.tensor(target_rms))

        if bridge_type == "continuous":
            self.bridge = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)
        elif bridge_type == "diffusion":
            self.bridge = DiffusionBridge(src_dim, tgt_dim, num_latents, heads, depth, diffusion_steps)
        elif bridge_type == "mlp":
            self.bridge = MLPBridge(src_dim, tgt_dim, num_latents)
        elif bridge_type == "linear":
            self.bridge = LinearBridge(src_dim, tgt_dim, num_latents)
        elif bridge_type == "meanpool":
            self.bridge = MeanPoolBridge(src_dim, tgt_dim, num_latents)
        elif bridge_type == "identity":
            self.bridge = IdentityBridge(src_dim, tgt_dim, num_latents)
        else:
            raise ValueError(f"Unknown bridge type: {bridge_type}")

    def forward(self, src_hidden, src_mask=None):
        out = self.bridge(src_hidden, src_mask)
        # RMS normalize
        rms = torch.sqrt((out ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (out / rms) * self.target_rms
        return out


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    bridge_type: str
    source_model: str
    target_model: str
    source_layer: int
    num_latents: int
    depth: int
    heads: int
    steps: int
    batch_size: int
    lr: float
    diversity_weight: float
    trained: bool = True  # False for untrained baseline


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_bridge(config: ExperimentConfig, bridge: UnifiedBridge,
                 src_model, tgt_model, src_tok, tgt_tok,
                 train_ds, src_device, tgt_device, verbose=True):
    """Train a bridge with given config. Supports multi-GPU (src on one, tgt on another)."""
    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=config.lr, weight_decay=0.01)

    dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    iter_dl = iter(dl)

    losses = []

    for step in tqdm(range(config.steps), desc=f"Train {config.name}", disable=not verbose):
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        optimizer.zero_grad()

        # Source forward (on src_device)
        inputs = batch['sentence']
        labels = ["negative" if l == 0 else "positive" for l in batch['label']]
        B = len(inputs)

        src_texts = [f"Review: {t}\nSentiment:" for t in inputs]

        with torch.no_grad():
            src_enc = src_tok(src_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128).to(src_device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            # Transfer hidden states to target device
            src_h = src_out.hidden_states[config.source_layer].to(tgt_device).to(bridge.target_rms.dtype)
            src_mask = src_enc.attention_mask.to(tgt_device)

        # Bridge (on tgt_device)
        soft_tokens = bridge(src_h, src_mask)

        # Diversity loss
        div_loss = torch.tensor(0.0, device=tgt_device)
        if config.diversity_weight > 0 and B > 1:
            flat = soft_tokens.reshape(B, -1).float()
            flat_norm = F.normalize(flat, dim=1)
            sim = torch.mm(flat_norm, flat_norm.t())
            mask = ~torch.eye(B, dtype=torch.bool, device=tgt_device)
            div_loss = sim[mask].mean()

        # Target forward (on tgt_device)
        with torch.no_grad():
            primer_enc = tgt_tok(["Sentiment:"] * B, return_tensors="pt", add_special_tokens=False).to(tgt_device)
            primer_emb = tgt_model.get_input_embeddings()(primer_enc.input_ids).to(soft_tokens.dtype)

            tgt_texts = [f" {l}{tgt_tok.eos_token}" for l in labels]
            tgt_enc = tgt_tok(tgt_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=16, add_special_tokens=False).to(tgt_device)
            answer_emb = tgt_model.get_input_embeddings()(tgt_enc.input_ids).to(soft_tokens.dtype)

        inputs_embeds = torch.cat([primer_emb, soft_tokens, answer_emb], dim=1)

        K = soft_tokens.shape[1]
        P = primer_emb.shape[1]
        ignore = torch.full((B, P + K), -100, dtype=torch.long, device=tgt_device)
        answer_labels = tgt_enc.input_ids.clone()
        answer_labels[tgt_enc.attention_mask == 0] = -100
        labels_tensor = torch.cat([ignore, answer_labels], dim=1)

        soft_mask = torch.ones(B, K, dtype=torch.long, device=tgt_device)
        full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

        outputs = tgt_model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels_tensor)

        loss = outputs.loss + config.diversity_weight * div_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


def evaluate_bridge(bridge: UnifiedBridge, src_model, tgt_model, src_tok, tgt_tok,
                   eval_ds, source_layer, src_device, tgt_device, num_samples=200):
    """Evaluate a bridge. Supports multi-GPU (src on one, tgt on another)."""
    bridge.eval()
    correct = 0
    total = 0

    for i in tqdm(range(min(num_samples, len(eval_ds))), desc="Eval", leave=False):
        item = eval_ds[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        src_input = f"Review: {text}\nSentiment:"

        with torch.no_grad():
            # Source forward (on src_device)
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(src_device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            # Transfer to target device
            src_h = src_out.hidden_states[source_layer].to(tgt_device).to(bridge.target_rms.dtype)
            src_mask = src_enc.attention_mask.to(tgt_device)

            # Bridge (on tgt_device)
            soft_tokens = bridge(src_h, src_mask)

            # Target forward (on tgt_device)
            primer = "Sentiment:"
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(tgt_device)
            primer_emb = tgt_model.get_input_embeddings()(primer_enc.input_ids).to(soft_tokens.dtype)

            combined = torch.cat([primer_emb, soft_tokens], dim=1)
            attn_mask = torch.ones(combined.shape[:2], device=tgt_device, dtype=torch.long)

            out_ids = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        if label in output:
            correct += 1
        total += 1

    return {"accuracy": 100 * correct / total, "correct": correct, "total": total}


def eval_text_baseline(model, tokenizer, ds, num_samples, device, model_name):
    """Text baseline - model sees full text."""
    correct = 0
    total = 0

    for i in tqdm(range(min(num_samples, len(ds))), desc=f"{model_name} text", leave=False):
        item = ds[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        prompt = f"Review: {text}\nSentiment (positive or negative):"

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            out_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

        if label in output:
            correct += 1
        total += 1

    return {"accuracy": 100 * correct / total, "correct": correct, "total": total}


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_all_experiments(args):
    """Run all experiments."""
    # Set seed for reproducibility
    set_seed(42)

    # Multi-GPU setup: put source model on GPU 0, target model on GPU 1
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        DEVICE_LLAMA = torch.device("cuda:0")
        DEVICE_MISTRAL = torch.device("cuda:1")
        print(f"Multi-GPU mode: Llama on cuda:0, Mistral on cuda:1")
    else:
        DEVICE_LLAMA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEVICE_MISTRAL = DEVICE_LLAMA
        print(f"Single device mode: {DEVICE_LLAMA}")

    print("=" * 80)
    print("COMPREHENSIVE TELEPATHY BRIDGE EXPERIMENTS")
    print("=" * 80)
    print(f"GPUs available: {num_gpus}")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Seed: 42")
    print("")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load models on separate GPUs
    print("Loading models...")
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16, device_map={"": DEVICE_LLAMA}
    ).eval()
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama_tok.pad_token = llama_tok.eos_token

    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16, device_map={"": DEVICE_MISTRAL}
    ).eval()
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    # Compute target RMS
    with torch.no_grad():
        mistral_rms = mistral.get_input_embeddings().weight.float().pow(2).mean(1).sqrt().median().item()
        llama_rms = llama.get_input_embeddings().weight.float().pow(2).mean(1).sqrt().median().item()

    print(f"Mistral embedding RMS: {mistral_rms:.4f}")
    print(f"Llama embedding RMS: {llama_rms:.4f}")

    # Load datasets
    print("Loading datasets...")
    train_ds = load_dataset("glue", "sst2", split="train")
    eval_ds = load_dataset("glue", "sst2", split="validation")

    # Shard for speed
    train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    all_results = {
        "meta": {
            "start_time": datetime.now().isoformat(),
            "args": vars(args),
            "mistral_rms": mistral_rms,
            "llama_rms": llama_rms,
        },
        "baselines": {},
        "experiments": {}
    }

    # =========================================================================
    # BASELINES
    # =========================================================================
    print("\n" + "=" * 80)
    print("BASELINES")
    print("=" * 80)

    # Text baselines
    print("\n[Baseline] Llama text...")
    all_results["baselines"]["llama_text"] = eval_text_baseline(
        llama, llama_tok, eval_ds, args.eval_samples, DEVICE_LLAMA, "Llama"
    )
    print(f"  Accuracy: {all_results['baselines']['llama_text']['accuracy']:.1f}%")

    print("\n[Baseline] Mistral text...")
    all_results["baselines"]["mistral_text"] = eval_text_baseline(
        mistral, mistral_tok, eval_ds, args.eval_samples, DEVICE_MISTRAL, "Mistral"
    )
    print(f"  Accuracy: {all_results['baselines']['mistral_text']['accuracy']:.1f}%")

    # Random and majority
    labels = [item['label'] for item in eval_ds]
    pos = sum(labels)
    all_results["baselines"]["random"] = {"accuracy": 50.0}
    all_results["baselines"]["majority"] = {"accuracy": 100 * max(pos, len(labels)-pos) / len(labels)}

    # Save intermediate
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # =========================================================================
    # EXPERIMENT MATRIX
    # =========================================================================

    experiments = []

    # 1. Bridge architecture comparison (all use layer 16, 32 tokens)
    for bridge_type in args.bridge_types:
        experiments.append(ExperimentConfig(
            name=f"bridge_{bridge_type}",
            bridge_type=bridge_type,
            source_model="llama",
            target_model="mistral",
            source_layer=16,
            num_latents=32,
            depth=2,
            heads=8,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=2e-4,
            diversity_weight=0.1,
            trained=True
        ))
        # Untrained baseline
        experiments.append(ExperimentConfig(
            name=f"bridge_{bridge_type}_untrained",
            bridge_type=bridge_type,
            source_model="llama",
            target_model="mistral",
            source_layer=16,
            num_latents=32,
            depth=2,
            heads=8,
            steps=0,
            batch_size=args.batch_size,
            lr=0,
            diversity_weight=0,
            trained=False
        ))

    # 2. Layer ablation (continuous bridge only)
    # Skip layer 16 since it's already covered in bridge_continuous
    for layer in args.source_layers:
        if layer == 16:  # Already covered by bridge_continuous
            continue
        experiments.append(ExperimentConfig(
            name=f"layer_{layer}",
            bridge_type="continuous",
            source_model="llama",
            target_model="mistral",
            source_layer=layer,
            num_latents=32,
            depth=2,
            heads=8,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=2e-4,
            diversity_weight=0.1,
            trained=True
        ))

    # 3. Compression ablation (num_latents)
    # Skip 32 since it's already covered in bridge_continuous
    for num_latents in args.num_latents_list:
        if num_latents == 32:  # Already covered by bridge_continuous
            continue
        experiments.append(ExperimentConfig(
            name=f"latents_{num_latents}",
            bridge_type="continuous",
            source_model="llama",
            target_model="mistral",
            source_layer=16,
            num_latents=num_latents,
            depth=2,
            heads=8,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=2e-4,
            diversity_weight=0.1,
            trained=True
        ))

    # 4. Depth ablation
    # Skip depth 2 since it's already covered in bridge_continuous
    for depth in args.depths:
        if depth == 2:  # Already covered by bridge_continuous
            continue
        experiments.append(ExperimentConfig(
            name=f"depth_{depth}",
            bridge_type="continuous",
            source_model="llama",
            target_model="mistral",
            source_layer=16,
            num_latents=32,
            depth=depth,
            heads=8,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=2e-4,
            diversity_weight=0.1,
            trained=True
        ))

    # 5. Diversity weight ablation
    # Skip 0.1 since it's already covered in bridge_continuous
    for div_weight in args.diversity_weights:
        if div_weight == 0.1:  # Already covered by bridge_continuous
            continue
        experiments.append(ExperimentConfig(
            name=f"div_weight_{div_weight}",
            bridge_type="continuous",
            source_model="llama",
            target_model="mistral",
            source_layer=16,
            num_latents=32,
            depth=2,
            heads=8,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=2e-4,
            diversity_weight=div_weight,
            trained=True
        ))

    # 6. Transfer direction
    if args.test_reverse:
        experiments.append(ExperimentConfig(
            name="reverse_mistral_to_llama",
            bridge_type="continuous",
            source_model="mistral",
            target_model="llama",
            source_layer=16,
            num_latents=32,
            depth=2,
            heads=8,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=2e-4,
            diversity_weight=0.1,
            trained=True
        ))

    # =========================================================================
    # RUN EXPERIMENTS
    # =========================================================================

    print(f"\n{'=' * 80}")
    print(f"RUNNING {len(experiments)} EXPERIMENTS")
    print("=" * 80)

    for i, config in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {config.name}")
        print("-" * 60)

        # Select models and devices based on config
        if config.source_model == "llama":
            src_model, src_tok, src_dim, src_device = llama, llama_tok, llama.config.hidden_size, DEVICE_LLAMA
        else:
            src_model, src_tok, src_dim, src_device = mistral, mistral_tok, mistral.config.hidden_size, DEVICE_MISTRAL

        if config.target_model == "mistral":
            tgt_model, tgt_tok, tgt_dim, tgt_rms, tgt_device = mistral, mistral_tok, mistral.config.hidden_size, mistral_rms, DEVICE_MISTRAL
        else:
            tgt_model, tgt_tok, tgt_dim, tgt_rms, tgt_device = llama, llama_tok, llama.config.hidden_size, llama_rms, DEVICE_LLAMA

        # Create bridge on target device (where we generate)
        bridge = UnifiedBridge(
            config.bridge_type, src_dim, tgt_dim,
            num_latents=config.num_latents,
            depth=config.depth,
            heads=config.heads,
            target_rms=tgt_rms
        ).to(tgt_device).to(torch.bfloat16)

        # Train if needed
        train_losses = []
        if config.trained and config.steps > 0:
            train_losses = train_bridge(
                config, bridge, src_model, tgt_model, src_tok, tgt_tok,
                train_ds, src_device, tgt_device, verbose=True
            )

        # Evaluate
        result = evaluate_bridge(
            bridge, src_model, tgt_model, src_tok, tgt_tok,
            eval_ds, config.source_layer, src_device, tgt_device, args.eval_samples
        )

        result["config"] = asdict(config)
        result["train_losses"] = train_losses[-10:] if train_losses else []  # Last 10 losses
        result["final_loss"] = train_losses[-1] if train_losses else None

        all_results["experiments"][config.name] = result
        print(f"  Accuracy: {result['accuracy']:.1f}%")

        # Save checkpoint
        if config.trained and config.steps > 0:
            ckpt_path = os.path.join(output_dir, f"{config.name}.pt")
            torch.save(bridge.state_dict(), ckpt_path)

        # Save intermediate results
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)

        # Clean up
        del bridge
        torch.cuda.empty_cache()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    all_results["meta"]["end_time"] = datetime.now().isoformat()

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\n{'Experiment':<35} {'Accuracy':>10}")
    print("-" * 50)

    # Baselines
    for name, res in all_results["baselines"].items():
        print(f"{name:<35} {res['accuracy']:>9.1f}%")

    print("-" * 50)

    # Experiments sorted by accuracy
    sorted_exps = sorted(all_results["experiments"].items(),
                        key=lambda x: x[1]["accuracy"], reverse=True)
    for name, res in sorted_exps:
        print(f"{name:<35} {res['accuracy']:>9.1f}%")

    # Save final results
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_dir}/results.json")
    print(f"End time: {datetime.now().isoformat()}")
    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--output_dir", default="runs/comprehensive_experiments")
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)

    # Ablation ranges
    parser.add_argument("--bridge_types", nargs="+",
                       default=["continuous", "diffusion", "mlp", "linear", "meanpool", "identity"])
    parser.add_argument("--source_layers", nargs="+", type=int,
                       default=[0, 8, 16, 24, 31])
    parser.add_argument("--num_latents_list", nargs="+", type=int,
                       default=[4, 8, 16, 32, 64])
    parser.add_argument("--depths", nargs="+", type=int,
                       default=[1, 2, 4])
    parser.add_argument("--diversity_weights", nargs="+", type=float,
                       default=[0.0, 0.1, 0.5])

    # Optional experiments
    parser.add_argument("--test_reverse", action="store_true", default=True)

    args = parser.parse_args()
    run_all_experiments(args)


if __name__ == "__main__":
    main()
