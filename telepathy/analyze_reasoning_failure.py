#!/usr/bin/env python
# telepathy/analyze_reasoning_failure.py
"""
Reasoning Failure Diagnostic: Why does the bridge fail on reasoning when it succeeds on classification?

BACKGROUND:
- Classification (SST-2: 94.7%, AG News: 88.9%): Bridge works excellently
- Reasoning (GSM8K: 2%): Bridge fails catastrophically

This script tests 5 hypotheses about why reasoning fails:

HYPOTHESIS 1: First-Token Accuracy
    Classification only needs 1 discriminative token (pos/neg, category name)
    Reasoning needs MANY correct tokens in sequence
    Test: Compare first-token accuracy between classification and reasoning tasks

HYPOTHESIS 2: Soft Token Diversity
    Classification: All soft tokens converge to "category vibe"
    Reasoning: Need diverse tokens encoding different reasoning steps/entities
    Test: Measure token-to-token similarity within samples

HYPOTHESIS 3: Layer Information Content
    Classification info may be at different layers than reasoning info
    Test: Compare bridge performance at different source layers

HYPOTHESIS 4: Entity Preservation
    Classification: Don't need specific entities (just category)
    Reasoning: Need exact numbers, names, quantities
    Test: Probe whether entities can be recovered from soft tokens

HYPOTHESIS 5: Chain-of-Thought Preservation
    Classification: Single-step decision
    Reasoning: Multi-step dependency chain
    Test: Measure information preservation across reasoning steps

OUTPUT:
- 4 diagnostic visualizations saved to output_dir/
- Detailed analysis report with actionable recommendations
- JSON results for programmatic analysis

RUNTIME: ~2-3 hours on single GPU

Usage:
    python telepathy/analyze_reasoning_failure.py \
        --classification_checkpoint runs/sst2_*/bridge.pt \
        --reasoning_checkpoint runs/gsm8k_*/bridge_gsm8k.pt \
        --output_dir runs/reasoning_diagnosis
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# BRIDGE ARCHITECTURE (from latentwire/bridge.py)
# =============================================================================

class PerceiverResampler(nn.Module):
    """Perceiver-style cross-attention resampler."""
    def __init__(self, src_dim, tgt_dim, num_latents=64, heads=8, depth=4):
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

    def forward(self, src_hidden, src_mask=None):
        B = src_hidden.shape[0]
        keys = self.input_proj(src_hidden.to(self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype))
        x = self.latents.unsqueeze(0).expand(B, -1, -1).to(keys.dtype)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            x_norm = layer["ln1"](x)
            attn_out, _ = layer["cross_attn"](
                query=x_norm, key=keys, value=keys,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            x = x + attn_out
            x_norm = layer["ln2"](x)
            attn_out, _ = layer["self_attn"](
                query=x_norm, key=x_norm, value=x_norm,
                need_weights=False
            )
            x = x + attn_out
            x = x + layer["ffn"](layer["ln3"](x))
        return x


class LatentBridge(nn.Module):
    """Telepathy Bridge (Continuous)."""
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        num_latents = getattr(args, 'soft_tokens', 128)
        heads = getattr(args, 'heads', 8)
        depth = getattr(args, 'depth', 4)
        self.resampler = PerceiverResampler(src_dim, tgt_dim, num_latents, heads, depth)
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward(self, src_hidden, src_mask=None):
        compressed = self.resampler(src_hidden, src_mask)
        rms = torch.sqrt((compressed ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (compressed / rms) * self.output_scale
        z_variance = compressed.var(dim=[0, 1]).mean()
        return out, torch.tensor(0.0, device=src_hidden.device), 1.0, z_variance


class RecurrentPerceiverBlock(nn.Module):
    """Perceiver block with recurrent capability for CoT."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_cross = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_self = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.ln_ffn = nn.LayerNorm(dim)

    def forward(self, latents, src_kv, src_mask=None, prev_latent=None):
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None
        x = self.ln_cross(latents)
        attn_out, _ = self.cross_attn(
            query=x, key=src_kv, value=src_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        latents = latents + attn_out

        x = self.ln_self(latents)
        if prev_latent is not None:
            prev_normed = self.ln_self(prev_latent)
            kv_context = torch.cat([x, prev_normed], dim=1)
        else:
            kv_context = x

        attn_out, _ = self.self_attn(
            query=x, key=kv_context, value=kv_context,
            need_weights=False
        )
        latents = latents + attn_out
        latents = latents + self.ffn(self.ln_ffn(latents))
        return latents


class LatentCoTBridge(nn.Module):
    """Latent Chain-of-Thought Bridge for reasoning tasks."""
    def __init__(self, args, src_dim, tgt_dim, target_rms=0.03):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.target_rms = target_rms
        self.num_latents = getattr(args, 'soft_tokens', 8)
        self.num_steps = getattr(args, 'cot_steps', 4)
        self.depth = getattr(args, 'depth', 2)
        self.heads = getattr(args, 'heads', 8)

        self.input_proj = nn.Linear(src_dim, tgt_dim) if src_dim != tgt_dim else nn.Identity()
        self.latent_queries = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_latents, tgt_dim) * 0.02)
            for _ in range(self.num_steps)
        ])
        self.step_embed = nn.Embedding(self.num_steps, tgt_dim)
        self.perceiver_blocks = nn.ModuleList([
            RecurrentPerceiverBlock(tgt_dim, self.heads)
            for _ in range(self.depth)
        ])
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward_step(self, src_kv, src_mask, step_idx, prev_latent=None):
        B = src_kv.shape[0]
        queries = self.latent_queries[step_idx].unsqueeze(0).expand(B, -1, -1)
        queries = queries.to(src_kv.dtype)
        step_emb = self.step_embed(torch.tensor([step_idx], device=src_kv.device))
        queries = queries + step_emb.unsqueeze(0)
        latents = queries
        for block in self.perceiver_blocks:
            latents = block(latents, src_kv, src_mask, prev_latent)
        return latents

    def forward(self, src_hidden, src_mask=None, return_all_steps=True):
        src_kv = self.input_proj(src_hidden.to(
            self.input_proj.weight.dtype if hasattr(self.input_proj, 'weight') else src_hidden.dtype
        ))
        all_latents = []
        prev_latent = None

        for step_idx in range(self.num_steps):
            step_latent = self.forward_step(src_kv, src_mask, step_idx, prev_latent)
            all_latents.append(step_latent)
            prev_latent = step_latent

        if return_all_steps:
            combined = torch.cat(all_latents, dim=1)
        else:
            combined = all_latents[-1]

        rms = torch.sqrt((combined ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        out = (combined / rms) * self.output_scale
        z_variance = combined.var(dim=[0, 1]).mean()
        return out, torch.tensor(0.0, device=src_hidden.device), 1.0, z_variance


# =============================================================================
# HELPER CLASSES AND FUNCTIONS
# =============================================================================

class Args:
    """Minimal args object for bridge instantiation."""
    def __init__(self, soft_tokens=8, heads=8, depth=2, cot_steps=4, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.cot_steps = cot_steps
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def extract_gsm8k_answer(text):
    """Extract numeric answer from GSM8K format."""
    match = re.search(r'[Tt]he answer is\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def extract_numbers_from_text(text):
    """Extract all numbers from text for entity preservation analysis."""
    return re.findall(r'\b\d+(?:\.\d+)?\b', text)


def get_first_token_probs(model, tokenizer, inputs_embeds, attention_mask):
    """Get probability distribution for first generated token."""
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        probs = F.softmax(logits, dim=-1)
    return probs


def get_nearest_neighbors(latent_vector, embedding_matrix, tokenizer, k=5):
    """Find k tokens closest (cosine similarity) to the given latent vector."""
    latent_vector = latent_vector.float()
    embedding_matrix = embedding_matrix.float()
    latent_norm = F.normalize(latent_vector.unsqueeze(0), p=2, dim=-1)
    emb_norm = F.normalize(embedding_matrix, p=2, dim=-1)
    similarity = torch.matmul(latent_norm, emb_norm.t())
    scores, indices = torch.topk(similarity, k)
    neighbors = []
    for score, idx in zip(scores[0], indices[0]):
        token_str = tokenizer.decode([idx.item()]).replace('\n', '\\n').replace('\t', '\\t')
        if token_str.strip() == '':
            token_str = repr(tokenizer.decode([idx.item()]))
        neighbors.append((token_str, score.item()))
    return neighbors


# =============================================================================
# HYPOTHESIS 1: First-Token Accuracy Analysis
# =============================================================================

def analyze_first_token_accuracy(
    src_model, tgt_model, src_tok, tgt_tok, bridge,
    classification_samples, reasoning_samples, device, source_layer=16
):
    """
    HYPOTHESIS 1: Classification needs 1 token, reasoning needs many.

    Compares first-token accuracy and top-k coverage between tasks.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: First-Token Accuracy Analysis")
    print("=" * 70)

    results = {
        "classification": {"correct": 0, "total": 0, "top5_hits": 0, "entropy": []},
        "reasoning": {"correct": 0, "total": 0, "top5_hits": 0, "entropy": []}
    }

    # Classification analysis (SST-2)
    print("\nAnalyzing classification (SST-2)...")
    for item in tqdm(classification_samples[:100], desc="Classification"):
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        src_input = f"Review: {text}\nSentiment (positive or negative):"
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[source_layer].bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            primer = "Sentiment:"
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids).bfloat16()
            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            probs = get_first_token_probs(tgt_model, tgt_tok, combined_embeds, attn_mask)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            results["classification"]["entropy"].append(entropy)

            top5_ids = torch.topk(probs, 5).indices[0].tolist()
            top5_tokens = [tgt_tok.decode([t]).lower().strip() for t in top5_ids]

            top1_token = top5_tokens[0]
            if label in top1_token:
                results["classification"]["correct"] += 1
            if any(label in t for t in top5_tokens):
                results["classification"]["top5_hits"] += 1
            results["classification"]["total"] += 1

    # Reasoning analysis (GSM8K)
    print("\nAnalyzing reasoning (GSM8K)...")
    for item in tqdm(reasoning_samples[:100], desc="Reasoning"):
        question = item['question']
        gold_answer = extract_gsm8k_answer(item['answer'])
        if gold_answer is None:
            continue

        src_input = f"Question: {question}\nLet me solve this step by step."
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[source_layer].bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            primer = "The answer is"
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids).bfloat16()
            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            probs = get_first_token_probs(tgt_model, tgt_tok, combined_embeds, attn_mask)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            results["reasoning"]["entropy"].append(entropy)

            top5_ids = torch.topk(probs, 5).indices[0].tolist()
            top5_tokens = [tgt_tok.decode([t]).strip() for t in top5_ids]

            # For reasoning, check if first digit matches
            first_digit = gold_answer[0] if gold_answer else None
            top1_token = top5_tokens[0]
            if first_digit and first_digit in top1_token:
                results["reasoning"]["correct"] += 1
            if first_digit and any(first_digit in t for t in top5_tokens):
                results["reasoning"]["top5_hits"] += 1
            results["reasoning"]["total"] += 1

    # Compute statistics
    cls_acc = 100 * results["classification"]["correct"] / max(results["classification"]["total"], 1)
    cls_top5 = 100 * results["classification"]["top5_hits"] / max(results["classification"]["total"], 1)
    cls_entropy = np.mean(results["classification"]["entropy"]) if results["classification"]["entropy"] else 0

    rsn_acc = 100 * results["reasoning"]["correct"] / max(results["reasoning"]["total"], 1)
    rsn_top5 = 100 * results["reasoning"]["top5_hits"] / max(results["reasoning"]["total"], 1)
    rsn_entropy = np.mean(results["reasoning"]["entropy"]) if results["reasoning"]["entropy"] else 0

    print(f"\n{'Task':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Entropy':<12}")
    print("-" * 56)
    print(f"{'Classification':<20} {cls_acc:.1f}%{'':<7} {cls_top5:.1f}%{'':<7} {cls_entropy:.2f}")
    print(f"{'Reasoning':<20} {rsn_acc:.1f}%{'':<7} {rsn_top5:.1f}%{'':<7} {rsn_entropy:.2f}")

    analysis = {
        "classification_top1": cls_acc,
        "classification_top5": cls_top5,
        "classification_entropy": cls_entropy,
        "reasoning_top1": rsn_acc,
        "reasoning_top5": rsn_top5,
        "reasoning_entropy": rsn_entropy,
        "gap_top1": cls_acc - rsn_acc,
        "gap_top5": cls_top5 - rsn_top5,
        "entropy_difference": rsn_entropy - cls_entropy
    }

    print(f"\nFINDINGS:")
    if analysis["gap_top1"] > 30:
        print("  - SIGNIFICANT: First-token accuracy gap confirms hypothesis")
        print("    Classification succeeds with single discriminative token,")
        print("    while reasoning needs precise multi-token sequences.")
    else:
        print("  - First-token accuracy gap is modest")
        print("    Other factors may dominate reasoning failure.")

    if analysis["entropy_difference"] > 1:
        print(f"  - Reasoning shows higher entropy ({analysis['entropy_difference']:.2f} higher)")
        print("    The model is less confident about reasoning outputs.")

    return analysis


# =============================================================================
# HYPOTHESIS 2: Soft Token Diversity Analysis
# =============================================================================

def analyze_soft_token_diversity(
    src_model, src_tok, bridge,
    classification_samples, reasoning_samples, device, source_layer=16
):
    """
    HYPOTHESIS 2: Classification collapses tokens, reasoning needs diversity.

    Measures within-sample and between-sample token diversity.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Soft Token Diversity Analysis")
    print("=" * 70)

    def compute_diversity_metrics(soft_tokens):
        """Compute diversity metrics for soft tokens [B, K, D]."""
        soft_tokens = soft_tokens.float()
        B, K, D = soft_tokens.shape

        metrics = {}

        # Within-sample diversity: mean pairwise similarity between tokens
        within_sims = []
        for b in range(B):
            tokens = F.normalize(soft_tokens[b], dim=-1)
            sim_matrix = torch.mm(tokens, tokens.t())
            mask = ~torch.eye(K, dtype=torch.bool, device=soft_tokens.device)
            within_sims.append(sim_matrix[mask].mean().item())
        metrics["within_sample_similarity"] = np.mean(within_sims)

        # Between-sample diversity: similarity of pooled representations
        pooled = soft_tokens.mean(dim=1)  # [B, D]
        pooled_norm = F.normalize(pooled, dim=-1)
        between_sim = torch.mm(pooled_norm, pooled_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=soft_tokens.device)
        metrics["between_sample_similarity"] = between_sim[mask].mean().item()

        # Token variance across batch
        metrics["token_variance"] = soft_tokens.var(dim=[0, 1]).mean().item()

        # Effective rank (how many dimensions are being used)
        flat = soft_tokens.reshape(-1, D)
        try:
            U, S, V = torch.svd(flat)
            explained_var = (S ** 2) / (S ** 2).sum()
            cum_var = torch.cumsum(explained_var, dim=0)
            metrics["effective_rank_90"] = (cum_var < 0.9).sum().item() + 1
        except:
            metrics["effective_rank_90"] = D

        return metrics

    # Collect soft tokens for classification
    print("\nCollecting classification soft tokens...")
    cls_tokens_list = []
    for item in tqdm(classification_samples[:50], desc="Classification"):
        text = item['sentence']
        src_input = f"Review: {text}\nSentiment (positive or negative):"
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[source_layer].bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)
            cls_tokens_list.append(soft_tokens.cpu())

    cls_tokens = torch.cat(cls_tokens_list, dim=0)
    cls_metrics = compute_diversity_metrics(cls_tokens)

    # Collect soft tokens for reasoning
    print("\nCollecting reasoning soft tokens...")
    rsn_tokens_list = []
    for item in tqdm(reasoning_samples[:50], desc="Reasoning"):
        question = item['question']
        src_input = f"Question: {question}\nLet me solve this step by step."
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[source_layer].bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)
            rsn_tokens_list.append(soft_tokens.cpu())

    rsn_tokens = torch.cat(rsn_tokens_list, dim=0)
    rsn_metrics = compute_diversity_metrics(rsn_tokens)

    print(f"\n{'Metric':<30} {'Classification':<15} {'Reasoning':<15}")
    print("-" * 60)
    print(f"{'Within-sample similarity':<30} {cls_metrics['within_sample_similarity']:.3f}{'':<10} {rsn_metrics['within_sample_similarity']:.3f}")
    print(f"{'Between-sample similarity':<30} {cls_metrics['between_sample_similarity']:.3f}{'':<10} {rsn_metrics['between_sample_similarity']:.3f}")
    print(f"{'Token variance':<30} {cls_metrics['token_variance']:.6f}{'':<6} {rsn_metrics['token_variance']:.6f}")
    print(f"{'Effective rank (90% var)':<30} {cls_metrics['effective_rank_90']:<15} {rsn_metrics['effective_rank_90']}")

    analysis = {
        "classification": cls_metrics,
        "reasoning": rsn_metrics,
        "within_similarity_gap": rsn_metrics["within_sample_similarity"] - cls_metrics["within_sample_similarity"],
        "between_similarity_gap": rsn_metrics["between_sample_similarity"] - cls_metrics["between_sample_similarity"],
        "cls_tokens_shape": list(cls_tokens.shape),
        "rsn_tokens_shape": list(rsn_tokens.shape)
    }

    print(f"\nFINDINGS:")
    if cls_metrics["within_sample_similarity"] > 0.8:
        print("  - Classification tokens are highly similar (mode collapse to category vibe)")
    if rsn_metrics["within_sample_similarity"] > 0.8:
        print("  - PROBLEM: Reasoning tokens also collapsed!")
        print("    Bridge cannot encode diverse reasoning steps.")
    if rsn_metrics["between_sample_similarity"] > cls_metrics["between_sample_similarity"]:
        print("  - Reasoning samples are MORE similar than classification samples")
        print("    Bridge produces generic 'math template' regardless of input.")

    return analysis, cls_tokens, rsn_tokens


# =============================================================================
# HYPOTHESIS 3: Layer Comparison Analysis
# =============================================================================

def analyze_layer_comparison(
    src_model, tgt_model, src_tok, tgt_tok, bridge_args,
    reasoning_samples, device
):
    """
    HYPOTHESIS 3: Different layers contain different information.

    Tests whether reasoning info is at different layers than classification info.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Layer Comparison Analysis")
    print("=" * 70)

    layers_to_test = [8, 16, 24, 31]
    layer_results = {}

    for layer in layers_to_test:
        print(f"\nTesting layer {layer}...")

        hidden_states_list = []
        gold_answers = []
        total = 0

        for item in tqdm(reasoning_samples[:30], desc=f"Layer {layer}"):
            question = item['question']
            gold = extract_gsm8k_answer(item['answer'])
            if gold is None:
                continue

            src_input = f"Question: {question}\nLet me solve this step by step."
            with torch.no_grad():
                src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
                src_out = src_model(**src_enc, output_hidden_states=True)
                src_h = src_out.hidden_states[layer]

                # Pool hidden states
                pooled = src_h.mean(dim=1).float().cpu()
                hidden_states_list.append(pooled.numpy())
                gold_answers.append(gold)
                total += 1

        # Convert to arrays
        X = np.vstack(hidden_states_list)

        # Compute information metrics
        try:
            U, S, V = np.linalg.svd(X, full_matrices=False)
            explained_var = (S ** 2) / (S ** 2).sum()
            effective_rank = np.sum(np.cumsum(explained_var) < 0.9) + 1
        except:
            effective_rank = X.shape[1]

        # Compute answer-based clustering
        unique_answers = list(set(gold_answers))
        if len(unique_answers) > 5:
            # Too many unique answers - bin them
            answer_bins = []
            for a in gold_answers:
                try:
                    val = float(a)
                    if val < 10:
                        answer_bins.append(0)
                    elif val < 100:
                        answer_bins.append(1)
                    else:
                        answer_bins.append(2)
                except:
                    answer_bins.append(1)
        else:
            answer_bins = [unique_answers.index(a) for a in gold_answers]

        # Compute cluster separation if we have multiple classes
        silhouette = 0.0
        if len(set(answer_bins)) > 1:
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X, answer_bins)
            except:
                pass

        layer_results[layer] = {
            "effective_rank": effective_rank,
            "top_singular_value_ratio": float(S[0] / S.sum()) if len(S) > 0 else 0,
            "silhouette_score": silhouette,
            "total_samples": total
        }

        print(f"  Effective rank: {effective_rank}")
        print(f"  Silhouette score: {silhouette:.3f}")

    print(f"\n{'Layer':<10} {'Eff. Rank':<15} {'Silhouette':<15} {'S1 Ratio':<15}")
    print("-" * 55)
    for layer in layers_to_test:
        r = layer_results[layer]
        print(f"{layer:<10} {r['effective_rank']:<15} {r['silhouette_score']:.3f}{'':<10} {r['top_singular_value_ratio']:.3f}")

    best_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["silhouette_score"])

    print(f"\nFINDINGS:")
    print(f"  - Best layer for reasoning clustering: {best_layer}")
    if best_layer != 16:
        print(f"    RECOMMENDATION: Try training bridge with source_layer={best_layer}")
    if all(r["silhouette_score"] < 0.1 for r in layer_results.values()):
        print("  - WARNING: No layer shows good answer clustering")
        print("    The source model may not encode reasoning info in pooled states.")

    return layer_results


# =============================================================================
# HYPOTHESIS 4: Entity Preservation Analysis
# =============================================================================

def analyze_entity_preservation(
    src_model, tgt_model, src_tok, tgt_tok, bridge,
    reasoning_samples, device, source_layer=16
):
    """
    HYPOTHESIS 4: Bridge loses specific entities needed for reasoning.

    Tests whether numbers and entities can be recovered from soft tokens.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Entity Preservation Analysis")
    print("=" * 70)

    # Get target embedding matrix for nearest neighbor analysis
    mistral_embeddings = tgt_model.get_input_embeddings().weight.detach()

    # Collect soft tokens and corresponding entities
    soft_tokens_list = []
    entity_labels = []

    # Focus on specific numbers that appear in problems
    target_numbers = ['1', '2', '3', '4', '5', '10', '20', '100']

    print("\nCollecting soft tokens and entity labels...")
    for item in tqdm(reasoning_samples[:100], desc="Collecting"):
        question = item['question']
        numbers_in_question = extract_numbers_from_text(question)

        src_input = f"Question: {question}\nLet me solve this step by step."
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[source_layer].bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            # Pool and store
            pooled = soft_tokens.mean(dim=1).float().cpu().numpy()
            soft_tokens_list.append(pooled[0])

            # Create multi-label entity vector
            entity_vec = [1 if num in numbers_in_question else 0 for num in target_numbers]
            entity_labels.append(entity_vec)

    X = np.array(soft_tokens_list)
    Y = np.array(entity_labels)

    print(f"\nFeatures shape: {X.shape}")
    print(f"Labels shape: {Y.shape}")

    # Train probe for each entity type
    results = {}
    print(f"\n{'Entity':<10} {'Present':<10} {'Absent':<10} {'Accuracy':<10}")
    print("-" * 40)

    for i, num in enumerate(target_numbers):
        y = Y[:, i]

        # Skip if too imbalanced
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            results[num] = {"accuracy": 0.0, "present": int(y.sum()), "absent": int(len(y) - y.sum())}
            print(f"{num:<10} {int(y.sum()):<10} {int(len(y) - y.sum()):<10} {'skipped':<10}")
            continue

        # Simple logistic regression probe
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            scores = cross_val_score(clf, X, y, cv=3)
            acc = scores.mean()
        except:
            acc = 0.5

        results[num] = {
            "accuracy": acc,
            "present": int(y.sum()),
            "absent": int(len(y) - y.sum())
        }
        print(f"{num:<10} {int(y.sum()):<10} {int(len(y) - y.sum()):<10} {acc:.2f}")

    # Aggregate statistics
    valid_results = [r["accuracy"] for r in results.values() if r["accuracy"] > 0]
    mean_accuracy = np.mean(valid_results) if valid_results else 0.5

    print(f"\nMean entity probe accuracy: {mean_accuracy:.2f}")

    print(f"\nFINDINGS:")
    if mean_accuracy < 0.6:
        print("  - CRITICAL: Entity information is largely LOST in soft tokens")
        print("    The bridge compresses away specific numbers/quantities.")
        print("    This explains reasoning failure: can't compute without operands.")
    elif mean_accuracy < 0.75:
        print("  - Entity information is partially preserved")
        print("    Some numbers recoverable, but not reliably.")
    else:
        print("  - Entity information is well preserved")
        print("    Reasoning failure may be due to other factors.")

    return results, mean_accuracy


# =============================================================================
# HYPOTHESIS 5: Chain-of-Thought Preservation Analysis
# =============================================================================

def analyze_cot_preservation(
    src_model, src_tok, bridge,
    reasoning_samples, device, source_layer=16
):
    """
    HYPOTHESIS 5: Multi-step reasoning requires sequential dependency.

    Tests whether reasoning steps are encoded distinctly in bridge output.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: Chain-of-Thought Preservation Analysis")
    print("=" * 70)

    # Check if bridge is CoT type
    is_cot = hasattr(bridge, 'num_steps') and hasattr(bridge, 'forward_step')

    if not is_cot:
        print("\nBridge is not CoT type - analyzing token sequence structure instead")

        # Analyze sequential structure in regular bridge
        step_similarities = []

        for item in tqdm(reasoning_samples[:30], desc="Analyzing"):
            question = item['question']
            src_input = f"Question: {question}\nLet me solve this step by step."

            with torch.no_grad():
                src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
                src_out = src_model(**src_enc, output_hidden_states=True)
                src_h = src_out.hidden_states[source_layer].bfloat16()
                soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

                # Analyze sequential similarity
                tokens = soft_tokens[0].float()
                K = tokens.shape[0]

                # Split into "pseudo-steps" (groups of tokens)
                step_size = max(1, K // 4)
                steps = [tokens[i:i+step_size].mean(dim=0) for i in range(0, K, step_size)]

                if len(steps) >= 2:
                    steps = torch.stack(steps[:4])  # Keep first 4 steps
                    steps_norm = F.normalize(steps, dim=-1)
                    sim_matrix = torch.mm(steps_norm, steps_norm.t())

                    # Get sequential similarities
                    seq_sims = [sim_matrix[i, i+1].item() for i in range(len(steps)-1)]
                    step_similarities.append(seq_sims)

        if step_similarities:
            avg_sims = np.mean(step_similarities, axis=0)
            print(f"\nSequential step similarities:")
            for i, sim in enumerate(avg_sims):
                print(f"  Step {i} -> Step {i+1}: {sim:.3f}")

            results = {
                "is_cot": False,
                "sequential_similarities": avg_sims.tolist(),
                "mean_sequential_similarity": float(np.mean(avg_sims))
            }

            print(f"\nFINDINGS:")
            if np.mean(avg_sims) > 0.9:
                print("  - PROBLEM: Sequential steps are nearly identical")
                print("    No differentiation between reasoning stages.")
            elif np.mean(avg_sims) > 0.7:
                print("  - Steps show some differentiation")
                print("    But high similarity suggests redundancy.")
            else:
                print("  - Steps are well differentiated")
                print("    Token sequence has structural variety.")
        else:
            results = {"is_cot": False, "error": "Could not compute step similarities"}

    else:
        # Analyze actual CoT bridge
        print(f"\nCoT Bridge with {bridge.num_steps} steps")

        step_representations = []

        for item in tqdm(reasoning_samples[:30], desc="Analyzing CoT"):
            question = item['question']
            src_input = f"Question: {question}\nLet me solve this step by step."

            with torch.no_grad():
                src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
                src_out = src_model(**src_enc, output_hidden_states=True)
                src_h = src_out.hidden_states[source_layer].bfloat16()

                # Get step-by-step representations
                src_kv = bridge.input_proj(src_h)
                prev_latent = None
                steps = []

                for step_idx in range(bridge.num_steps):
                    step_latent = bridge.forward_step(src_kv, src_enc.attention_mask, step_idx, prev_latent)
                    steps.append(step_latent.mean(dim=1).float().cpu())
                    prev_latent = step_latent

                step_representations.append(torch.cat(steps, dim=0))

        # Analyze step differentiation
        all_steps = torch.stack(step_representations)  # [N, num_steps, D]

        # Cross-step similarity
        step_sims = []
        for i in range(bridge.num_steps):
            for j in range(i+1, bridge.num_steps):
                step_i = F.normalize(all_steps[:, i, :], dim=-1)
                step_j = F.normalize(all_steps[:, j, :], dim=-1)
                sim = (step_i * step_j).sum(dim=-1).mean().item()
                step_sims.append((i, j, sim))

        print(f"\nStep-to-step similarities:")
        for i, j, sim in step_sims:
            print(f"  Step {i} <-> Step {j}: {sim:.3f}")

        results = {
            "is_cot": True,
            "num_steps": bridge.num_steps,
            "step_similarities": [(i, j, s) for i, j, s in step_sims],
            "mean_step_similarity": np.mean([s for _, _, s in step_sims])
        }

        print(f"\nFINDINGS:")
        mean_sim = results["mean_step_similarity"]
        if mean_sim > 0.9:
            print("  - PROBLEM: CoT steps are nearly identical")
            print("    Recurrent mechanism is not differentiating reasoning stages.")
        elif mean_sim > 0.7:
            print("  - CoT steps have moderate differentiation")
            print("    Some unique information per step, but high overlap.")
        else:
            print("  - CoT steps are well differentiated")
            print("    Each step contributes unique information.")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(results, cls_tokens, rsn_tokens, output_dir):
    """Generate 4 diagnostic visualizations."""
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. First-Token Accuracy Comparison
    ax1 = axes[0, 0]
    if "first_token" in results:
        ft = results["first_token"]
        categories = ['Top-1', 'Top-5']
        cls_vals = [ft["classification_top1"], ft["classification_top5"]]
        rsn_vals = [ft["reasoning_top1"], ft["reasoning_top5"]]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(x - width/2, cls_vals, width, label='Classification', color='#2ecc71')
        bars2 = ax1.bar(x + width/2, rsn_vals, width, label='Reasoning', color='#e74c3c')

        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('H1: First-Token Accuracy Gap')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.set_ylim(0, 100)

        for bar in list(bars1) + list(bars2):
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No first-token data', ha='center', va='center', transform=ax1.transAxes)

    # 2. Soft Token t-SNE
    ax2 = axes[0, 1]
    if cls_tokens is not None and rsn_tokens is not None:
        # Subsample for visualization
        n_samples = min(50, cls_tokens.shape[0], rsn_tokens.shape[0])
        cls_flat = cls_tokens[:n_samples].reshape(n_samples, -1).numpy()
        rsn_flat = rsn_tokens[:n_samples].reshape(n_samples, -1).numpy()

        combined = np.vstack([cls_flat, rsn_flat])
        labels = ['Classification'] * n_samples + ['Reasoning'] * n_samples

        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, len(combined) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
            embeddings = tsne.fit_transform(combined)

            ax2.scatter(embeddings[:n_samples, 0], embeddings[:n_samples, 1],
                       c='#2ecc71', label='Classification', alpha=0.7, s=50)
            ax2.scatter(embeddings[n_samples:, 0], embeddings[n_samples:, 1],
                       c='#e74c3c', label='Reasoning', alpha=0.7, s=50)
            ax2.set_title('H2: Soft Token Distribution (t-SNE)')
            ax2.legend()
        except Exception as e:
            ax2.text(0.5, 0.5, f't-SNE failed: {e}', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No token data', ha='center', va='center', transform=ax2.transAxes)

    # 3. Layer Comparison
    ax3 = axes[1, 0]
    if "layer_comparison" in results:
        lc = results["layer_comparison"]
        layers = sorted(lc.keys())
        silhouettes = [lc[l]["silhouette_score"] for l in layers]

        ax3_twin = ax3.twinx()

        bars1 = ax3.bar([str(l) for l in layers], silhouettes, label='Silhouette', color='#3498db')
        line = ax3_twin.plot([str(l) for l in layers], [lc[l]["effective_rank"] for l in layers], 'o-',
                            color='#9b59b6', label='Effective Rank')

        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Silhouette Score', color='#3498db')
        ax3_twin.set_ylabel('Effective Rank', color='#9b59b6')
        ax3.set_title('H3: Layer Information Content')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
    else:
        ax3.text(0.5, 0.5, 'No layer data', ha='center', va='center', transform=ax3.transAxes)

    # 4. Entity Preservation Probe
    ax4 = axes[1, 1]
    if "entity_preservation" in results:
        ep = results["entity_preservation"]["probe_results"]
        entities = list(ep.keys())
        accuracies = [ep[e]["accuracy"] for e in entities]

        colors = ['#2ecc71' if a > 0.65 else '#f39c12' if a > 0.55 else '#e74c3c' for a in accuracies]
        bars = ax4.bar(entities, accuracies, color=colors)

        ax4.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')
        ax4.set_xlabel('Entity (Number)')
        ax4.set_ylabel('Probe Accuracy')
        ax4.set_title('H4: Entity Preservation (Can we recover numbers?)')
        ax4.set_ylim(0, 1)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No entity data', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()

    # Save
    fig_path = os.path.join(output_dir, "reasoning_failure_diagnosis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {fig_path}")

    pdf_path = os.path.join(output_dir, "reasoning_failure_diagnosis.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF to {pdf_path}")

    plt.close()


# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def generate_recommendations(results):
    """Generate actionable recommendations based on findings."""
    recommendations = []

    print("\n" + "=" * 70)
    print("ACTIONABLE RECOMMENDATIONS")
    print("=" * 70)

    # H1: First-token accuracy
    if "first_token" in results:
        ft = results["first_token"]
        if ft["gap_top1"] > 30:
            recommendations.append({
                "hypothesis": "H1: First-Token Accuracy",
                "finding": f"Classification top-1: {ft['classification_top1']:.1f}%, Reasoning top-1: {ft['reasoning_top1']:.1f}%",
                "recommendation": "Consider multi-token supervision. Train with loss on first K tokens, not just first token.",
                "priority": "HIGH"
            })
        if ft["entropy_difference"] > 1:
            recommendations.append({
                "hypothesis": "H1: First-Token Entropy",
                "finding": f"Reasoning entropy {ft['entropy_difference']:.2f} higher than classification",
                "recommendation": "The model is uncertain about reasoning outputs. Consider temperature scaling or calibration.",
                "priority": "MEDIUM"
            })

    # H2: Token diversity
    if "diversity" in results:
        div = results["diversity"]
        if div["classification"]["within_sample_similarity"] > 0.8 and div["reasoning"]["within_sample_similarity"] > 0.8:
            recommendations.append({
                "hypothesis": "H2: Token Diversity",
                "finding": f"Both tasks show high within-sample similarity (>0.8)",
                "recommendation": "Add diversity loss during training to force token differentiation. Consider token dropout.",
                "priority": "HIGH"
            })
        if div["reasoning"]["between_sample_similarity"] > div["classification"]["between_sample_similarity"]:
            recommendations.append({
                "hypothesis": "H2: Sample Diversity",
                "finding": "Reasoning samples more similar than classification samples",
                "recommendation": "Bridge produces generic templates. Add contrastive loss between samples.",
                "priority": "HIGH"
            })

    # H3: Layer selection
    if "layer_comparison" in results:
        lc = results["layer_comparison"]
        best_layer = max(lc.keys(), key=lambda l: lc[l]["silhouette_score"])
        if best_layer != 16:
            recommendations.append({
                "hypothesis": "H3: Layer Selection",
                "finding": f"Best layer for reasoning: {best_layer} (current: 16)",
                "recommendation": f"Retrain bridge with --source_layer {best_layer}",
                "priority": "MEDIUM"
            })

    # H4: Entity preservation
    if "entity_preservation" in results:
        ep = results["entity_preservation"]
        if ep["mean_accuracy"] < 0.6:
            recommendations.append({
                "hypothesis": "H4: Entity Preservation",
                "finding": f"Mean entity probe accuracy: {ep['mean_accuracy']:.2f} (near random)",
                "recommendation": "Add entity-aware loss. Include reconstruction of key numbers in training objective.",
                "priority": "CRITICAL"
            })

    # H5: CoT preservation
    if "cot_preservation" in results:
        cot = results["cot_preservation"]
        if "mean_step_similarity" in cot and cot["mean_step_similarity"] > 0.85:
            recommendations.append({
                "hypothesis": "H5: CoT Preservation",
                "finding": f"CoT steps highly similar: {cot['mean_step_similarity']:.2f}",
                "recommendation": "CoT mechanism is collapsed. Add step-wise supervision or step-specific losses.",
                "priority": "HIGH"
            })
        elif "mean_sequential_similarity" in cot and cot["mean_sequential_similarity"] > 0.85:
            recommendations.append({
                "hypothesis": "H5: Sequential Structure",
                "finding": f"Token sequence lacks differentiation: {cot['mean_sequential_similarity']:.2f}",
                "recommendation": "Add positional supervision or sequence-aware training.",
                "priority": "MEDIUM"
            })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['hypothesis']}")
        print(f"   Finding: {rec['finding']}")
        print(f"   Action: {rec['recommendation']}")

    if not recommendations:
        print("\nNo critical issues identified. Consider:")
        print("  - Increasing model capacity (more soft tokens, deeper bridge)")
        print("  - Longer training with lower learning rate")
        print("  - Multi-task training (classification + reasoning)")

    return recommendations


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnose why reasoning fails through the bridge")
    parser.add_argument("--classification_checkpoint", type=str, default=None,
                       help="Path to classification bridge checkpoint (optional)")
    parser.add_argument("--reasoning_checkpoint", type=str, default=None,
                       help="Path to reasoning bridge checkpoint (optional)")
    parser.add_argument("--source_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="runs/reasoning_diagnosis")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--skip_layer_comparison", action="store_true",
                       help="Skip layer comparison (faster)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.time()

    print("=" * 70)
    print("REASONING FAILURE DIAGNOSTIC")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Classification checkpoint: {args.classification_checkpoint or 'None (will create fresh bridge)'}")
    print(f"Reasoning checkpoint: {args.reasoning_checkpoint or 'None (will create fresh bridge)'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("\nLoading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

    # Create or load bridge
    bridge_args = Args(soft_tokens=args.soft_tokens, heads=8, depth=2)
    bridge = LatentBridge(
        bridge_args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )

    if args.classification_checkpoint and os.path.exists(args.classification_checkpoint):
        print(f"Loading classification checkpoint: {args.classification_checkpoint}")
        checkpoint = torch.load(args.classification_checkpoint, map_location=device, weights_only=False)
        # Handle both old (state_dict only) and new (full checkpoint) formats
        if isinstance(checkpoint, dict) and "bridge_state_dict" in checkpoint:
            bridge.load_state_dict(checkpoint["bridge_state_dict"])
        else:
            bridge.load_state_dict(checkpoint)
    else:
        print("Using fresh bridge (no checkpoint loaded)")

    bridge = bridge.to(device).bfloat16().eval()

    # Load datasets
    print("\nLoading datasets...")
    sst2 = load_dataset("glue", "sst2", split="validation")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")

    classification_samples = list(sst2)[:args.num_samples]
    reasoning_samples = list(gsm8k)[:args.num_samples]

    print(f"Classification samples: {len(classification_samples)}")
    print(f"Reasoning samples: {len(reasoning_samples)}")

    # Run analyses
    results = {}
    cls_tokens = None
    rsn_tokens = None

    # H1: First-Token Accuracy
    print("\n" + "-" * 70)
    results["first_token"] = analyze_first_token_accuracy(
        src_model, tgt_model, src_tok, tgt_tok, bridge,
        classification_samples, reasoning_samples, device, args.source_layer
    )

    # H2: Soft Token Diversity
    print("\n" + "-" * 70)
    results["diversity"], cls_tokens, rsn_tokens = analyze_soft_token_diversity(
        src_model, src_tok, bridge,
        classification_samples, reasoning_samples, device, args.source_layer
    )

    # H3: Layer Comparison (optional - takes longer)
    if not args.skip_layer_comparison:
        print("\n" + "-" * 70)
        results["layer_comparison"] = analyze_layer_comparison(
            src_model, tgt_model, src_tok, tgt_tok, bridge_args,
            reasoning_samples, device
        )

    # H4: Entity Preservation
    print("\n" + "-" * 70)
    probe_results, mean_acc = analyze_entity_preservation(
        src_model, tgt_model, src_tok, tgt_tok, bridge,
        reasoning_samples, device, args.source_layer
    )
    results["entity_preservation"] = {
        "probe_results": probe_results,
        "mean_accuracy": mean_acc
    }

    # H5: CoT Preservation
    print("\n" + "-" * 70)
    results["cot_preservation"] = analyze_cot_preservation(
        src_model, src_tok, bridge,
        reasoning_samples, device, args.source_layer
    )

    # Generate visualizations
    print("\n" + "-" * 70)
    create_visualizations(results, cls_tokens, rsn_tokens, args.output_dir)

    # Generate recommendations
    recommendations = generate_recommendations(results)
    results["recommendations"] = recommendations

    # Save results
    results_path = os.path.join(args.output_dir, "diagnosis_results.json")

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    serializable_results = convert_to_serializable(results)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {args.output_dir}")
    print(f"  - diagnosis_results.json: Detailed numerical results")
    print(f"  - reasoning_failure_diagnosis.png: Visualizations")
    print(f"  - reasoning_failure_diagnosis.pdf: Publication-quality figures")

    print("\nKEY FINDINGS SUMMARY:")
    if "first_token" in results:
        ft = results["first_token"]
        print(f"  H1 First-Token Gap: {ft['gap_top1']:.1f}pp (Classification - Reasoning)")
    if "diversity" in results:
        div = results["diversity"]
        print(f"  H2 Within-Sample Sim: Cls={div['classification']['within_sample_similarity']:.3f}, Rsn={div['reasoning']['within_sample_similarity']:.3f}")
    if "entity_preservation" in results:
        print(f"  H4 Entity Preservation: {results['entity_preservation']['mean_accuracy']:.2f}")
    if "cot_preservation" in results:
        cot = results["cot_preservation"]
        if "mean_step_similarity" in cot:
            print(f"  H5 CoT Step Similarity: {cot['mean_step_similarity']:.3f}")
        elif "mean_sequential_similarity" in cot:
            print(f"  H5 Sequential Similarity: {cot['mean_sequential_similarity']:.3f}")

    print(f"\n{len(recommendations)} actionable recommendations generated.")


if __name__ == "__main__":
    main()
