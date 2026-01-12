#!/usr/bin/env python
"""
Telepathy Reasoning Failure Analysis

This script performs systematic diagnosis of WHY the Telepathy bridge fails on
reasoning tasks (GSM8K) while succeeding on classification tasks (SST-2, AG News).

Known issues from REPORT.md:
- Classification: 85-90% accuracy (working)
- Reasoning: 0-5% accuracy (failing)
- Entity tracking loss: Bridge loses specific numbers/entities

Hypotheses to test:
1. Compression bottleneck: Not enough tokens for multi-step reasoning
2. Layer selection: Layer 16 captures concepts, layer 31 captures answers
3. Entity tracking: Bridge loses specific numbers needed for math
4. Chain preservation: Multi-step reasoning chains collapse

Diagnostic tests:
A. First-token accuracy: Does bridge preserve question semantics?
B. Soft token diversity: Are tokens collapsed or diverse?
C. Layer comparison: Layer 16 vs Layer 31 representations
D. Entity preservation: Can bridge transmit specific numbers?
E. Chain-of-thought test: Does reasoning chain survive compression?

Expected runtime: 2-3 hours on 4× H100 GPUs

Usage:
    python telepathy/analyze_reasoning_failure.py \
        --checkpoint runs/gsm8k_bridge/bridge.pt \
        --output_dir runs/reasoning_analysis \
        --num_samples 200
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC

# Import existing bridge infrastructure
from latent_bridge_v15 import LatentBridgeV15
from inspect_latents import get_nearest_neighbors, analyze_latent_geometry


# =============================================================================
# Configuration
# =============================================================================

class Args:
    """Args object for LatentBridgeV15 interface."""
    def __init__(self, soft_tokens=8, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


# =============================================================================
# Answer Extraction (from gsm8k_eval.py)
# =============================================================================

def extract_answer(text):
    """Extract numeric answer following standard GSM8K format."""
    # Primary: Look for "The answer is X"
    match = re.search(r'[Tt]he answer is\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')

    # Secondary: Look for #### pattern
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')

    # Fallback: last number in text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def extract_numbers_from_question(question):
    """Extract all numbers mentioned in the question."""
    numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', question)
    return [n.replace(',', '') for n in numbers]


# =============================================================================
# Test A: First-Token Accuracy
# =============================================================================

def test_first_token_accuracy(
    bridge, src_model, tgt_model, src_tok, tgt_tok,
    dataset, source_layer, device, num_samples=100
):
    """
    Test if bridge preserves enough information for first token prediction.

    Measures:
    - Top-1/Top-5 first token accuracy
    - Compares latent vs text baseline

    Hypothesis: If first token is wrong, rest of generation will fail.
    """
    print("\n" + "="*70)
    print("TEST A: FIRST-TOKEN ACCURACY")
    print("="*70)
    print("Hypothesis: Bridge must preserve question semantics for first token")
    print("")

    results = {
        "latent_top1": 0,
        "latent_top5": 0,
        "text_top1": 0,
        "text_top5": 0,
        "total": 0
    }

    primer = "Analysis of received thought vector: "
    primer_ids = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    for i in tqdm(range(num_samples), desc="First-token test"):
        item = dataset[i]
        question = item['question']
        answer = item['answer']

        # Get first token of answer
        answer_text = answer.split('\n')[0]  # First line of reasoning
        answer_ids = tgt_tok(answer_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if answer_ids.shape[1] == 0:
            continue
        gold_first_token = answer_ids[0, 0].item()

        # === Latent Bridge Path ===
        with torch.no_grad():
            # Encode question with source model
            src_inputs = src_tok(question, return_tensors="pt", padding=True).to(device)
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]
            src_mask = src_inputs.attention_mask

            # Bridge: hidden states -> soft tokens
            latents, _, _, _ = bridge(src_hidden, src_mask)

            # Get target model embeddings for soft tokens
            tgt_embeds = tgt_model.get_input_embeddings()(primer_ids)
            combined = torch.cat([tgt_embeds, latents], dim=1)

            # Get logits for next token prediction
            attention_mask = torch.ones(1, combined.shape[1], device=device, dtype=torch.long)
            outputs = tgt_model(inputs_embeds=combined, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]  # Last position logits

            # Check top-k
            top_tokens = torch.topk(logits, k=5).indices
            results["latent_top1"] += (top_tokens[0].item() == gold_first_token)
            results["latent_top5"] += (gold_first_token in top_tokens.tolist())

        # === Text Baseline ===
        with torch.no_grad():
            # Full text prompt
            prompt = f"Q: {question}\nA:"
            tgt_inputs = tgt_tok(prompt, return_tensors="pt").to(device)
            tgt_out = tgt_model(**tgt_inputs)
            logits = tgt_out.logits[0, -1, :]

            top_tokens = torch.topk(logits, k=5).indices
            results["text_top1"] += (top_tokens[0].item() == gold_first_token)
            results["text_top5"] += (gold_first_token in top_tokens.tolist())

        results["total"] += 1

    # Compute percentages
    total = results["total"]
    print(f"\nResults ({total} samples):")
    print(f"  Latent Bridge:")
    print(f"    Top-1: {results['latent_top1']}/{total} = {100*results['latent_top1']/total:.1f}%")
    print(f"    Top-5: {results['latent_top5']}/{total} = {100*results['latent_top5']/total:.1f}%")
    print(f"  Text Baseline:")
    print(f"    Top-1: {results['text_top1']}/{total} = {100*results['text_top1']/total:.1f}%")
    print(f"    Top-5: {results['text_top5']}/{total} = {100*results['text_top5']/total:.1f}%")
    print("")
    print(f"  Gap: {results['text_top1'] - results['latent_top1']} tokens lost")

    if results['latent_top1'] < 0.05 * total:
        print("  ⚠️  CRITICAL: First token is essentially random!")
        print("      Bridge is not preserving question semantics.")
    elif results['latent_top1'] < 0.5 * results['text_top1']:
        print("  ⚠️  DEGRADED: Bridge loses significant information.")
    else:
        print("  ✓  First token accuracy is reasonable.")

    return results


# =============================================================================
# Test B: Soft Token Diversity & Collapse
# =============================================================================

def test_soft_token_diversity(
    bridge, src_model, src_tok, dataset, source_layer, device, num_samples=100
):
    """
    Test if soft tokens are diverse or collapsed.

    Measures:
    - Intra-sample diversity: Are tokens within one sample different?
    - Inter-sample diversity: Are different questions encoded differently?
    - Rank/dimensionality: Is bridge using full capacity?

    Hypothesis: Mode collapse = all questions produce similar tokens.
    """
    print("\n" + "="*70)
    print("TEST B: SOFT TOKEN DIVERSITY")
    print("="*70)
    print("Hypothesis: Collapsed tokens cannot encode specific question details")
    print("")

    all_latents = []  # [num_samples, soft_tokens, dim]

    for i in tqdm(range(num_samples), desc="Collecting latents"):
        item = dataset[i]
        question = item['question']

        with torch.no_grad():
            src_inputs = src_tok(question, return_tensors="pt", padding=True).to(device)
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]
            src_mask = src_inputs.attention_mask

            latents, _, _, _ = bridge(src_hidden, src_mask)
            all_latents.append(latents[0].cpu().float())  # [soft_tokens, dim]

    all_latents = torch.stack(all_latents)  # [num_samples, soft_tokens, dim]

    # === Metric 1: Intra-sample diversity (tokens within one question) ===
    intra_similarities = []
    for sample_latents in all_latents:
        # Normalize tokens
        normed = F.normalize(sample_latents, dim=-1)
        # Pairwise similarity
        sim_matrix = torch.matmul(normed, normed.t())
        # Off-diagonal mean (how similar are different tokens)
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
        intra_sim = sim_matrix[mask].mean().item()
        intra_similarities.append(intra_sim)

    mean_intra_sim = np.mean(intra_similarities)

    print(f"Intra-sample diversity:")
    print(f"  Mean token similarity: {mean_intra_sim:.3f}")
    if mean_intra_sim > 0.9:
        print(f"  ⚠️  COLLAPSED: All tokens in a sample are nearly identical!")
        print(f"      Bridge is producing K copies of the same vector.")
    elif mean_intra_sim > 0.7:
        print(f"  ⚡ MODERATE: Tokens share structure but have variations.")
    else:
        print(f"  ✓  DIVERSE: Tokens encode different aspects.")

    # === Metric 2: Inter-sample diversity (different questions) ===
    # Pool all tokens: [num_samples * soft_tokens, dim]
    pooled = all_latents.view(-1, all_latents.shape[-1])
    # Sample 1000 random pairs
    num_pairs = min(1000, pooled.shape[0] // 2)
    indices = torch.randperm(pooled.shape[0])[:num_pairs*2]
    pairs_a = pooled[indices[::2]]
    pairs_b = pooled[indices[1::2]]

    # Cosine similarity
    normed_a = F.normalize(pairs_a, dim=-1)
    normed_b = F.normalize(pairs_b, dim=-1)
    inter_sim = (normed_a * normed_b).sum(dim=-1).mean().item()

    print(f"\nInter-sample diversity:")
    print(f"  Mean similarity between random token pairs: {inter_sim:.3f}")
    if inter_sim > 0.8:
        print(f"  ⚠️  COLLAPSED: Different questions produce similar tokens!")
        print(f"      This explains why reasoning fails.")
    elif inter_sim > 0.5:
        print(f"  ⚡ MODERATE: Some overlap between questions.")
    else:
        print(f"  ✓  DIVERSE: Questions are distinguishable.")

    # === Metric 3: Effective dimensionality ===
    # Flatten to [num_samples * soft_tokens, dim]
    flattened = all_latents.view(-1, all_latents.shape[-1]).float()

    # Center the data
    mean = flattened.mean(dim=0, keepdim=True)
    centered = flattened - mean

    # Compute covariance and eigenvalues
    cov = torch.matmul(centered.t(), centered) / (centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]

    # Compute effective rank (participation ratio)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove numerical zeros
    if len(eigenvalues) > 0:
        normalized_eigs = eigenvalues / eigenvalues.sum()
        effective_rank = 1.0 / (normalized_eigs ** 2).sum().item()
    else:
        effective_rank = 0

    # Variance explained by top components
    variance_90 = 0
    for i, eig in enumerate(normalized_eigs):
        variance_90 += eig.item()
        if variance_90 >= 0.9:
            dims_for_90 = i + 1
            break
    else:
        dims_for_90 = len(eigenvalues)

    print(f"\nEffective dimensionality:")
    print(f"  Full dimension: {all_latents.shape[-1]}")
    print(f"  Effective rank: {effective_rank:.1f}")
    print(f"  Dims for 90% variance: {dims_for_90}")

    if effective_rank < 10:
        print(f"  ⚠️  COLLAPSED: Only {effective_rank:.0f} effective dimensions!")
        print(f"      Bridge is not using its full capacity.")
    elif effective_rank < 100:
        print(f"  ⚡ MODERATE: {effective_rank:.0f} effective dimensions used.")
    else:
        print(f"  ✓  DIVERSE: High-dimensional representation.")

    results = {
        "intra_similarity": mean_intra_sim,
        "inter_similarity": inter_sim,
        "effective_rank": effective_rank,
        "dims_for_90_variance": dims_for_90
    }

    return results, all_latents


# =============================================================================
# Test C: Layer Comparison (16 vs 31)
# =============================================================================

def test_layer_comparison(
    bridge_16, bridge_31, src_model, tgt_model, src_tok, tgt_tok,
    dataset, device, num_samples=50
):
    """
    Compare Layer 16 (concepts) vs Layer 31 (answers).

    Tests if different layers encode different information:
    - Layer 16: Semantic concepts ("this is about ducks and eggs")
    - Layer 31: Answer-oriented features ("the answer is...")

    Hypothesis: Reasoning needs Layer 31, classification works with Layer 16.
    """
    print("\n" + "="*70)
    print("TEST C: LAYER COMPARISON (16 vs 31)")
    print("="*70)
    print("Hypothesis: Layer 16 = concepts, Layer 31 = answers")
    print("")

    results = {
        "layer_16_accuracy": 0,
        "layer_31_accuracy": 0,
        "total": 0
    }

    primer = "Analysis of received thought vector: "
    primer_ids = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    for i in tqdm(range(num_samples), desc="Layer comparison"):
        item = dataset[i]
        question = item['question']
        gold_answer = extract_answer(item['answer'])
        if gold_answer is None:
            continue

        # === Layer 16 ===
        with torch.no_grad():
            src_inputs = src_tok(question, return_tensors="pt", padding=True).to(device)
            src_out = src_model(**src_inputs, output_hidden_states=True)

            # Layer 16
            src_hidden_16 = src_out.hidden_states[16]
            src_mask = src_inputs.attention_mask
            latents_16, _, _, _ = bridge_16(src_hidden_16, src_mask)

            # Generate with Layer 16
            tgt_embeds = tgt_model.get_input_embeddings()(primer_ids)
            combined = torch.cat([tgt_embeds, latents_16], dim=1)
            attention_mask = torch.ones(1, combined.shape[1], device=device, dtype=torch.long)

            outputs = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tgt_tok.pad_token_id
            )
            pred_16 = tgt_tok.decode(outputs[0], skip_special_tokens=True)
            pred_ans_16 = extract_answer(pred_16)

            if pred_ans_16 == gold_answer:
                results["layer_16_accuracy"] += 1

            # Layer 31
            src_hidden_31 = src_out.hidden_states[31]
            latents_31, _, _, _ = bridge_31(src_hidden_31, src_mask)

            combined = torch.cat([tgt_embeds, latents_31], dim=1)
            outputs = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tgt_tok.pad_token_id
            )
            pred_31 = tgt_tok.decode(outputs[0], skip_special_tokens=True)
            pred_ans_31 = extract_answer(pred_31)

            if pred_ans_31 == gold_answer:
                results["layer_31_accuracy"] += 1

        results["total"] += 1

    total = results["total"]
    print(f"\nResults ({total} samples):")
    print(f"  Layer 16 accuracy: {results['layer_16_accuracy']}/{total} = {100*results['layer_16_accuracy']/total:.1f}%")
    print(f"  Layer 31 accuracy: {results['layer_31_accuracy']}/{total} = {100*results['layer_31_accuracy']/total:.1f}%")
    print(f"  Gap: {results['layer_31_accuracy'] - results['layer_16_accuracy']} questions")

    if results['layer_31_accuracy'] > 2 * results['layer_16_accuracy']:
        print("  ✓  Layer 31 is significantly better for reasoning!")
        print("     Recommendation: Use Layer 31 for math tasks.")
    elif results['layer_16_accuracy'] > 2 * results['layer_31_accuracy']:
        print("  ⚠️  Layer 16 is better? Unexpected result.")
    else:
        print("  ⚡ Layers perform similarly. Problem is elsewhere.")

    return results


# =============================================================================
# Test D: Entity Preservation (Number Tracking)
# =============================================================================

def test_entity_preservation(
    bridge, src_model, tgt_model, src_tok, tgt_tok,
    dataset, source_layer, device, num_samples=100
):
    """
    Test if bridge can transmit specific numbers from questions.

    Measures:
    - Number recall: Are question numbers present in nearest neighbors?
    - Digit token presence: Do any soft tokens map to digit tokens?

    Hypothesis: If numbers are lost, math reasoning is impossible.
    """
    print("\n" + "="*70)
    print("TEST D: ENTITY PRESERVATION (NUMBER TRACKING)")
    print("="*70)
    print("Hypothesis: Bridge must preserve specific numbers for math")
    print("")

    # Get Mistral embedding matrix
    mistral_embeddings = tgt_model.get_input_embeddings().weight.detach()

    # Identify digit tokens
    digit_tokens = []
    for i in range(mistral_embeddings.shape[0]):
        token_str = tgt_tok.decode([i])
        if re.match(r'^\s*\d+\s*$', token_str):
            digit_tokens.append(i)

    print(f"Found {len(digit_tokens)} digit tokens in vocabulary")

    results = {
        "number_recall": [],  # For each sample, how many question numbers appear in top-50 neighbors
        "digit_token_rank": [],  # Minimum rank of any digit token
        "samples_with_digits": 0,  # Samples where at least one digit appears in top-10
        "total": 0
    }

    for i in tqdm(range(num_samples), desc="Entity tracking"):
        item = dataset[i]
        question = item['question']
        question_numbers = extract_numbers_from_question(question)

        if len(question_numbers) == 0:
            continue

        with torch.no_grad():
            src_inputs = src_tok(question, return_tensors="pt", padding=True).to(device)
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]
            src_mask = src_inputs.attention_mask

            latents, _, _, _ = bridge(src_hidden, src_mask)
            latents = latents[0]  # [soft_tokens, dim]

        # For each soft token, find nearest neighbors
        all_neighbors = []
        for tok_idx in range(latents.shape[0]):
            neighbors = get_nearest_neighbors(
                latents[tok_idx], mistral_embeddings, tgt_tok, k=50
            )
            all_neighbors.extend([tok for tok, _ in neighbors])

        # Check number recall
        numbers_found = 0
        for num in question_numbers:
            # Check if number appears in any neighbor token
            if any(num in neighbor for neighbor in all_neighbors[:50]):
                numbers_found += 1

        recall = numbers_found / len(question_numbers) if len(question_numbers) > 0 else 0
        results["number_recall"].append(recall)

        # Check if any digit token appears in top-10
        top_10 = all_neighbors[:10]
        has_digit = any(re.search(r'\d', tok) for tok in top_10)
        if has_digit:
            results["samples_with_digits"] += 1

        results["total"] += 1

    # Aggregate results
    mean_recall = np.mean(results["number_recall"]) if results["number_recall"] else 0

    print(f"\nResults ({results['total']} samples with numbers):")
    print(f"  Mean number recall: {100*mean_recall:.1f}%")
    print(f"    (How many question numbers appear in top-50 neighbors)")
    print(f"  Samples with digits in top-10: {results['samples_with_digits']}/{results['total']} = {100*results['samples_with_digits']/results['total']:.1f}%")

    if mean_recall < 0.1:
        print("  ⚠️  CRITICAL: Numbers are NOT preserved!")
        print("      Bridge loses specific numerical values.")
        print("      This explains GSM8K failure.")
    elif mean_recall < 0.3:
        print("  ⚡ DEGRADED: Most numbers are lost.")
    else:
        print("  ✓  Numbers are reasonably preserved.")

    return results


# =============================================================================
# Test E: Chain-of-Thought Preservation
# =============================================================================

def test_chain_preservation(
    bridge, src_model, tgt_model, src_tok, tgt_tok,
    dataset, source_layer, device, num_samples=50
):
    """
    Test if reasoning chain survives compression.

    Compares:
    - Text baseline: Full CoT answer provided
    - Latent bridge: Compressed question only

    Measures:
    - Reasoning step overlap: How many reasoning steps match?
    - Operation preservation: Are arithmetic operations (+ - * /) preserved?

    Hypothesis: Multi-step chain requires multiple soft tokens per step.
    """
    print("\n" + "="*70)
    print("TEST E: CHAIN-OF-THOUGHT PRESERVATION")
    print("="*70)
    print("Hypothesis: Multi-step reasoning needs explicit chain encoding")
    print("")

    results = {
        "step_overlap": [],
        "operation_match": [],
        "total": 0
    }

    primer = "Analysis of received thought vector: "
    primer_ids = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    for i in tqdm(range(num_samples), desc="Chain preservation"):
        item = dataset[i]
        question = item['question']
        gold_answer_full = item['answer']

        # Extract reasoning steps from gold answer
        gold_steps = gold_answer_full.split('.')
        gold_steps = [s.strip() for s in gold_steps if s.strip()]
        gold_operations = re.findall(r'[+\-*/=]', gold_answer_full)

        # === Latent Bridge ===
        with torch.no_grad():
            src_inputs = src_tok(question, return_tensors="pt", padding=True).to(device)
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]
            src_mask = src_inputs.attention_mask

            latents, _, _, _ = bridge(src_hidden, src_mask)

            tgt_embeds = tgt_model.get_input_embeddings()(primer_ids)
            combined = torch.cat([tgt_embeds, latents], dim=1)
            attention_mask = torch.ones(1, combined.shape[1], device=device, dtype=torch.long)

            outputs = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tgt_tok.pad_token_id
            )
            pred = tgt_tok.decode(outputs[0], skip_special_tokens=True)

        # Extract predicted steps
        pred_steps = pred.split('.')
        pred_steps = [s.strip() for s in pred_steps if s.strip()]
        pred_operations = re.findall(r'[+\-*/=]', pred)

        # Compute step overlap (Jaccard similarity)
        gold_set = set(' '.join(gold_steps).lower().split())
        pred_set = set(' '.join(pred_steps).lower().split())
        if len(gold_set | pred_set) > 0:
            step_overlap = len(gold_set & pred_set) / len(gold_set | pred_set)
        else:
            step_overlap = 0

        # Compute operation match
        gold_op_counts = Counter(gold_operations)
        pred_op_counts = Counter(pred_operations)

        if len(gold_op_counts) > 0:
            op_match = sum((gold_op_counts & pred_op_counts).values()) / sum(gold_op_counts.values())
        else:
            op_match = 0

        results["step_overlap"].append(step_overlap)
        results["operation_match"].append(op_match)
        results["total"] += 1

    mean_step_overlap = np.mean(results["step_overlap"]) if results["step_overlap"] else 0
    mean_op_match = np.mean(results["operation_match"]) if results["operation_match"] else 0

    print(f"\nResults ({results['total']} samples):")
    print(f"  Mean step overlap: {100*mean_step_overlap:.1f}%")
    print(f"    (Jaccard similarity between reasoning steps)")
    print(f"  Mean operation match: {100*mean_op_match:.1f}%")
    print(f"    (Fraction of arithmetic operations preserved)")

    if mean_step_overlap < 0.1:
        print("  ⚠️  CRITICAL: Reasoning chain is lost!")
        print("      Bridge does not preserve multi-step reasoning.")
    elif mean_step_overlap < 0.3:
        print("  ⚡ DEGRADED: Partial reasoning preserved.")
    else:
        print("  ✓  Reasoning chain is reasonably preserved.")

    if mean_op_match < 0.3:
        print("  ⚠️  Operations are not preserved.")
        print("      Model doesn't know which arithmetic to perform.")

    return results


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(all_results, output_dir):
    """Create diagnostic visualizations."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: First token accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    test_a = all_results["test_a"]
    categories = ["Latent\nTop-1", "Latent\nTop-5", "Text\nTop-1", "Text\nTop-5"]
    values = [
        100 * test_a["latent_top1"] / test_a["total"],
        100 * test_a["latent_top5"] / test_a["total"],
        100 * test_a["text_top1"] / test_a["total"],
        100 * test_a["text_top5"] / test_a["total"]
    ]
    colors = ['#e74c3c', '#e74c3c', '#3498db', '#3498db']
    ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("First-Token Prediction Accuracy")
    ax.set_ylim([0, 100])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "first_token_accuracy.png", dpi=150)
    print(f"  Saved: {output_dir / 'first_token_accuracy.png'}")
    plt.close()

    # Plot 2: Diversity metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    test_b = all_results["test_b"]

    # Intra vs inter similarity
    similarities = ["Intra-sample\n(within question)", "Inter-sample\n(between questions)"]
    values = [test_b["intra_similarity"], test_b["inter_similarity"]]
    ax1.bar(similarities, values, color=['#2ecc71', '#e67e22'], alpha=0.7)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Token Diversity")
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Collapse threshold')
    ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Critical collapse')
    ax1.legend()
    ax1.set_ylim([0, 1])

    # Effective dimensionality
    dims = ["Full\nDimension", "Effective\nRank", "90%\nVariance"]
    values = [4096, test_b["effective_rank"], test_b["dims_for_90_variance"]]
    ax2.bar(dims, values, color=['#3498db', '#9b59b6', '#1abc9c'], alpha=0.7)
    ax2.set_ylabel("Number of Dimensions")
    ax2.set_title("Effective Dimensionality")
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "diversity_metrics.png", dpi=150)
    print(f"  Saved: {output_dir / 'diversity_metrics.png'}")
    plt.close()

    # Plot 3: Entity preservation
    fig, ax = plt.subplots(figsize=(8, 6))
    test_d = all_results["test_d"]

    # Histogram of number recall
    if test_d["number_recall"]:
        ax.hist(test_d["number_recall"], bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Number Recall (fraction)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Number Preservation")
        ax.axvline(x=np.mean(test_d["number_recall"]), color='blue', linestyle='--',
                   label=f'Mean = {np.mean(test_d["number_recall"]):.2f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "number_preservation.png", dpi=150)
        print(f"  Saved: {output_dir / 'number_preservation.png'}")
        plt.close()

    # Plot 4: Chain preservation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    test_e = all_results["test_e"]

    if test_e["step_overlap"]:
        ax1.hist(test_e["step_overlap"], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_xlabel("Step Overlap (Jaccard)")
        ax1.set_ylabel("Count")
        ax1.set_title("Reasoning Step Preservation")
        ax1.axvline(x=np.mean(test_e["step_overlap"]), color='red', linestyle='--',
                   label=f'Mean = {np.mean(test_e["step_overlap"]):.2f}')
        ax1.legend()

    if test_e["operation_match"]:
        ax2.hist(test_e["operation_match"], bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Operation Match (fraction)")
        ax2.set_ylabel("Count")
        ax2.set_title("Arithmetic Operation Preservation")
        ax2.axvline(x=np.mean(test_e["operation_match"]), color='red', linestyle='--',
                   label=f'Mean = {np.mean(test_e["operation_match"]):.2f}')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "chain_preservation.png", dpi=150)
    print(f"  Saved: {output_dir / 'chain_preservation.png'}")
    plt.close()

    print("  All visualizations complete!")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose reasoning failure in Telepathy bridge")
    parser.add_argument("--checkpoint", required=True, help="Path to bridge checkpoint")
    parser.add_argument("--checkpoint_layer16", default=None, help="Optional: separate Layer 16 checkpoint for comparison")
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=31, help="Which layer to extract (16 or 31)")
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=200, help="Samples per test")
    parser.add_argument("--output_dir", default="runs/reasoning_analysis")
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("TELEPATHY REASONING FAILURE ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source Layer: {args.source_layer}")
    print(f"Soft Tokens: {args.soft_tokens}")
    print(f"Device: {device}")
    print(f"Samples per test: {args.num_samples}")
    print("")
    print("Known Issues (from REPORT.md):")
    print("  - Classification: 85-90% accuracy (working)")
    print("  - Reasoning: 0-5% accuracy (failing)")
    print("  - Entity tracking loss: Bridge loses specific numbers")
    print("")
    print("Tests to run:")
    print("  A. First-token accuracy")
    print("  B. Soft token diversity")
    print("  C. Layer comparison (if checkpoint_layer16 provided)")
    print("  D. Entity preservation (numbers)")
    print("  E. Chain-of-thought preservation")
    print("="*70)

    # Load models
    print("\n[1/6] Loading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map={"": device}
    ).eval()

    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map={"": device}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    print(f"  Source: {args.source_model}")
    print(f"  Target: {args.target_model}")

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
    print(f"  Target RMS: {target_rms:.4f}")

    # Load bridge
    print("\n[2/6] Loading bridge...")
    bridge_args = Args(soft_tokens=args.soft_tokens, heads=args.heads, depth=args.depth, use_fsq=False)
    bridge = LatentBridgeV15(
        bridge_args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    bridge.load_state_dict(checkpoint)
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.eval()
    print(f"  Loaded: {args.checkpoint}")

    # Load Layer 16 bridge if provided (for Test C)
    bridge_16 = None
    if args.checkpoint_layer16:
        print(f"  Loading Layer 16 bridge: {args.checkpoint_layer16}")
        bridge_16 = LatentBridgeV15(
            bridge_args,
            src_dim=src_model.config.hidden_size,
            tgt_dim=tgt_model.config.hidden_size,
            target_rms=target_rms
        ).to(device)
        checkpoint_16 = torch.load(args.checkpoint_layer16, map_location=device, weights_only=True)
        bridge_16.load_state_dict(checkpoint_16)
        if args.bf16:
            bridge_16 = bridge_16.bfloat16()
        bridge_16.eval()

    # Load dataset
    print("\n[3/6] Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  Loaded {len(dataset)} test samples")

    # Run tests
    all_results = {}

    print("\n[4/6] Running diagnostic tests...")

    # Test A: First-token accuracy
    all_results["test_a"] = test_first_token_accuracy(
        bridge, src_model, tgt_model, src_tok, tgt_tok,
        dataset, args.source_layer, device, num_samples=min(args.num_samples, 100)
    )

    # Test B: Soft token diversity
    test_b_results, all_latents = test_soft_token_diversity(
        bridge, src_model, src_tok, dataset, args.source_layer, device,
        num_samples=min(args.num_samples, 100)
    )
    all_results["test_b"] = test_b_results

    # Test C: Layer comparison (only if bridge_16 provided)
    if bridge_16 is not None:
        all_results["test_c"] = test_layer_comparison(
            bridge_16, bridge, src_model, tgt_model, src_tok, tgt_tok,
            dataset, device, num_samples=min(args.num_samples, 50)
        )
    else:
        print("\n[SKIP] Test C: No Layer 16 checkpoint provided")

    # Test D: Entity preservation
    all_results["test_d"] = test_entity_preservation(
        bridge, src_model, tgt_model, src_tok, tgt_tok,
        dataset, args.source_layer, device, num_samples=min(args.num_samples, 100)
    )

    # Test E: Chain preservation
    all_results["test_e"] = test_chain_preservation(
        bridge, src_model, tgt_model, src_tok, tgt_tok,
        dataset, args.source_layer, device, num_samples=min(args.num_samples, 50)
    )

    # Create visualizations
    print("\n[5/6] Creating visualizations...")
    create_visualizations(all_results, args.output_dir)

    # Save results
    print("\n[6/6] Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for test_name, test_data in all_results.items():
            json_results[test_name] = {}
            for key, value in test_data.items():
                if isinstance(value, list):
                    json_results[test_name][key] = value
                elif isinstance(value, (int, float)):
                    json_results[test_name][key] = value
                else:
                    json_results[test_name][key] = str(value)

        json.dump(json_results, f, indent=2)

    print(f"  Saved: {output_dir / 'results.json'}")

    # Print summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print("\n1. FIRST-TOKEN ACCURACY")
    test_a = all_results["test_a"]
    latent_acc = 100 * test_a["latent_top1"] / test_a["total"]
    text_acc = 100 * test_a["text_top1"] / test_a["total"]
    print(f"   Latent: {latent_acc:.1f}% | Text: {text_acc:.1f}%")
    if latent_acc < 5:
        print("   ❌ CRITICAL: Bridge loses question semantics")
    elif latent_acc < 0.5 * text_acc:
        print("   ⚠️  WARNING: Significant information loss")
    else:
        print("   ✓  Reasonable preservation")

    print("\n2. SOFT TOKEN DIVERSITY")
    test_b = all_results["test_b"]
    print(f"   Intra-sample similarity: {test_b['intra_similarity']:.3f}")
    print(f"   Inter-sample similarity: {test_b['inter_similarity']:.3f}")
    print(f"   Effective rank: {test_b['effective_rank']:.1f} / 4096")
    if test_b['inter_similarity'] > 0.8:
        print("   ❌ CRITICAL: Mode collapse - all questions look the same")
    elif test_b['effective_rank'] < 10:
        print("   ❌ CRITICAL: Severe dimensional collapse")
    else:
        print("   ✓  Tokens are reasonably diverse")

    if "test_c" in all_results:
        print("\n3. LAYER COMPARISON")
        test_c = all_results["test_c"]
        l16_acc = 100 * test_c["layer_16_accuracy"] / test_c["total"]
        l31_acc = 100 * test_c["layer_31_accuracy"] / test_c["total"]
        print(f"   Layer 16: {l16_acc:.1f}% | Layer 31: {l31_acc:.1f}%")
        if l31_acc > 2 * l16_acc:
            print("   ✓  Layer 31 is better for reasoning (as expected)")
        else:
            print("   ⚠️  No clear layer advantage")

    print("\n4. ENTITY PRESERVATION")
    test_d = all_results["test_d"]
    mean_recall = np.mean(test_d["number_recall"]) if test_d["number_recall"] else 0
    digit_pct = 100 * test_d["samples_with_digits"] / test_d["total"] if test_d["total"] > 0 else 0
    print(f"   Number recall: {100*mean_recall:.1f}%")
    print(f"   Samples with digits: {digit_pct:.1f}%")
    if mean_recall < 0.1:
        print("   ❌ CRITICAL: Numbers are lost - explains GSM8K failure")
    else:
        print("   ✓  Numbers are partially preserved")

    print("\n5. CHAIN PRESERVATION")
    test_e = all_results["test_e"]
    step_overlap = np.mean(test_e["step_overlap"]) if test_e["step_overlap"] else 0
    op_match = np.mean(test_e["operation_match"]) if test_e["operation_match"] else 0
    print(f"   Step overlap: {100*step_overlap:.1f}%")
    print(f"   Operation match: {100*op_match:.1f}%")
    if step_overlap < 0.1:
        print("   ❌ CRITICAL: Reasoning chain is lost")
    else:
        print("   ✓  Partial chain preservation")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    recommendations = []

    if latent_acc < 5:
        recommendations.append("• Increase soft tokens (8→16 or 8→32)")
        recommendations.append("• Add contrastive loss to improve discrimination")

    if test_b['inter_similarity'] > 0.8:
        recommendations.append("• Mode collapse detected - add diversity regularization")
        recommendations.append("• Consider contrastive learning (InfoNCE)")

    if mean_recall < 0.1:
        recommendations.append("• Bridge loses numbers - add entity-aware loss")
        recommendations.append("• Try explicit number token supervision")

    if step_overlap < 0.1:
        recommendations.append("• Multi-step reasoning lost - need more capacity")
        recommendations.append("• Consider multi-stage bridge (one soft token per step)")

    if "test_c" in all_results and test_c["layer_31_accuracy"] > 2 * test_c["layer_16_accuracy"]:
        recommendations.append("• Use Layer 31 instead of Layer 16 for reasoning tasks")

    if not recommendations:
        recommendations.append("• No clear bottleneck identified - problem may be in training")
        recommendations.append("• Consider longer training or different optimization")

    for rec in recommendations:
        print(rec)

    print("\n" + "="*70)
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
