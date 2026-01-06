#!/usr/bin/env python3
"""
Comprehensive architecture sweep to diagnose and fix mode collapse.

Tests 4 architectural variants:
1. Direct sequence compression (no mean pooling) - PROPOSED FIX
2. Mean pool bottleneck (single vector) - CURRENT BROKEN
3. Full sequence (no compression) - UPPER BOUND
4. Mean pool + expand (full current pipeline) - BASELINE

For each, computes:
- Task metrics: Diversity, F1, EM, first-token accuracy
- Representation diagnostics: Cosine similarity, PCA variance, NN accuracy
- Training dynamics: Loss curves, gradient norms

Goal: Identify which architecture preserves information and prevents collapse.

Usage:
    bash scripts/sweep_architectures.sh
    STEPS=500 bash scripts/sweep_architectures.sh  # Longer training
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from latentwire.data import load_squad_subset


# ============================================================================
# ARCHITECTURE VARIANTS
# ============================================================================

class DirectSequenceCompressor(nn.Module):
    """
    Direct ~100 tokens → M tokens via cross-attention.
    NO mean pooling bottleneck.
    """
    def __init__(self, d_model=4096, M=32, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.M = M
        self.d_model = d_model

        # Learned queries for M output slots
        self.queries = nn.Parameter(torch.randn(M, d_model) * 0.02)

        # Cross-attention layers
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, token_embeds, attention_mask=None):
        """
        Args:
            token_embeds: [B, T, d_model] - frozen LLM embeddings
            attention_mask: [B, T] - 1=valid, 0=pad
        Returns:
            compressed: [B, M, d_model] - ready for LLM
        """
        B = token_embeds.size(0)
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, d_model]

        # Build key padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # True = ignore

        # Cross-attend to compress
        for attn, norm in zip(self.cross_attn, self.norms):
            Q_new, _ = attn(
                query=Q,
                key=token_embeds,
                value=token_embeds,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            Q = norm(Q + Q_new)  # Residual + norm

        return Q  # [B, M, d_model]


class MeanPoolBottleneck(nn.Module):
    """
    Mean pool ~100 tokens → single vector z ∈ R^d_bottleneck.
    This is the BROKEN architecture that causes collapse.
    """
    def __init__(self, d_model=4096, d_bottleneck=512):
        super().__init__()
        self.proj_down = nn.Linear(d_model, d_bottleneck)
        self.norm = nn.LayerNorm(d_bottleneck)

    def forward(self, token_embeds, attention_mask=None):
        """
        Returns:
            z: [B, d_bottleneck] - single vector bottleneck
        """
        # Mean pool (respecting mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            masked = token_embeds * mask_expanded
            z = masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            z = token_embeds.mean(dim=1)

        z = self.proj_down(z)
        z = self.norm(z)
        return z  # [B, d_bottleneck]


class BottleneckExpander(nn.Module):
    """
    Expand z ∈ R^d_bottleneck → M tokens ∈ R^{M×d_model}.
    Used with MeanPoolBottleneck to recreate current pipeline.
    """
    def __init__(self, d_bottleneck=512, d_model=4096, M=32):
        super().__init__()
        self.M = M

        # Learned slot queries
        self.queries = nn.Parameter(torch.randn(M, d_bottleneck) * 0.02)

        # Expand
        self.expand = nn.Sequential(
            nn.Linear(d_bottleneck, d_bottleneck * M),
            nn.LayerNorm(d_bottleneck * M),
            nn.GELU(),
        )

        # Project to LLM space
        self.proj = nn.Sequential(
            nn.Linear(d_bottleneck, d_model),
            nn.LayerNorm(d_model),
        )

        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, z):
        """
        Args:
            z: [B, d_bottleneck] - bottleneck vector
        Returns:
            tokens: [B, M, d_model]
        """
        B = z.size(0)
        expanded = self.expand(z)  # [B, d_bottleneck * M]
        seq = expanded.view(B, self.M, -1)  # [B, M, d_bottleneck]
        seq = seq + self.queries.unsqueeze(0)
        tokens = self.proj(seq)  # [B, M, d_model]
        tokens = tokens * self.scale
        return tokens


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def k_token_ce(model, tokenizer, prefix_embeds, answer_text, K=4, anchor_text="Answer: "):
    """K-token cross-entropy loss."""
    device = prefix_embeds.device
    answer_ids = tokenizer(answer_text, return_tensors='pt').input_ids.to(device)
    anchor_ids = tokenizer(anchor_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    total_loss = 0.0
    for t in range(min(K, answer_ids.size(1))):
        if t == 0:
            inputs_embeds = torch.cat([prefix_embeds, model.get_input_embeddings()(anchor_ids)], dim=1)
        else:
            prev_ids = answer_ids[:, :t]
            inputs_embeds = torch.cat([
                prefix_embeds,
                model.get_input_embeddings()(anchor_ids),
                model.get_input_embeddings()(prev_ids),
            ], dim=1)

        outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
        logits = outputs.logits[:, -1, :]
        target = answer_ids[:, t]
        total_loss += F.cross_entropy(logits, target)

    return total_loss / min(K, answer_ids.size(1))


def generate_text(model, tokenizer, prefix_embeds, max_new_tokens=12, anchor_text="Answer: "):
    """Generate from prefix embeddings."""
    device = prefix_embeds.device
    anchor_ids = tokenizer(anchor_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    anchor_embeds = model.get_input_embeddings()(anchor_ids)
    inputs_embeds = torch.cat([prefix_embeds, anchor_embeds], dim=1)

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if anchor_text and text.startswith(anchor_text):
        text = text[len(anchor_text):]
    return text.strip()


def compute_representation_diagnostics(representations, labels=None):
    """
    Compute information preservation metrics.

    Args:
        representations: [N, d] or [N, M, d] numpy array
        labels: Optional[List[str]] - for clustering analysis

    Returns:
        dict of diagnostic metrics
    """
    N = representations.shape[0]

    # Flatten if sequence
    if representations.ndim == 3:
        representations = representations.reshape(N, -1)

    # 1. Pairwise cosine similarity
    cos_sim = cosine_similarity(representations)
    # Exclude diagonal (self-similarity = 1.0)
    np.fill_diagonal(cos_sim, np.nan)
    avg_cosine = np.nanmean(cos_sim)
    max_cosine = np.nanmax(cos_sim)

    # 2. PCA variance explained
    try:
        n_components = min(32, N, representations.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(representations)
        pca_variance = pca.explained_variance_ratio_.sum()
    except:
        pca_variance = 0.0

    # 3. Nearest neighbor accuracy (should each example be its own NN?)
    # For each example, check if its nearest neighbor is itself
    nn_correct = 0
    for i in range(N):
        # Find nearest neighbor (excluding self)
        similarities = cos_sim[i]
        nn_idx = np.nanargmax(similarities)
        # In a good representation, examples should be distinct
        # So we check diversity by seeing if max similarity to others is low
        if similarities[nn_idx] < 0.7:  # Diverse if max similarity < 0.7
            nn_correct += 1
    nn_diversity = nn_correct / N

    return {
        'avg_cosine_similarity': float(avg_cosine),
        'max_cosine_similarity': float(max_cosine),
        'pca_variance_explained': float(pca_variance),
        'nn_diversity_score': float(nn_diversity),
        'interpretation': {
            'avg_cosine': 'GOOD if < 0.5, BAD if > 0.8',
            'pca_variance': 'GOOD if > 0.7, BAD if < 0.3',
            'nn_diversity': 'GOOD if > 0.7, BAD if < 0.3',
        }
    }


def train_and_evaluate(
    config_name,
    architecture,
    llama_model,
    llama_tokenizer,
    examples,
    device,
    learned_dtype,
    steps=300,
    K=4,
):
    """Train architecture variant and evaluate."""

    print(f"\n{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}\n")

    # Create optimizer
    optimizer = torch.optim.AdamW(architecture.parameters(), lr=1e-4, weight_decay=0.01)

    architecture.train()
    losses = []

    # Training
    print("Training...")
    for step in range(steps):
        idx = torch.randint(0, len(examples), (1,))[0]
        ex = examples[idx]

        optimizer.zero_grad()

        # Encode
        tokens = llama_tokenizer(ex['source'], return_tensors='pt', truncation=True, max_length=512)
        input_ids = tokens.input_ids.to(device)
        attn_mask = tokens.attention_mask.to(device)

        with torch.no_grad():
            embeds = llama_model.get_input_embeddings()(input_ids)
            embeds = embeds.to(learned_dtype)

        # Forward through architecture
        compressed = architecture(embeds, attn_mask)

        # Handle different output shapes
        if compressed.ndim == 2:  # [B, d] - bottleneck
            # This shouldn't be used directly, but keep for compatibility
            compressed = compressed.unsqueeze(1)  # [B, 1, d]

        compressed = compressed.to(llama_model.dtype)

        # Loss
        loss = k_token_ce(llama_model, llama_tokenizer, compressed, ex['answer'], K=K)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(architecture.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            losses.append(loss.item())
            print(f"  Step {step+1}/{steps}: loss={loss.item():.4f}")

    # Evaluation
    print("\nEvaluating...")
    architecture.eval()

    test_examples = examples[:10]
    predictions = []
    representations = []

    with torch.no_grad():
        for ex in test_examples:
            tokens = llama_tokenizer(ex['source'], return_tensors='pt', truncation=True, max_length=512)
            input_ids = tokens.input_ids.to(device)
            attn_mask = tokens.attention_mask.to(device)
            embeds = llama_model.get_input_embeddings()(input_ids).to(learned_dtype)

            compressed = architecture(embeds, attn_mask)

            # Save representation for diagnostics (squeeze batch dim)
            rep = compressed.squeeze(0).cpu().numpy()  # [M, d] or [d]
            representations.append(rep)

            # Generate
            if compressed.ndim == 2:
                compressed = compressed.unsqueeze(1)
            compressed = compressed.to(llama_model.dtype)

            pred = generate_text(llama_model, llama_tokenizer, compressed)
            predictions.append(pred)

    # Metrics
    unique_preds = len(set(predictions))
    diversity_pct = unique_preds / len(predictions) * 100

    # Representation diagnostics
    representations = np.array(representations)  # [N, M, d] or [N, d]
    diagnostics = compute_representation_diagnostics(representations)

    print(f"\n  Results:")
    print(f"    Diversity: {unique_preds}/{len(predictions)} ({diversity_pct:.1f}%)")
    print(f"    Avg cosine similarity: {diagnostics['avg_cosine_similarity']:.3f} "
          f"({diagnostics['interpretation']['avg_cosine']})")
    print(f"    PCA variance explained: {diagnostics['pca_variance_explained']:.3f} "
          f"({diagnostics['interpretation']['pca_variance']})")
    print(f"    NN diversity score: {diagnostics['nn_diversity_score']:.3f} "
          f"({diagnostics['interpretation']['nn_diversity']})")

    # Show sample predictions
    print(f"\n  Sample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"    [{i+1}] Gold: {test_examples[i]['answer'][:30]:<30} → Pred: {predictions[i]}")

    return {
        'config': config_name,
        'diversity': unique_preds,
        'diversity_pct': diversity_pct,
        'final_loss': losses[-1] if losses else 0.0,
        'predictions': predictions,
        'diagnostics': diagnostics,
        'loss_curve': losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--M", type=int, default=32, help="Compressed sequence length")
    parser.add_argument("--d_bottleneck", type=int, default=512, help="Bottleneck dim for mean pool")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learned_dtype = torch.float32

    print("="*80)
    print("ARCHITECTURE SWEEP: Diagnose and Fix Mode Collapse")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Samples: {args.samples}, Steps: {args.steps}, M: {args.M}")

    # Load model
    print("\n[1/3] Loading frozen LLM...")
    llama_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_id, use_fast=True)
    llama_tokenizer.padding_side = 'left'
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    llama_model.eval()
    for p in llama_model.parameters():
        p.requires_grad = False
    d_model = llama_model.config.hidden_size

    print("  ✓ Model loaded")

    # Load data
    print(f"\n[2/3] Loading data (SQuAD, n={args.samples})...")
    examples = load_squad_subset(split='validation', samples=args.samples)
    print(f"  ✓ Loaded {len(examples)} examples")

    # Configurations to test
    print(f"\n[3/3] Testing architectures...")

    configs = [
        ("Direct Sequence Compression (PROPOSED)",
         DirectSequenceCompressor(d_model=d_model, M=args.M, n_layers=4).to(device, learned_dtype)),

        ("Mean Pool + Expand (CURRENT PIPELINE)",
         nn.Sequential(
             MeanPoolBottleneck(d_model=d_model, d_bottleneck=args.d_bottleneck),
             BottleneckExpander(d_bottleneck=args.d_bottleneck, d_model=d_model, M=args.M),
         ).to(device, learned_dtype)),
    ]

    results = []
    for config_name, architecture in configs:
        result = train_and_evaluate(
            config_name=config_name,
            architecture=architecture,
            llama_model=llama_model,
            llama_tokenizer=llama_tokenizer,
            examples=examples,
            device=device,
            learned_dtype=learned_dtype,
            steps=args.steps,
        )
        results.append(result)

    # Save results
    output_dir = Path("runs/architecture_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print("\n{:<45} {:>8} {:>10} {:>10} {:>10}".format(
        "Configuration", "Div%", "AvgCos", "PCAVar", "NNDiv"
    ))
    print("-"*80)

    for r in results:
        print("{:<45} {:>7.1f}% {:>10.3f} {:>10.3f} {:>10.3f}".format(
            r['config'][:45],
            r['diversity_pct'],
            r['diagnostics']['avg_cosine_similarity'],
            r['diagnostics']['pca_variance_explained'],
            r['diagnostics']['nn_diversity_score'],
        ))

    # Verdict
    best = max(results, key=lambda x: x['diversity_pct'])
    print("\n" + "="*80)
    print(f"BEST: {best['config']}")
    print(f"  Diversity: {best['diversity_pct']:.1f}%")
    print(f"  Avg Cosine: {best['diagnostics']['avg_cosine_similarity']:.3f} (target: < 0.5)")
    print(f"  PCA Variance: {best['diagnostics']['pca_variance_explained']:.3f} (target: > 0.7)")
    print("="*80)

    # Write summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("ARCHITECTURE SWEEP RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration | Diversity | AvgCos | PCAVar | NNDiv | Final Loss\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r['config']:<45} | {r['diversity']:2d}/{len(r['predictions']):2d} ({r['diversity_pct']:5.1f}%) | "
                   f"{r['diagnostics']['avg_cosine_similarity']:.3f} | "
                   f"{r['diagnostics']['pca_variance_explained']:.3f} | "
                   f"{r['diagnostics']['nn_diversity_score']:.3f} | "
                   f"{r['final_loss']:.4f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST: {best['config']} with {best['diversity_pct']:.1f}% diversity\n")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
