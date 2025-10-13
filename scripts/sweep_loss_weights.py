#!/usr/bin/env python3
"""
Sweep different loss weight configurations to diagnose mode collapse.

Tests multiple configurations in one run:
1. No semantic loss (the "buggy" version that had 3/5 diversity)
2. Very weak semantic (0.01)
3. Weak semantic (0.05)
4. Medium semantic (0.1)
5. Strong semantic (0.5) - current
6. Increased K-token supervision (K=8 vs K=4)

Each configuration trains for 300 steps and evaluates diversity.
Results saved to runs/loss_weight_sweep/results.txt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

from latentwire.data import load_squad_subset

# Import architecture from test script
import sys
sys.path.append('scripts')
from test_new_interlingua import AlignmentTransformer, InterlinguaAdapter, k_token_ce_simple, generate_from_prefix


def train_and_evaluate(
    config_name: str,
    sem_weight: float,
    K: int,
    d_model_llama: int,
    d_sem: int,
    d_inter: int,
    num_slots: int,
    llama_model,
    llama_tokenizer,
    sem_encoder,
    examples,
    device,
    learned_dtype,
    steps=300,
    lr=1e-4,
):
    """Train with given configuration and evaluate diversity."""

    print(f"\n{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"  sem_weight={sem_weight}, K={K}")
    print(f"{'='*80}\n")

    # Create fresh model instances (proper reset)
    alignment_tf = AlignmentTransformer(
        d_model_llama=d_model_llama,
        d_model_qwen=2048,
        d_sem=d_sem,
        d_inter=d_inter,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
    ).to(device=device, dtype=learned_dtype)

    adapter_llama = InterlinguaAdapter(
        d_inter=d_inter,
        d_model=d_model_llama,
        num_slots=num_slots,
        dropout=0.1,
    ).to(device=device, dtype=learned_dtype)

    # Create optimizer
    params = list(alignment_tf.parameters()) + list(adapter_llama.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # Training
    alignment_tf.train()
    adapter_llama.train()

    losses_log = []

    for step in range(steps):
        # Sample a batch
        batch_indices = torch.randint(0, len(examples), (1,))
        ex = examples[batch_indices[0]]

        text = ex['source']
        answer = ex['answer']

        optimizer.zero_grad()

        # Encode with semantic encoder
        with torch.no_grad():
            z_sem = torch.tensor(
                sem_encoder.encode([text], convert_to_tensor=False, show_progress_bar=False),
                dtype=learned_dtype,
                device=device,
            )

        # Llama
        llama_tokens = llama_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        llama_input_ids = llama_tokens.input_ids.to(device)
        llama_attn_mask = llama_tokens.attention_mask.to(device)

        with torch.no_grad():
            llama_embeds = llama_model.get_input_embeddings()(llama_input_ids)
            llama_embeds = llama_embeds.to(learned_dtype)

        z_llama = alignment_tf(llama_embeds, z_sem, 'llama', llama_attn_mask)
        prefix_llama = adapter_llama(z_llama)
        prefix_llama = prefix_llama.to(llama_model.dtype)

        # Generation loss
        loss_gen = k_token_ce_simple(
            llama_model, llama_tokenizer, prefix_llama, answer, K=K, anchor_text="Answer: "
        )

        # Semantic anchor loss (if weight > 0)
        if sem_weight > 0:
            z_sem_proj = alignment_tf.proj_sem(z_sem)
            loss_sem = F.mse_loss(z_llama, z_sem_proj)
            loss = loss_gen + sem_weight * loss_sem
        else:
            loss_sem = torch.tensor(0.0)
            loss = loss_gen

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        # Log
        if step % 50 == 0 or step == steps - 1:
            log_entry = {
                'step': step + 1,
                'loss': loss.item(),
                'gen': loss_gen.item(),
                'sem': loss_sem.item() if isinstance(loss_sem, torch.Tensor) else 0.0,
            }
            losses_log.append(log_entry)
            print(f"  Step {step+1}/{steps}: loss={loss.item():.4f} gen={loss_gen.item():.4f} sem={log_entry['sem']:.4f}")

    # Evaluation
    alignment_tf.eval()
    adapter_llama.eval()

    test_examples = examples[:10]  # Test on 10 examples for better diversity metric
    preds = []

    print("\n  Predictions:")
    for i, ex in enumerate(test_examples):
        text = ex['source']
        answer = ex['answer']

        with torch.no_grad():
            z_sem = torch.tensor(
                sem_encoder.encode([text], convert_to_tensor=False, show_progress_bar=False),
                dtype=learned_dtype,
                device=device,
            )

            llama_tokens = llama_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            llama_input_ids = llama_tokens.input_ids.to(device)
            llama_attn_mask = llama_tokens.attention_mask.to(device)
            llama_embeds = llama_model.get_input_embeddings()(llama_input_ids)
            llama_embeds = llama_embeds.to(learned_dtype)

            z_llama = alignment_tf(llama_embeds, z_sem, 'llama', llama_attn_mask)
            prefix_llama = adapter_llama(z_llama)
            prefix_llama = prefix_llama.to(llama_model.dtype)

        pred = generate_from_prefix(llama_model, llama_tokenizer, prefix_llama, anchor_text="Answer: ", max_new_tokens=12)
        preds.append(pred)

        if i < 5:  # Show first 5
            print(f"    [{i+1}] Gold: {answer[:30]:<30} → Pred: {pred}")

    unique_preds = len(set(preds))
    diversity_pct = unique_preds / len(preds) * 100

    print(f"\n  Diversity: {unique_preds}/{len(preds)} unique ({diversity_pct:.1f}%)")

    return {
        'config': config_name,
        'sem_weight': sem_weight,
        'K': K,
        'diversity': unique_preds,
        'diversity_pct': diversity_pct,
        'total_examples': len(preds),
        'final_loss': losses_log[-1]['loss'],
        'final_gen_loss': losses_log[-1]['gen'],
        'final_sem_loss': losses_log[-1]['sem'],
        'losses_log': losses_log,
        'predictions': preds,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep loss weight configurations")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--steps", type=int, default=300, help="Steps per configuration")
    parser.add_argument("--d_inter", type=int, default=512, help="Interlingua dimension")
    parser.add_argument("--num_slots", type=int, default=32, help="Number of soft token slots")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learned_dtype = torch.float32

    print("="*80)
    print("LOSS WEIGHT SWEEP")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Samples: {args.samples}, Steps per config: {args.steps}")

    # Load models
    print("\n[1/2] Loading frozen models...")

    sem_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sem_encoder.eval()
    for p in sem_encoder.parameters():
        p.requires_grad = False
    d_sem = 384

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
    d_model_llama = llama_model.config.hidden_size

    print("  ✓ Models loaded")

    # Load data
    print(f"\n[2/3] Loading test data (SQuAD, n={args.samples})...")
    examples = load_squad_subset(split='validation', samples=args.samples)
    print(f"  ✓ Loaded {len(examples)} examples")

    # Configurations to test
    configs = [
        ("No semantic loss (buggy version)", 0.0, 4),
        ("Very weak semantic (0.01)", 0.01, 4),
        ("Weak semantic (0.05)", 0.05, 4),
        ("Medium semantic (0.1)", 0.1, 4),
        ("Strong semantic (0.5)", 0.5, 4),
        ("Increased K-token (K=8)", 0.05, 8),
        ("Increased K-token (K=12)", 0.05, 12),
    ]

    results = []

    for config_name, sem_weight, K in configs:
        result = train_and_evaluate(
            config_name=config_name,
            sem_weight=sem_weight,
            K=K,
            d_model_llama=d_model_llama,
            d_sem=d_sem,
            d_inter=args.d_inter,
            num_slots=args.num_slots,
            llama_model=llama_model,
            llama_tokenizer=llama_tokenizer,
            sem_encoder=sem_encoder,
            examples=examples,
            device=device,
            learned_dtype=learned_dtype,
            steps=args.steps,
        )
        results.append(result)

    # Save results
    output_dir = Path("runs/loss_weight_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print("\n{:<40} {:>8} {:>6} {:>10} {:>10}".format(
        "Configuration", "K", "Div%", "Loss", "Gen Loss"
    ))
    print("-"*80)

    for r in results:
        print("{:<40} {:>8} {:>5.1f}% {:>10.4f} {:>10.4f}".format(
            r['config'][:40],
            r['K'],
            r['diversity_pct'],
            r['final_loss'],
            r['final_gen_loss'],
        ))

    # Find best configuration
    best = max(results, key=lambda x: x['diversity_pct'])
    print("\n" + "="*80)
    print(f"BEST: {best['config']} with {best['diversity_pct']:.1f}% diversity")
    print("="*80)

    # Save text summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write("LOSS WEIGHT SWEEP RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration | K | Diversity | Final Loss | Gen Loss | Sem Loss\n")
        f.write("-"*80 + "\n")

        for r in results:
            f.write(f"{r['config']:<40} | {r['K']:2d} | {r['diversity']:2d}/{r['total_examples']:2d} ({r['diversity_pct']:5.1f}%) | "
                   f"{r['final_loss']:8.4f} | {r['final_gen_loss']:8.4f} | {r['final_sem_loss']:8.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST: {best['config']} with {best['diversity_pct']:.1f}% diversity\n")
        f.write(f"  K={best['K']}, sem_weight={best['sem_weight']}\n")
        f.write(f"  Final loss: {best['final_loss']:.4f}\n")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  - results.json (full data)")
    print(f"  - summary.txt (readable summary)")


if __name__ == "__main__":
    main()
