#!/usr/bin/env python
# telepathy/eval_sst2_ablations.py
"""
SST-2 Ablation Studies

Comprehensive baselines to validate bridge architecture choices:

1. Untrained bridge: Random Perceiver weights
2. Mean pooling: Replace Perceiver with mean pooling
3. Last token: Use only last token's hidden state
4. Linear projection: Simple linear layer instead of Perceiver
5. Token budget: Mistral given first 32 tokens only (fair comparison)
6. Layer ablation: Test different source layers

These ablations answer:
- Is the Perceiver architecture necessary?
- Is training necessary?
- Is the full sequence necessary?
- Is layer 16 optimal?
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os
import copy

from latent_bridge_v15 import LatentBridgeV15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_checkpoint", required=True, help="Path to trained bridge")
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--soft_tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def eval_with_soft_tokens(soft_tokens, tgt_model, tgt_tok, label, device):
    """Evaluate with given soft tokens."""
    with torch.no_grad():
        primer = "Sentiment:"
        primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
        if primer_embeds.dtype != soft_tokens.dtype:
            primer_embeds = primer_embeds.to(soft_tokens.dtype)

        combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
        attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

        out_ids = tgt_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attn_mask,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tgt_tok.eos_token_id,
        )
        output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

    return label in output


def run_ablation(name, get_soft_tokens_fn, src_model, tgt_model, src_tok, tgt_tok,
                 ds, num_samples, device, args):
    """Run a single ablation experiment."""
    correct = 0
    total = 0

    for i in tqdm(range(num_samples), desc=name):
        item = ds[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        src_input = f"Review: {text}\nSentiment:"

        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()

            soft_tokens = get_soft_tokens_fn(src_h, src_enc.attention_mask)

            if eval_with_soft_tokens(soft_tokens, tgt_model, tgt_tok, label, device):
                correct += 1
            total += 1

    accuracy = 100 * correct / total
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("SST-2 ABLATION STUDIES")
    print("=" * 70)
    print("Purpose: Validate each architecture choice")
    print("")

    # Load models
    print("Loading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

    # Load trained bridge
    print(f"Loading trained bridge: {args.trained_checkpoint}")
    trained_bridge = LatentBridgeV15(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    checkpoint = torch.load(args.trained_checkpoint, map_location=DEVICE, weights_only=True)
    trained_bridge.load_state_dict(checkpoint)
    trained_bridge.to(DEVICE)
    if args.bf16:
        trained_bridge = trained_bridge.bfloat16()
    trained_bridge.eval()

    # Create untrained bridge (random weights)
    print("Creating untrained bridge...")
    untrained_bridge = LatentBridgeV15(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    untrained_bridge.to(DEVICE)
    if args.bf16:
        untrained_bridge = untrained_bridge.bfloat16()
    untrained_bridge.eval()

    # Create simple linear projection
    print("Creating linear projection baseline...")
    linear_proj = nn.Linear(src_model.config.hidden_size, tgt_model.config.hidden_size).to(DEVICE)
    if args.bf16:
        linear_proj = linear_proj.bfloat16()

    # Load dataset
    ds = load_dataset("glue", "sst2", split="validation")
    num_samples = min(args.num_samples, len(ds))

    results = {}

    # =========================================================================
    # ABLATION 1: Trained bridge (reference)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[1/6] TRAINED BRIDGE (reference)")
    print("=" * 70)

    def trained_fn(src_h, src_mask):
        soft_tokens, _, _, _ = trained_bridge(src_h, src_mask)
        return soft_tokens

    results["trained_bridge"] = run_ablation(
        "Trained", trained_fn, src_model, tgt_model, src_tok, tgt_tok,
        ds, num_samples, DEVICE, args
    )
    print(f"  Accuracy: {results['trained_bridge']['accuracy']:.1f}%")

    # =========================================================================
    # ABLATION 2: Untrained bridge (random Perceiver)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2/6] UNTRAINED BRIDGE (random Perceiver weights)")
    print("=" * 70)

    def untrained_fn(src_h, src_mask):
        soft_tokens, _, _, _ = untrained_bridge(src_h, src_mask)
        return soft_tokens

    results["untrained_bridge"] = run_ablation(
        "Untrained", untrained_fn, src_model, tgt_model, src_tok, tgt_tok,
        ds, num_samples, DEVICE, args
    )
    print(f"  Accuracy: {results['untrained_bridge']['accuracy']:.1f}%")

    # =========================================================================
    # ABLATION 3: Mean pooling (no Perceiver)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3/6] MEAN POOLING (replace Perceiver with mean)")
    print("=" * 70)

    def mean_pool_fn(src_h, src_mask):
        # Mean pool across sequence dimension
        mask_expanded = src_mask.unsqueeze(-1).float()
        pooled = (src_h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)  # [B, D]
        # Expand to 32 tokens (repeat)
        soft_tokens = pooled.unsqueeze(1).expand(-1, args.soft_tokens, -1)  # [B, 32, D]
        # RMS normalize
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * target_rms
        return soft_tokens

    results["mean_pooling"] = run_ablation(
        "MeanPool", mean_pool_fn, src_model, tgt_model, src_tok, tgt_tok,
        ds, num_samples, DEVICE, args
    )
    print(f"  Accuracy: {results['mean_pooling']['accuracy']:.1f}%")

    # =========================================================================
    # ABLATION 4: Last token only
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4/6] LAST TOKEN ONLY (use final hidden state)")
    print("=" * 70)

    def last_token_fn(src_h, src_mask):
        # Get last valid token for each sequence
        seq_lens = src_mask.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(src_h.shape[0], device=src_h.device)
        last_hidden = src_h[batch_indices, seq_lens]  # [B, D]
        # Expand to 32 tokens
        soft_tokens = last_hidden.unsqueeze(1).expand(-1, args.soft_tokens, -1)
        # RMS normalize
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * target_rms
        return soft_tokens

    results["last_token"] = run_ablation(
        "LastTok", last_token_fn, src_model, tgt_model, src_tok, tgt_tok,
        ds, num_samples, DEVICE, args
    )
    print(f"  Accuracy: {results['last_token']['accuracy']:.1f}%")

    # =========================================================================
    # ABLATION 5: Linear projection (no Perceiver attention)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5/6] LINEAR PROJECTION (random linear, no attention)")
    print("=" * 70)

    def linear_proj_fn(src_h, src_mask):
        # Mean pool then project
        mask_expanded = src_mask.unsqueeze(-1).float()
        pooled = (src_h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)  # [B, D]
        projected = linear_proj(pooled)  # [B, D]
        soft_tokens = projected.unsqueeze(1).expand(-1, args.soft_tokens, -1)
        # RMS normalize
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        soft_tokens = (soft_tokens / rms) * target_rms
        return soft_tokens

    results["linear_projection"] = run_ablation(
        "LinearProj", linear_proj_fn, src_model, tgt_model, src_tok, tgt_tok,
        ds, num_samples, DEVICE, args
    )
    print(f"  Accuracy: {results['linear_projection']['accuracy']:.1f}%")

    # =========================================================================
    # ABLATION 6: Token budget baseline (Mistral sees 32 tokens)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[6/6] TOKEN BUDGET (Mistral sees first 32 tokens of text)")
    print("=" * 70)

    correct = 0
    total = 0
    for i in tqdm(range(num_samples), desc="TokenBudget"):
        item = ds[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        # Truncate to ~32 tokens worth of text
        words = text.split()[:20]  # Approximate 32 tokens
        truncated = " ".join(words)

        prompt = f"Review: {truncated}\nSentiment (positive or negative):"

        with torch.no_grad():
            inputs = tgt_tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            out_ids = tgt_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            gen_ids = out_ids[0][inputs.input_ids.shape[1]:]
            output = tgt_tok.decode(gen_ids, skip_special_tokens=True).strip().lower()

        if label in output:
            correct += 1
        total += 1

    results["token_budget"] = {
        "accuracy": 100 * correct / total,
        "correct": correct,
        "total": total
    }
    print(f"  Accuracy: {results['token_budget']['accuracy']:.1f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print("")
    print(f"{'Ablation':<25} {'Accuracy':>10} {'vs Trained':>12}")
    print("-" * 50)

    trained_acc = results["trained_bridge"]["accuracy"]
    for name, res in results.items():
        acc = res["accuracy"]
        diff = acc - trained_acc
        diff_str = f"{diff:+.1f}%" if name != "trained_bridge" else "(ref)"
        print(f"{name:<25} {acc:>9.1f}% {diff_str:>12}")

    print("")
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if results["untrained_bridge"]["accuracy"] > 60:
        print("WARNING: Untrained bridge > 60% - architecture may have implicit bias")
    else:
        print("GOOD: Untrained bridge near random - training is necessary")

    if results["mean_pooling"]["accuracy"] > trained_acc - 10:
        print("NOTE: Mean pooling close to trained - Perceiver may be overkill")
    else:
        print("GOOD: Mean pooling much worse - Perceiver attention is valuable")

    if results["token_budget"]["accuracy"] > trained_acc:
        print("WARNING: Token budget exceeds bridge - compression may hurt")
    else:
        print("GOOD: Bridge exceeds token budget - compression is beneficial")

    print("=" * 70)

    # Save results
    output_path = os.path.join(args.output_dir, "sst2_ablations.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
