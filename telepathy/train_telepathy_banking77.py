#!/usr/bin/env python
# telepathy/train_telepathy_banking77.py
"""
Phase 21: Banking77 (Bandwidth Stress Test)

GOAL: Test if 8-16 soft tokens can handle 77-class classification.
This is 19x more classes than AG News (4 classes).

If this succeeds, the bridge has massive capacity.
If this fails, we need more tokens for fine-grained tasks.

Dataset: PolyAI/banking77
- 77 intent classes for banking queries
- ~10,000 training examples
"""
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from latent_bridge_v15 import LatentBridgeV15


class Args:
    """Args object for LatentBridgeV15 interface."""
    def __init__(self, soft_tokens=16, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def evaluate(bridge, src_model, tgt_model, src_tok, tgt_tok, labels, test_data, device, num_samples=200):
    """Evaluate on test set."""
    bridge.eval()
    correct = 0
    total = 0

    primer = "Intent: "
    primer_tokens = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        primer_embeds = tgt_model.get_input_embeddings()(primer_tokens.input_ids)

    results = []

    for i, item in enumerate(test_data):
        if i >= num_samples:
            break

        text = item['text']
        label_idx = item['label']
        label = labels[label_idx]

        src_inputs = src_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[31]
            src_mask = src_inputs.attention_mask

            latents, _, _, _ = bridge(src_hidden, src_mask)

            combined = torch.cat([primer_embeds, latents], dim=1)
            attn_mask = torch.ones(1, combined.shape[1], device=device)

            out_ids = tgt_model.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=20,
                pad_token_id=tgt_tok.eos_token_id,
                do_sample=False
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).lower()

        # Check if any label matches (labels are like "card_arrival", "pin_blocked")
        pred_label = None
        for l in labels:
            if l.lower().replace("_", " ") in output or l.lower() in output:
                pred_label = l
                break

        is_correct = (label.lower() in output) or (label.lower().replace("_", " ") in output)
        if is_correct:
            correct += 1

        if i < 5:
            results.append(f"GT: {label} | Output: {output[:60]}")

        total += 1

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {acc:.1f}% ({correct}/{total})")
    print("Samples:")
    for r in results:
        print(f"  {r}")

    return {"accuracy": acc, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/banking77")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--soft_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("BANKING77 EXPERIMENT (77-class Bandwidth Test)")
    print("=" * 60)
    print(f"Classes: 77 (vs AG News: 4)")
    print(f"Soft tokens: {args.soft_tokens}")
    print(f"Steps: {args.steps}")
    print("=" * 60)

    # Load dataset
    print("\nLoading Banking77...")
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    labels = dataset['train'].features['label'].names
    print(f"Classes: {len(labels)}")
    print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    # Load models
    print("\nLoading models...")
    src_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token
    src_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16
    ).to(device)
    src_model.eval()

    tgt_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16
    ).to(device)
    tgt_model.eval()

    # Freeze LLMs
    for p in src_model.parameters():
        p.requires_grad = False
    for p in tgt_model.parameters():
        p.requires_grad = False

    # Init bridge
    bridge_args = Args(soft_tokens=args.soft_tokens, heads=8, depth=2, use_fsq=False)
    bridge = LatentBridgeV15(bridge_args, src_dim=4096, tgt_dim=4096, target_rms=0.03)
    bridge = bridge.to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr)

    # Primer
    primer = "Intent: "
    primer_tokens = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        primer_embeds_single = tgt_model.get_input_embeddings()(primer_tokens.input_ids)
        primer_embeds = primer_embeds_single.repeat(args.batch_size, 1, 1)

    # Training
    loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
    bridge.train()
    iter_loader = iter(loader)

    pbar = tqdm(range(args.steps), desc="Training")
    metrics_log = []

    for step in pbar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        texts = batch['text']
        label_indices = batch['label']
        target_strs = [labels[l] for l in label_indices]

        B = len(texts)

        # Source
        src_inputs = src_tok(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[31]
            src_mask = src_inputs.attention_mask

        # Bridge
        latents, aux_loss, _, _ = bridge(src_hidden, src_mask)

        # Diversity loss
        flat_tokens = latents.view(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        div_loss = sim_matrix[mask].mean()

        # Target
        tgt_inputs = tgt_tok(
            target_strs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(device)
        tgt_embeds = tgt_model.get_input_embeddings()(tgt_inputs.input_ids)

        primer_batch = primer_embeds[:B]
        inputs_embeds = torch.cat([primer_batch, latents, tgt_embeds], dim=1)

        # Labels
        ignore_len = primer_batch.shape[1] + latents.shape[1]
        labels_tensor = torch.full((B, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
        labels_tensor[:, ignore_len:] = tgt_inputs.input_ids

        attn_mask = torch.ones(B, inputs_embeds.shape[1], device=device)

        outputs = tgt_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_tensor)
        lm_loss = outputs.loss

        loss = lm_loss + args.diversity_weight * div_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({
            'lm': f"{lm_loss.item():.3f}",
            'div': f"{div_loss.item():.3f}"
        })

        if (step + 1) % args.eval_every == 0:
            eval_results = evaluate(
                bridge, src_model, tgt_model, src_tok, tgt_tok,
                labels, dataset['test'], device, num_samples=100
            )
            metrics_log.append({"step": step + 1, **eval_results})
            bridge.train()

    # Final eval
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_results = evaluate(
        bridge, src_model, tgt_model, src_tok, tgt_tok,
        labels, dataset['test'], device, num_samples=200
    )

    # Save
    torch.save(bridge.state_dict(), os.path.join(args.output_dir, "bridge_banking77.pt"))

    results = {
        "experiment": "banking77",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "num_classes": len(labels),
        "final_results": final_results,
        "training_log": metrics_log
    }

    with open(os.path.join(args.output_dir, "banking77_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
