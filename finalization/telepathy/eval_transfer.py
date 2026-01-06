#!/usr/bin/env python3
"""
Cross-task transfer evaluation.

Addresses reviewer concern: "Can a bridge trained on SST-2 work on AG News?"

Tests if trained bridges have any zero-shot transfer capability.

Usage:
    python eval_transfer.py --source_task sst2 --checkpoint runs/sst2/bridge.pt
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
    configs = {
        "sst2": {
            "hf_name": ("glue", "sst2"),
            "text_field": "sentence",
            "label_field": "label",
            "label_map": {0: "negative", 1: "positive"},
            "num_classes": 2,
            "eval_split": "validation",
        },
        "agnews": {
            "hf_name": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            "num_classes": 4,
            "eval_split": "test",
        },
        "trec": {
            "hf_name": ("trec",),
            "text_field": "text",
            "label_field": "coarse_label",
            "label_map": {0: "ABBR", 1: "ENTY", 2: "DESC", 3: "HUM", 4: "LOC", 5: "NUM"},
            "num_classes": 6,
            "eval_split": "test",
        },
    }
    return configs[dataset_name]


class TelepathyBridge(nn.Module):
    """Bridge module that transforms sender hidden states to receiver soft tokens."""

    def __init__(
        self,
        sender_dim: int = 4096,
        receiver_dim: int = 4096,
        num_soft_tokens: int = 16,
        internal_dim: int = 512,
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens

        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_soft_tokens, internal_dim) * 0.02)

        # Project sender hidden states
        self.sender_proj = nn.Linear(sender_dim, internal_dim)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=internal_dim,
            num_heads=8,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(internal_dim, receiver_dim)

    def forward(self, sender_hidden_states):
        """
        Args:
            sender_hidden_states: [batch, seq_len, sender_dim]
        Returns:
            soft_tokens: [batch, num_soft_tokens, receiver_dim]
        """
        batch_size = sender_hidden_states.shape[0]

        # Project sender states
        sender_proj = self.sender_proj(sender_hidden_states)

        # Expand queries for batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to sender
        attended, _ = self.cross_attn(
            query=queries,
            key=sender_proj,
            value=sender_proj,
        )

        # Project to receiver embedding space
        soft_tokens = self.output_proj(attended)

        return soft_tokens


def load_bridge(checkpoint_path, device):
    """Load a trained bridge checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use defaults
    config = checkpoint.get("config", {})
    bridge = TelepathyBridge(
        sender_dim=config.get("sender_dim", 4096),
        receiver_dim=config.get("receiver_dim", 4096),
        num_soft_tokens=config.get("num_soft_tokens", 16),
        internal_dim=config.get("internal_dim", 512),
    )

    bridge.load_state_dict(checkpoint["bridge_state_dict"])
    bridge.to(device)
    bridge.eval()

    return bridge, config


def evaluate_transfer(
    bridge, sender, receiver, sender_tok, receiver_tok, eval_ds, config, device, max_samples=200
):
    """Evaluate bridge on a target task (zero-shot transfer)."""
    label_map = config["label_map"]
    label_names = list(label_map.values())

    correct = 0
    total = 0
    predictions = []

    for i, item in enumerate(tqdm(eval_ds, total=min(max_samples, len(eval_ds)))):
        if i >= max_samples:
            break

        text = item[config["text_field"]]
        true_label = label_map[item[config["label_field"]]]

        # Encode with sender
        inputs = sender_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            sender_outputs = sender(**inputs, output_hidden_states=True)
            sender_hidden = sender_outputs.hidden_states[-1]

            # Transform through bridge
            soft_tokens = bridge(sender_hidden)

            # Generate with receiver
            outputs = receiver.generate(
                inputs_embeds=soft_tokens,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=receiver_tok.eos_token_id,
            )

        response = receiver_tok.decode(outputs[0], skip_special_tokens=True)
        response = response.strip().lower()

        # Parse prediction
        pred_label = None
        for label in label_names:
            if label.lower() in response:
                pred_label = label
                break

        is_correct = pred_label and pred_label.lower() == true_label.lower()
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "text": text[:100],
            "true_label": true_label,
            "pred_label": pred_label,
            "response": response[:50],
            "correct": is_correct,
        })

        if i < 5:
            print(f"[{i}] True: {true_label}, Pred: {response[:30]}, Correct: {is_correct}")

    accuracy = 100 * correct / total
    random_chance = 100 / config["num_classes"]

    return accuracy, correct, total, random_chance, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_task", choices=["sst2", "agnews", "trec"], required=True,
                       help="Task the bridge was trained on")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to bridge checkpoint")
    parser.add_argument("--target_tasks", nargs="+", default=["sst2", "agnews", "trec"],
                       help="Tasks to evaluate transfer on")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="runs/transfer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Remove source task from targets (we want zero-shot transfer)
    target_tasks = [t for t in args.target_tasks if t != args.source_task]
    if not target_tasks:
        print("No target tasks specified (excluding source). Testing on all tasks.")
        target_tasks = ["sst2", "agnews", "trec"]

    # Load models
    print("Loading Llama (sender)...")
    llama_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama = AutoModelForCausalLM.from_pretrained(
        llama_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    llama_tok = AutoTokenizer.from_pretrained(llama_id)
    llama_tok.pad_token = llama_tok.eos_token

    print("Loading Mistral (receiver)...")
    mistral_id = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral = AutoModelForCausalLM.from_pretrained(
        mistral_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    mistral_tok = AutoTokenizer.from_pretrained(mistral_id)
    mistral_tok.pad_token = mistral_tok.eos_token

    # Load bridge
    print(f"\nLoading bridge trained on {args.source_task}...")
    bridge, bridge_config = load_bridge(args.checkpoint, device)

    all_results = {}

    # First evaluate on source task (sanity check)
    print(f"\n{'='*60}")
    print(f"Evaluating on SOURCE task: {args.source_task} (sanity check)")
    print(f"{'='*60}")

    source_config = get_dataset_config(args.source_task)
    source_ds = load_dataset(*source_config["hf_name"], split=source_config["eval_split"])

    acc, corr, tot, rand, preds = evaluate_transfer(
        bridge, llama, mistral, llama_tok, mistral_tok,
        source_ds, source_config, device, args.max_samples
    )
    print(f"\nSource task ({args.source_task}): {acc:.1f}% (random: {rand:.1f}%)")

    all_results[args.source_task] = {
        "accuracy": acc,
        "correct": corr,
        "total": tot,
        "random_chance": rand,
        "is_source": True,
        "gap_vs_random": acc - rand,
    }

    # Evaluate on target tasks (zero-shot transfer)
    for target_task in target_tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating TRANSFER to: {target_task}")
        print(f"{'='*60}")

        target_config = get_dataset_config(target_task)
        target_ds = load_dataset(*target_config["hf_name"], split=target_config["eval_split"])

        acc, corr, tot, rand, preds = evaluate_transfer(
            bridge, llama, mistral, llama_tok, mistral_tok,
            target_ds, target_config, device, args.max_samples
        )

        print(f"\nTransfer to {target_task}: {acc:.1f}% (random: {rand:.1f}%)")

        all_results[target_task] = {
            "accuracy": acc,
            "correct": corr,
            "total": tot,
            "random_chance": rand,
            "is_source": False,
            "gap_vs_random": acc - rand,
        }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/transfer_from_{args.source_task}.json"

    results = {
        "experiment": f"transfer_from_{args.source_task}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_task": args.source_task,
            "target_tasks": target_tasks,
            "checkpoint": args.checkpoint,
            "max_samples": args.max_samples,
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("TRANSFER SUMMARY")
    print("=" * 60)
    print(f"\nBridge trained on: {args.source_task}")
    print(f"\n{'Task':<12} {'Accuracy':<12} {'Random':<12} {'Gap':<12} {'Transfer?'}")
    print("-" * 60)
    for task, res in all_results.items():
        transfer = "Source" if res["is_source"] else ("Yes!" if res["gap_vs_random"] > 5 else "No")
        print(f"{task:<12} {res['accuracy']:.1f}%{'':<7} {res['random_chance']:.1f}%{'':<7} {res['gap_vs_random']:+.1f}pp{'':<6} {transfer}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    transfer_tasks = [t for t, r in all_results.items() if not r["is_source"] and r["gap_vs_random"] > 5]
    if transfer_tasks:
        print(f"Positive transfer to: {', '.join(transfer_tasks)}")
        print("The bridge learns some task-general representations!")
    else:
        print("No significant transfer observed.")
        print("The bridge is task-specific (as noted in limitations).")

    # Clean up
    del llama, mistral, bridge
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
