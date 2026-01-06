#!/usr/bin/env python3
"""
Same-Model Bridge Experiment.

Ablation study: What happens when sender == receiver?
- Llama → Bridge → Llama
- Mistral → Bridge → Mistral

This tests whether the bridge benefit comes from:
1. Cross-model communication (different representations)
2. The compression/bottleneck itself (forces abstraction)

If same-model bridge matches cross-model, the benefit is from compression.
If cross-model >> same-model, the benefit is from heterogeneous representations.

Usage:
    python train_same_model_bridge.py --model llama --dataset sst2
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from latent_bridge_v15 import LatentBridgeV15


class Args:
    """Args object for LatentBridgeV15 interface."""
    def __init__(self, soft_tokens=16, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
    configs = {
        "sst2": {
            "hf_name": ("glue", "sst2"),
            "text_field": "sentence",
            "label_field": "label",
            "label_map": {0: "negative", 1: "positive"},
            "train_split": "train",
            "eval_split": "validation",
        },
        "agnews": {
            "hf_name": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            "train_split": "train",
            "eval_split": "test",
        },
        "trec": {
            "hf_name": ("trec",),
            "text_field": "text",
            "label_field": "coarse_label",
            "label_map": {0: "ABBR", 1: "ENTY", 2: "DESC", 3: "HUM", 4: "LOC", 5: "NUM"},
            "train_split": "train",
            "eval_split": "test",
        },
    }
    return configs[dataset_name]


def get_model_config(model_name):
    """Get model-specific configuration."""
    configs = {
        "llama": {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "hidden_dim": 4096,
            "num_layers": 32,
        },
        "mistral": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "hidden_dim": 4096,
            "num_layers": 32,
        },
    }
    return configs[model_name]


def evaluate(bridge, model, tokenizer, test_data, labels, device, num_samples=200, source_layer=31):
    """Evaluate same-model bridge."""
    bridge.eval()

    primer = "Label: "
    primer_tokens = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        primer_embeds = model.get_input_embeddings()(primer_tokens.input_ids)

    correct = 0
    total = 0

    for i, item in enumerate(test_data):
        if i >= num_samples:
            break

        text = item["text"]
        label_idx = item["label"]
        label = labels[label_idx]

        # Encode with model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[source_layer]
            mask = inputs.attention_mask

            # Transform through bridge
            latents, _, _, _ = bridge(hidden, mask)

            # Combine with primer
            combined = torch.cat([primer_embeds, latents], dim=1)
            attn_mask = torch.ones(1, combined.shape[1], device=device)

            # Generate
            out_ids = model.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=15,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            output = tokenizer.decode(out_ids[0], skip_special_tokens=True).lower()

        # Check match
        is_correct = label.lower() in output
        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            print(f"[{i}] GT: {label} | Output: {output[:50]}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "mistral"], required=True,
                       help="Model to use for both sender and receiver")
    parser.add_argument("--dataset", choices=["sst2", "agnews", "trec"], required=True)
    parser.add_argument("--output_dir", default="runs/same_model_bridge")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--soft_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=400)
    parser.add_argument("--source_layer", type=int, default=31)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_config = get_dataset_config(args.dataset)
    model_config = get_model_config(args.model)

    print("=" * 60)
    print(f"SAME-MODEL BRIDGE: {args.model.upper()} → Bridge → {args.model.upper()}")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Soft tokens: {args.soft_tokens}")
    print(f"Model: {model_config['model_id']}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading {args.dataset}...")
    if args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")
        train_data = [{"text": x["sentence"], "label": x["label"]}
                     for x in dataset["train"]]
        test_data = [{"text": x["sentence"], "label": x["label"]}
                    for x in dataset["validation"]]
        labels = ["negative", "positive"]
    elif args.dataset == "agnews":
        dataset = load_dataset("ag_news")
        train_data = dataset["train"]
        test_data = dataset["test"]
        labels = ["World", "Sports", "Business", "Sci/Tech"]
    else:  # trec
        dataset = load_dataset("trec")
        train_data = dataset["train"]
        test_data = dataset["test"]
        labels = dataset["train"].features["coarse_label"].names

    print(f"Labels: {labels}")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Load model (same for sender and receiver)
    print(f"\nLoading {args.model.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # Create bridge (same hidden dim for sender and receiver)
    bridge_args = Args(soft_tokens=args.soft_tokens, heads=8, depth=2, use_fsq=False)
    bridge = LatentBridgeV15(
        bridge_args,
        src_dim=model_config["hidden_dim"],
        tgt_dim=model_config["hidden_dim"],
        target_rms=0.03
    )
    bridge = bridge.to(device).to(torch.bfloat16)

    bridge_params = sum(p.numel() for p in bridge.parameters())
    print(f"Bridge parameters: {bridge_params:,}")

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr)

    # Primer
    primer = "Label: "
    primer_tokens = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        primer_embeds_single = model.get_input_embeddings()(primer_tokens.input_ids)
        primer_embeds = primer_embeds_single.repeat(args.batch_size, 1, 1)

    # Training
    from torch.utils.data import DataLoader

    # Create simple dataset
    class SimpleDataset:
        def __init__(self, data, labels_list):
            self.data = data
            self.labels_list = labels_list

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            if isinstance(item, dict):
                return item["text"], item["label"]
            else:
                if args.dataset == "sst2":
                    return item["text"], item["label"]
                elif args.dataset == "agnews":
                    return item["text"], item["label"]
                else:
                    return item["text"], item["coarse_label"]

    train_dataset = SimpleDataset(train_data, labels)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    bridge.train()
    iter_loader = iter(loader)

    pbar = tqdm(range(args.steps), desc="Training")
    metrics_log = []

    for step in pbar:
        try:
            texts, label_indices = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            texts, label_indices = next(iter_loader)

        texts = list(texts)
        target_strs = [labels[l] for l in label_indices]
        B = len(texts)

        # Encode
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[args.source_layer]
            mask = inputs.attention_mask

        # Bridge
        latents, aux_loss, _, _ = bridge(hidden, mask)

        # Diversity loss
        flat_tokens = latents.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        div_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        div_loss = sim_matrix[div_mask].mean()

        # Target
        tgt_inputs = tokenizer(
            target_strs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(device)
        tgt_embeds = model.get_input_embeddings()(tgt_inputs.input_ids)

        primer_batch = primer_embeds[:B]
        inputs_embeds = torch.cat([primer_batch, latents, tgt_embeds], dim=1)

        # Labels
        ignore_len = primer_batch.shape[1] + latents.shape[1]
        labels_tensor = torch.full((B, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
        labels_tensor[:, ignore_len:] = tgt_inputs.input_ids

        attn_mask = torch.ones(B, inputs_embeds.shape[1], device=device)

        model_outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_tensor)
        lm_loss = model_outputs.loss

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
                bridge, model, tokenizer,
                test_data, labels, device, num_samples=100,
                source_layer=args.source_layer
            )
            metrics_log.append({"step": step + 1, **eval_results})
            print(f"\nStep {step+1}: {eval_results['accuracy']:.1f}%")
            bridge.train()

    # Final eval
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_results = evaluate(
        bridge, model, tokenizer,
        test_data, labels, device, num_samples=200,
        source_layer=args.source_layer
    )

    # Save results
    output_file = f"{args.output_dir}/same_model_{args.model}_{args.dataset}.json"
    results = {
        "experiment": f"same_model_bridge_{args.model}_{args.dataset}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "model_id": model_config["model_id"],
            "dataset": args.dataset,
            "soft_tokens": args.soft_tokens,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "source_layer": args.source_layer,
            "diversity_weight": args.diversity_weight,
            "seed": args.seed,
            "bridge_params": bridge_params,
        },
        "final_results": final_results,
        "training_log": metrics_log,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Comparison with cross-model
    cross_model_results = {
        "sst2": 96.7,
        "agnews": 90.7,
        "trec": 95.3,
    }

    print("\n" + "=" * 60)
    print("SAME-MODEL vs CROSS-MODEL COMPARISON")
    print("=" * 60)
    print(f"Same-Model ({args.model}→{args.model}): {final_results['accuracy']:.1f}%")
    print(f"Cross-Model (Llama→Mistral):     {cross_model_results.get(args.dataset, 'N/A')}%")
    gap = cross_model_results.get(args.dataset, 0) - final_results['accuracy']
    print(f"Difference: {'+' if gap > 0 else ''}{gap:.1f}pp")

    if gap > 5:
        print("\n→ Cross-model significantly better: heterogeneous representations help!")
    elif gap < -5:
        print("\n→ Same-model better: compression/bottleneck is the key!")
    else:
        print("\n→ Similar performance: both compression and heterogeneity matter.")


if __name__ == "__main__":
    main()
