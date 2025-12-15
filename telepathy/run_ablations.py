#!/usr/bin/env python3
"""
Comprehensive Ablation Studies for Telepathy Bridge.

Ablations:
1. Internal dimension: 256, 512, 1024
2. Number of cross-attention heads: 4, 8, 16
3. Source layer: 16, 24, 28, 31
4. Bridge depth: 1, 2, 4
5. Diversity weight: 0.0, 0.05, 0.1, 0.2

Usage:
    python run_ablations.py --ablation internal_dim --dataset sst2
    python run_ablations.py --ablation all --dataset sst2
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader


class LatentBridgeAblation(nn.Module):
    """
    Ablation-friendly bridge with configurable architecture.
    """
    def __init__(
        self,
        src_dim: int = 4096,
        tgt_dim: int = 4096,
        internal_dim: int = 512,
        num_soft_tokens: int = 16,
        num_heads: int = 8,
        depth: int = 2,
        target_rms: float = 0.03,
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        self.internal_dim = internal_dim
        self.target_rms = target_rms

        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_soft_tokens, internal_dim) * 0.02)

        # Project sender hidden states
        self.sender_proj = nn.Linear(src_dim, internal_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.MultiheadAttention(
                embed_dim=internal_dim,
                num_heads=num_heads,
                batch_first=True,
            ))

        # Output projection
        self.output_proj = nn.Linear(internal_dim, tgt_dim)

        # Initialize
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, sender_hidden_states, sender_attention_mask=None):
        batch_size = sender_hidden_states.shape[0]

        # Project sender states
        sender_proj = self.sender_proj(sender_hidden_states)

        # Expand queries for batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply cross-attention layers
        for layer in self.layers:
            attended, _ = layer(
                query=queries,
                key=sender_proj,
                value=sender_proj,
            )
            queries = queries + attended  # Residual connection

        # Project to receiver embedding space
        soft_tokens = self.output_proj(queries)

        # Normalize to target RMS
        rms = torch.sqrt(torch.mean(soft_tokens ** 2, dim=-1, keepdim=True) + 1e-8)
        soft_tokens = soft_tokens * (self.target_rms / rms)

        # Simple aux loss (diversity)
        flat = soft_tokens.reshape(batch_size, -1).float()
        flat_norm = F.normalize(flat, dim=1)
        sim = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=soft_tokens.device)
        aux_loss = sim[mask].mean() if batch_size > 1 else torch.tensor(0.0, device=soft_tokens.device)

        return soft_tokens, aux_loss


def get_dataset_config(dataset_name):
    """Get dataset configuration."""
    configs = {
        "sst2": {
            "hf_name": ("glue", "sst2"),
            "text_field": "sentence",
            "label_field": "label",
            "labels": ["negative", "positive"],
            "train_split": "train",
            "eval_split": "validation",
        },
        "agnews": {
            "hf_name": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "labels": ["World", "Sports", "Business", "Sci/Tech"],
            "train_split": "train",
            "eval_split": "test",
        },
    }
    return configs[dataset_name]


def train_and_eval_bridge(
    bridge_config,
    train_loader,
    test_data,
    llama,
    mistral,
    llama_tok,
    mistral_tok,
    labels,
    dataset_config,
    device,
    steps=1000,
    lr=1e-4,
    source_layer=31,
    diversity_weight=0.1,
):
    """Train and evaluate a bridge with given config."""
    # Create bridge
    bridge = LatentBridgeAblation(**bridge_config)
    bridge = bridge.to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr)

    # Primer
    primer = "Label: "
    primer_tokens = mistral_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        primer_embeds_single = mistral.get_input_embeddings()(primer_tokens.input_ids)

    # Training loop
    bridge.train()
    iter_loader = iter(train_loader)
    train_losses = []

    for step in range(steps):
        try:
            texts, label_indices = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            texts, label_indices = next(iter_loader)

        texts = list(texts)
        target_strs = [labels[l] for l in label_indices]
        B = len(texts)

        # Encode with Llama
        src_inputs = llama_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            src_out = llama(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]

        # Bridge
        latents, aux_loss = bridge(src_hidden, src_inputs.attention_mask)

        # Target
        tgt_inputs = mistral_tok(
            target_strs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(device)
        tgt_embeds = mistral.get_input_embeddings()(tgt_inputs.input_ids)

        primer_batch = primer_embeds_single.repeat(B, 1, 1)
        inputs_embeds = torch.cat([primer_batch, latents, tgt_embeds], dim=1)

        # Labels
        ignore_len = primer_batch.shape[1] + latents.shape[1]
        labels_tensor = torch.full((B, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
        labels_tensor[:, ignore_len:] = tgt_inputs.input_ids

        attn_mask = torch.ones(B, inputs_embeds.shape[1], device=device)

        outputs = mistral(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_tensor)
        lm_loss = outputs.loss

        loss = lm_loss + diversity_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        train_losses.append(loss.item())

    # Evaluation
    bridge.eval()
    correct = 0
    total = 0

    for i, item in enumerate(test_data):
        if i >= 200:
            break

        text = item[dataset_config["text_field"]]
        label_idx = item[dataset_config["label_field"]]
        true_label = labels[label_idx]

        src_inputs = llama_tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            src_out = llama(**src_inputs, output_hidden_states=True)
            src_hidden = src_out.hidden_states[source_layer]
            latents, _ = bridge(src_hidden, src_inputs.attention_mask)

            combined = torch.cat([primer_embeds_single, latents], dim=1)
            attn_mask = torch.ones(1, combined.shape[1], device=device)

            out_ids = mistral.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=10,
                pad_token_id=mistral_tok.eos_token_id,
                do_sample=False
            )
            output = mistral_tok.decode(out_ids[0], skip_special_tokens=True).lower()

        is_correct = true_label.lower() in output
        if is_correct:
            correct += 1
        total += 1

    accuracy = 100 * correct / total
    bridge_params = sum(p.numel() for p in bridge.parameters())

    del bridge, optimizer
    torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "bridge_params": bridge_params,
        "final_loss": sum(train_losses[-100:]) / 100 if len(train_losses) >= 100 else sum(train_losses) / len(train_losses),
    }


def run_ablation(ablation_type, dataset_name, device, output_dir, steps=1000):
    """Run a specific ablation study."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_type} on {dataset_name}")
    print(f"{'='*60}")

    config = get_dataset_config(dataset_name)
    labels = config["labels"]

    # Load dataset
    train_ds = load_dataset(*config["hf_name"], split=config["train_split"])
    test_ds = load_dataset(*config["hf_name"], split=config["eval_split"])

    # Create DataLoader
    class SimpleDataset:
        def __init__(self, ds, config):
            self.ds = ds
            self.config = config

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            return item[self.config["text_field"]], item[self.config["label_field"]]

    train_dataset = SimpleDataset(train_ds, config)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Load models
    print("Loading Llama...")
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16
    ).to(device)
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama_tok.pad_token = llama_tok.eos_token

    print("Loading Mistral...")
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16
    ).to(device)
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    for p in llama.parameters():
        p.requires_grad = False
    for p in mistral.parameters():
        p.requires_grad = False

    # Define ablation configurations
    base_config = {
        "src_dim": 4096,
        "tgt_dim": 4096,
        "internal_dim": 512,
        "num_soft_tokens": 16,
        "num_heads": 8,
        "depth": 2,
        "target_rms": 0.03,
    }

    if ablation_type == "internal_dim":
        variations = [
            {"name": "dim_256", "internal_dim": 256},
            {"name": "dim_512", "internal_dim": 512},
            {"name": "dim_1024", "internal_dim": 1024},
        ]
    elif ablation_type == "num_heads":
        variations = [
            {"name": "heads_4", "num_heads": 4},
            {"name": "heads_8", "num_heads": 8},
            {"name": "heads_16", "num_heads": 16},
        ]
    elif ablation_type == "source_layer":
        variations = [
            {"name": "layer_16", "source_layer": 16},
            {"name": "layer_24", "source_layer": 24},
            {"name": "layer_28", "source_layer": 28},
            {"name": "layer_31", "source_layer": 31},
        ]
    elif ablation_type == "depth":
        variations = [
            {"name": "depth_1", "depth": 1},
            {"name": "depth_2", "depth": 2},
            {"name": "depth_4", "depth": 4},
        ]
    elif ablation_type == "diversity_weight":
        variations = [
            {"name": "div_0.0", "diversity_weight": 0.0},
            {"name": "div_0.05", "diversity_weight": 0.05},
            {"name": "div_0.1", "diversity_weight": 0.1},
            {"name": "div_0.2", "diversity_weight": 0.2},
        ]
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    results = []

    for var in variations:
        name = var.pop("name")
        print(f"\n--- Running: {name} ---")

        # Update config
        bridge_config = base_config.copy()
        source_layer = var.pop("source_layer", 31)
        diversity_weight = var.pop("diversity_weight", 0.1)
        bridge_config.update(var)

        result = train_and_eval_bridge(
            bridge_config=bridge_config,
            train_loader=train_loader,
            test_data=test_ds,
            llama=llama,
            mistral=mistral,
            llama_tok=llama_tok,
            mistral_tok=mistral_tok,
            labels=labels,
            dataset_config=config,
            device=device,
            steps=steps,
            lr=1e-4,
            source_layer=source_layer,
            diversity_weight=diversity_weight,
        )

        result["name"] = name
        result["config"] = {**bridge_config, "source_layer": source_layer, "diversity_weight": diversity_weight}
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.1f}%, Params: {result['bridge_params']:,}")

    # Clean up
    del llama, mistral
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True,
                       choices=["internal_dim", "num_heads", "source_layer", "depth", "diversity_weight", "all"])
    parser.add_argument("--dataset", choices=["sst2", "agnews"], default="sst2")
    parser.add_argument("--output_dir", default="runs/ablations")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {
        "experiment": "ablation_study",
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "steps": args.steps,
        "ablations": {},
    }

    ablation_types = ["internal_dim", "num_heads", "source_layer", "depth", "diversity_weight"] if args.ablation == "all" else [args.ablation]

    for abl in ablation_types:
        results = run_ablation(abl, args.dataset, device, args.output_dir, args.steps)
        all_results["ablations"][abl] = results

        # Print summary
        print(f"\n{'='*60}")
        print(f"ABLATION SUMMARY: {abl}")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['name']:<20} {r['accuracy']:.1f}%  ({r['bridge_params']:,} params)")

    # Save
    output_file = f"{args.output_dir}/ablations_{args.dataset}_{args.ablation}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
