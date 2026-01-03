#!/usr/bin/env python3
"""
Generate t-SNE visualization of AG News latent representations.

This script:
1. Loads a trained Bridge model
2. Encodes AG News samples through the Bridge
3. Generates t-SNE plot showing category separation in latent space
4. Saves visualization as PDF for paper inclusion

Usage:
    python telepathy/generate_tsne_visualization.py \
        --checkpoint runs/enhanced_arxiv_*/phase3_multiseed/agnews_seed42/bridge.pt \
        --output figures/agnews_tsne.pdf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# BRIDGE ARCHITECTURE (copied from run_unified_comparison.py)
# =============================================================================

class PerceiverResampler(nn.Module):
    def __init__(self, src_dim, tgt_dim, num_latents=8, heads=8, depth=2):
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


class UnifiedBridge(nn.Module):
    def __init__(self, sender_dim, receiver_dim, num_tokens=8, depth=2, target_rms=0.03):
        super().__init__()
        self.perceiver = PerceiverResampler(sender_dim, receiver_dim, num_tokens, depth=depth)
        self.output_scale = nn.Parameter(torch.tensor(target_rms))
        self.num_tokens = num_tokens

    def forward(self, sender_hidden, attention_mask=None):
        soft_tokens = self.perceiver(sender_hidden, attention_mask)
        rms = torch.sqrt((soft_tokens ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        return (soft_tokens / rms) * self.output_scale


# =============================================================================
# VISUALIZATION
# =============================================================================

def extract_latents(bridge, sender_model, tokenizer, texts, labels, device, source_layer=16):
    """Extract latent representations for a list of texts."""
    latents = []
    label_list = []

    bridge.eval()
    sender_model.eval()

    with torch.no_grad():
        for text, label in zip(texts, labels):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get sender hidden states
            outputs = sender_model(**inputs, output_hidden_states=True)
            sender_hidden = outputs.hidden_states[source_layer]

            # Get latent representation
            soft_tokens = bridge(sender_hidden, inputs["attention_mask"])

            # Flatten to single vector (8 tokens × 4096 dim → 32768 dim)
            # Convert to float32 since numpy doesn't support bfloat16
            latent = soft_tokens.flatten().cpu().float().numpy()
            latents.append(latent)
            label_list.append(label)

    return np.array(latents), np.array(label_list)


def create_tsne_plot(latents, labels, label_names, output_path):
    """Create and save t-SNE visualization."""
    print(f"Running t-SNE on {len(latents)} samples...")

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(latents)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Color palette for 4 AG News categories
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # World, Sports, Business, Science

    for i, (label_id, label_name) in enumerate(label_names.items()):
        mask = labels == label_id
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[i],
            label=label_name,
            alpha=0.7,
            s=50
        )

    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title("AG News Latent Space: Bridge Separates Categories", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    # Also save as PNG for quick preview
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved preview to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE visualization of Bridge latents")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to bridge checkpoint")
    parser.add_argument("--output", type=str, default="figures/agnews_tsne.pdf", help="Output path")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Samples per category")
    parser.add_argument("--sender_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--source_layer", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sender model
    print(f"Loading sender model: {args.sender_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.sender_id)
    tokenizer.pad_token = tokenizer.eos_token
    sender_model = AutoModelForCausalLM.from_pretrained(
        args.sender_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    sender_dim = sender_model.config.hidden_size

    # Load bridge
    print(f"Loading bridge from: {args.checkpoint}")
    bridge = UnifiedBridge(sender_dim, sender_dim, num_tokens=8)
    bridge.load_state_dict(torch.load(args.checkpoint, map_location=device))
    bridge = bridge.to(device=device, dtype=torch.bfloat16)

    # Load AG News data
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news", split="test")

    # AG News labels
    label_names = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Tech"
    }

    # Sample balanced data
    texts = []
    labels = []
    for label_id in range(4):
        class_samples = [x for x in dataset if x["label"] == label_id][:args.samples_per_class]
        for sample in class_samples:
            texts.append(sample["text"])
            labels.append(label_id)

    print(f"Sampled {len(texts)} total samples ({args.samples_per_class} per class)")

    # Extract latents
    print("Extracting latent representations...")
    latents, label_array = extract_latents(
        bridge, sender_model, tokenizer, texts, labels, device, args.source_layer
    )

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Generate visualization
    create_tsne_plot(latents, label_array, label_names, args.output)

    print("\nDone! If categories form distinct clusters, this proves semantic disentanglement.")


if __name__ == "__main__":
    main()
