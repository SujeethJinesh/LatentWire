"""
Attention Extraction and Visualization Demo

This script demonstrates core techniques for extracting and visualizing
attention patterns from HuggingFace transformers, independent of LatentWire.

Techniques demonstrated:
1. Extracting attention weights using output_attentions=True
2. Creating attention heatmaps with matplotlib/seaborn
3. Analyzing attention to specific token ranges
4. Using BertViz for interactive visualization

Usage:
    # Basic demo with BERT
    python scripts/attention_extraction_demo.py --model bert-base-uncased

    # Demo with GPT-2
    python scripts/attention_extraction_demo.py --model gpt2

    # Save visualizations
    python scripts/attention_extraction_demo.py --model bert-base-uncased --output_dir runs/demo_attention
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Attention extraction demo")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="HuggingFace model ID")
    parser.add_argument("--text", type=str,
                        default="The quick brown fox jumps over the lazy dog.",
                        help="Text to analyze")
    parser.add_argument("--output_dir", type=str, default="runs/attention_demo")
    parser.add_argument("--layers", type=str, default="0,5,11",
                        help="Comma-separated layer indices")
    parser.add_argument("--heads", type=str, default="0,4,8,11",
                        help="Comma-separated head indices")
    parser.add_argument("--use_bertviz", action="store_true",
                        help="Use BertViz for interactive visualization (requires bertviz package)")
    return parser.parse_args()


class AttentionDemo:
    """Demonstrates attention extraction and visualization techniques."""

    def __init__(self, model_name: str, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

        print(f"Model loaded: {self.model.config.num_hidden_layers} layers, "
              f"{self.model.config.num_attention_heads} heads per layer")

    @torch.no_grad()
    def extract_attention(self, text: str) -> Tuple[torch.Tensor, List[str], tuple]:
        """
        Extract attention weights for input text.

        Returns:
            input_ids: Token IDs [seq_len]
            tokens: List of token strings
            attentions: Tuple of attention tensors [1, num_heads, seq_len, seq_len] per layer
        """
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        print(f"\nTokenized text ({len(tokens)} tokens):")
        print(" ".join(tokens))
        print()

        # Forward pass with attention extraction
        outputs = self.model(**inputs, output_attentions=True)

        # outputs.attentions is a tuple of length num_layers
        # Each element is [batch_size, num_heads, seq_len, seq_len]
        attentions = outputs.attentions

        print(f"Extracted attention weights:")
        print(f"  Layers: {len(attentions)}")
        print(f"  Shape per layer: {attentions[0].shape}")
        print()

        return input_ids, tokens, attentions

    def analyze_attention_to_range(
        self,
        attentions: tuple,
        start_idx: int,
        end_idx: int,
        range_name: str = "target",
    ) -> dict:
        """
        Analyze attention paid to a specific token range.

        This is analogous to analyzing attention to soft tokens in LatentWire.

        Args:
            attentions: Tuple of attention tensors
            start_idx: Start of token range (inclusive)
            end_idx: End of token range (exclusive)
            range_name: Name for the token range

        Returns:
            Dictionary with statistics per layer
        """
        stats = {"per_layer": []}

        for layer_idx, attn in enumerate(attentions):
            # attn: [1, num_heads, seq_len, seq_len]
            # We focus on attention FROM the last token TO the range
            last_token_attn = attn[0, :, -1, :]  # [num_heads, seq_len]

            # Sum attention to the specified range
            range_attn = last_token_attn[:, start_idx:end_idx].sum(dim=-1)  # [num_heads]

            stats["per_layer"].append({
                "layer": layer_idx,
                "mean": range_attn.mean().item(),
                "std": range_attn.std().item(),
                "min": range_attn.min().item(),
                "max": range_attn.max().item(),
                "per_head": range_attn.tolist(),
            })

        print(f"Attention to {range_name} tokens [{start_idx}:{end_idx}]:")
        for layer_stats in stats["per_layer"]:
            print(f"  Layer {layer_stats['layer']:2d}: "
                  f"mean={layer_stats['mean']:.4f}, "
                  f"std={layer_stats['std']:.4f}")
        print()

        return stats

    def visualize_attention_heatmap(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        layer_idx: int,
        head_idx: int,
        highlight_ranges: Optional[List[Tuple[int, int, str]]] = None,
    ):
        """
        Create attention heatmap for a specific layer and head.

        Args:
            attention: [seq_len, seq_len] attention matrix
            tokens: Token strings for labels
            layer_idx: Layer index
            head_idx: Head index
            highlight_ranges: Optional list of (start, end, color) tuples to highlight
        """
        attn_np = attention.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            attn_np,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            cbar_kws={"label": "Attention Weight"},
            vmin=0,
            vmax=1.0,
            ax=ax,
        )

        ax.set_title(f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}")
        ax.set_xlabel("Key Position (attending TO)")
        ax.set_ylabel("Query Position (attending FROM)")

        # Highlight specific ranges
        if highlight_ranges:
            for start, end, color in highlight_ranges:
                ax.axvline(x=start, color=color, linestyle='--', linewidth=2, alpha=0.7)
                ax.axvline(x=end, color=color, linestyle='--', linewidth=2, alpha=0.7)
                ax.axhline(y=start, color=color, linestyle='--', linewidth=2, alpha=0.7)
                ax.axhline(y=end, color=color, linestyle='--', linewidth=2, alpha=0.7)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = self.output_dir / f"heatmap_layer{layer_idx}_head{head_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved heatmap: {save_path}")

    def visualize_attention_by_layer(
        self,
        stats: dict,
        title: str = "Attention by Layer",
        save_name: str = "attention_by_layer.png",
    ):
        """Plot attention statistics across layers."""
        layers = [s["layer"] for s in stats["per_layer"]]
        means = [s["mean"] for s in stats["per_layer"]]
        stds = [s["std"] for s in stats["per_layer"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Mean attention by layer
        ax1.plot(layers, means, marker='o', linewidth=2, markersize=8)
        ax1.fill_between(
            layers,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.3,
        )
        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Mean Attention Weight', fontsize=12)
        ax1.set_title(f'{title} - Mean ± Std', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Right: Per-head attention for selected layers
        selected_layers = [0, len(layers) // 2, len(layers) - 1]
        for layer_idx in selected_layers:
            per_head = stats["per_layer"][layer_idx]["per_head"]
            ax2.plot(per_head, marker='o', label=f'Layer {layer_idx}', alpha=0.7)

        ax2.set_xlabel('Head Index', fontsize=12)
        ax2.set_ylabel('Attention Weight', fontsize=12)
        ax2.set_title('Per-Head Attention (Selected Layers)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved layer analysis: {save_path}")

    def use_bertviz_visualization(
        self,
        tokens: List[str],
        attentions: tuple,
    ):
        """
        Use BertViz for interactive visualization.

        NOTE: This requires the bertviz package:
            pip install bertviz
        """
        try:
            from bertviz import head_view, model_view
        except ImportError:
            print("BertViz not installed. Install with: pip install bertviz")
            return

        print("Generating BertViz visualizations...")

        # Convert attentions to required format
        # BertViz expects list of tensors or tensor directly
        attention_tensors = [attn.cpu() for attn in attentions]

        # Model view (bird's eye view of all layers and heads)
        print("Creating model view...")
        model_view(attention_tensors, tokens, html_action='save')
        print("Model view saved to: model_view.html")

        # Head view (detailed view of specific heads)
        print("Creating head view...")
        head_view(attention_tensors, tokens, html_action='save')
        print("Head view saved to: head_view.html")


def main():
    args = parse_args()

    print("=" * 80)
    print("Attention Extraction and Visualization Demo")
    print("=" * 80)
    print()

    # Initialize demo
    demo = AttentionDemo(args.model, args.output_dir)

    # Extract attention
    input_ids, tokens, attentions = demo.extract_attention(args.text)

    # Analyze attention to first few tokens (simulating "soft tokens")
    # For example, analyze attention to first 3 tokens
    first_k = min(3, len(tokens) - 1)
    stats = demo.analyze_attention_to_range(
        attentions,
        start_idx=0,
        end_idx=first_k,
        range_name=f"first {first_k}",
    )

    # Visualize by layer
    demo.visualize_attention_by_layer(
        stats,
        title=f"Attention to First {first_k} Tokens",
        save_name="attention_to_first_tokens.png",
    )

    # Parse layer and head indices
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    head_indices = [int(x.strip()) for x in args.heads.split(",")]

    # Create heatmaps for selected layers and heads
    print("\nCreating attention heatmaps...")
    for layer_idx in layer_indices:
        if layer_idx >= len(attentions):
            continue

        attn = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]

        for head_idx in head_indices:
            if head_idx >= attn.size(0):
                continue

            demo.visualize_attention_heatmap(
                attn[head_idx],
                tokens,
                layer_idx,
                head_idx,
                highlight_ranges=[(0, first_k, 'red')],  # Highlight first k tokens
            )

    # BertViz visualization (optional)
    if args.use_bertviz:
        demo.use_bertviz_visualization(tokens, attentions)

    print()
    print("=" * 80)
    print("Demo complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
