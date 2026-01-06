"""
Attention Pattern Analysis for LatentWire

This script extracts and visualizes attention patterns from transformer models,
with special focus on measuring attention to soft tokens (latent embeddings).

Key capabilities:
1. Extract attention weights from HuggingFace models
2. Visualize attention heatmaps for specific layers/heads
3. Measure attention distribution to soft tokens vs. text tokens
4. Compare attention patterns between text baseline and latent conditioning

Usage:
    python scripts/analyze_attention.py \
        --ckpt runs/8B_clean_answer_ftce/epoch23 \
        --samples 20 \
        --dataset squad \
        --output_dir runs/attention_analysis

Based on:
- HuggingFace output_attentions API
- BertViz visualization toolkit
- Soft prompt attention analysis literature
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from latentwire.models import (
    InterlinguaEncoder,
    Adapter,
    LMWrapper,
    LMConfig,
)
from latentwire.core_utils import (
    build_chat_prompts,
    make_anchor_text,
    bos_policy,
)
from latentwire.data import load_examples


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze attention patterns in LatentWire")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--samples", type=int, default=20, help="Number of examples to analyze")
    parser.add_argument("--dataset", type=str, default="squad", choices=["squad", "hotpotqa"])
    parser.add_argument("--output_dir", type=str, default="runs/attention_analysis")
    parser.add_argument("--model_key", type=str, default="llama", choices=["llama", "qwen"])
    parser.add_argument("--layers", type=str, default="0,15,31", help="Comma-separated layer indices to analyze")
    parser.add_argument("--heads", type=str, default="0,8,16,24,31", help="Comma-separated head indices to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class AttentionExtractor:
    """Extract attention weights from HuggingFace models."""

    def __init__(self, wrapper: LMWrapper):
        self.wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer

    @torch.no_grad()
    def extract_first_token_attention(
        self,
        prefix_embeds: torch.Tensor,
        anchor_token_text: Optional[str] = None,
        append_bos_after_prefix: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract attention weights when generating the first token.

        Returns:
            Dictionary containing:
            - attentions: Tuple of tensors [B, num_heads, seq_len, seq_len] for each layer
            - logits: [B, vocab_size] logits for first token
            - prefix_len: Number of soft tokens
            - anchor_len: Number of anchor tokens
            - bos_len: 1 if BOS appended, else 0
        """
        # Prepare inputs
        anchor_ids_seq = self.wrapper._encode_anchor_text(anchor_token_text) if anchor_token_text else None
        inputs_embeds, attn_mask, prepared_past = self.wrapper._compose_inputs_from_prefix(
            prefix_embeds,
            None,
            anchor_ids=anchor_ids_seq,
            append_bos_after_prefix=append_bos_after_prefix,
            deep_prefix=None,
        )

        # Forward pass with attention extraction
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            output_attentions=True,  # KEY: Enable attention extraction
            return_dict=True,
        )

        # Calculate sequence composition
        B, total_seq_len, _ = inputs_embeds.shape
        prefix_len = prefix_embeds.size(1)
        anchor_len = len(anchor_ids_seq) if anchor_ids_seq else 0
        bos_len = 1 if append_bos_after_prefix else 0

        return {
            "attentions": out.attentions,  # Tuple of [B, num_heads, seq_len, seq_len]
            "logits": out.logits[:, -1, :],  # [B, vocab_size]
            "prefix_len": prefix_len,
            "anchor_len": anchor_len,
            "bos_len": bos_len,
            "total_seq_len": total_seq_len,
        }

    @torch.no_grad()
    def extract_generation_attention(
        self,
        prefix_embeds: torch.Tensor,
        anchor_token_text: Optional[str] = None,
        append_bos_after_prefix: bool = True,
        max_new_tokens: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Extract attention weights during autoregressive generation.

        Returns list of dictionaries (one per generation step), each containing:
            - attentions: Attention weights for that step
            - token_id: Generated token ID
            - step: Generation step number
        """
        # Initial forward pass
        anchor_ids_seq = self.wrapper._encode_anchor_text(anchor_token_text) if anchor_token_text else None
        inputs_embeds, attn_mask, prepared_past = self.wrapper._compose_inputs_from_prefix(
            prefix_embeds,
            None,
            anchor_ids=anchor_ids_seq,
            append_bos_after_prefix=append_bos_after_prefix,
            deep_prefix=None,
        )

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        prefix_len = prefix_embeds.size(1)
        anchor_len = len(anchor_ids_seq) if anchor_ids_seq else 0
        bos_len = 1 if append_bos_after_prefix else 0

        # Store first step
        results = [{
            "step": 0,
            "attentions": out.attentions,
            "token_id": None,
            "prefix_len": prefix_len,
            "anchor_len": anchor_len,
            "bos_len": bos_len,
        }]

        past = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

        B = prefix_embeds.size(0)
        device = next(self.model.parameters()).device
        pad_id = self.tokenizer.pad_token_id or 0
        stop_ids = set(self.wrapper._stop_token_ids)

        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Generation loop
        for step in range(max_new_tokens):
            step_logits = next_token_logits.clone()
            step_logits[finished] = -1e9
            next_tokens = torch.argmax(step_logits, dim=-1)

            for b in range(B):
                if finished[b]:
                    continue
                nid = int(next_tokens[b].item())
                if nid in stop_ids:
                    finished[b] = True
                else:
                    generated[b].append(nid)

            if torch.all(finished).item():
                break

            feed_tokens = next_tokens.clone()
            feed_tokens[finished] = pad_id
            attn_mask_step = torch.ones((B, 1), dtype=torch.long, device=device)

            out = self.model(
                input_ids=feed_tokens.unsqueeze(-1),
                attention_mask=attn_mask_step,
                use_cache=True,
                past_key_values=past,
                output_attentions=True,  # Extract attention at each step
                return_dict=True,
            )

            results.append({
                "step": step + 1,
                "attentions": out.attentions,
                "token_id": next_tokens.cpu().tolist(),
                "prefix_len": prefix_len,
                "anchor_len": anchor_len,
                "bos_len": bos_len,
            })

            past = out.past_key_values
            next_token_logits = out.logits[:, -1, :]

        return results


class AttentionAnalyzer:
    """Analyze and visualize attention patterns."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("viridis")

    def compute_soft_token_attention_stats(
        self,
        attentions: Tuple[torch.Tensor, ...],
        prefix_len: int,
        anchor_len: int = 0,
        bos_len: int = 0,
    ) -> Dict[str, Any]:
        """
        Compute statistics about attention paid to soft tokens.

        Args:
            attentions: Tuple of attention tensors [B, num_heads, seq_len, seq_len]
            prefix_len: Number of soft tokens (latent embeddings)
            anchor_len: Number of anchor text tokens (e.g., "Answer: ")
            bos_len: 1 if BOS token appended, else 0

        Returns:
            Dictionary with attention statistics per layer and head
        """
        num_layers = len(attentions)
        stats = {
            "num_layers": num_layers,
            "per_layer": [],
        }

        # Soft tokens are at positions [0:prefix_len]
        # Then optionally anchor tokens [prefix_len:prefix_len+anchor_len]
        # Then optionally BOS [prefix_len+anchor_len:prefix_len+anchor_len+bos_len]

        for layer_idx, attn in enumerate(attentions):
            # attn: [B, num_heads, seq_len, seq_len]
            B, num_heads, seq_len, _ = attn.shape

            # Focus on last position (where we're generating first token)
            # attn[:, :, -1, :] gives attention FROM last position TO all previous positions
            last_pos_attn = attn[:, :, -1, :]  # [B, num_heads, seq_len]

            # Attention to soft tokens
            soft_token_attn = last_pos_attn[:, :, :prefix_len]  # [B, num_heads, prefix_len]
            soft_token_attn_sum = soft_token_attn.sum(dim=-1)  # [B, num_heads]

            # Attention to anchor tokens (if any)
            if anchor_len > 0:
                anchor_attn = last_pos_attn[:, :, prefix_len:prefix_len+anchor_len]
                anchor_attn_sum = anchor_attn.sum(dim=-1)
            else:
                anchor_attn_sum = torch.zeros_like(soft_token_attn_sum)

            # Attention to BOS token (if any)
            if bos_len > 0:
                bos_attn = last_pos_attn[:, :, prefix_len+anchor_len:prefix_len+anchor_len+bos_len]
                bos_attn_sum = bos_attn.sum(dim=-1)
            else:
                bos_attn_sum = torch.zeros_like(soft_token_attn_sum)

            layer_stats = {
                "layer_idx": layer_idx,
                "num_heads": num_heads,
                "soft_token_attn_mean": soft_token_attn_sum.mean().item(),
                "soft_token_attn_std": soft_token_attn_sum.std().item(),
                "soft_token_attn_min": soft_token_attn_sum.min().item(),
                "soft_token_attn_max": soft_token_attn_sum.max().item(),
                "anchor_attn_mean": anchor_attn_sum.mean().item() if anchor_len > 0 else 0.0,
                "bos_attn_mean": bos_attn_sum.mean().item() if bos_len > 0 else 0.0,
                "per_head_soft_attn": soft_token_attn_sum.mean(dim=0).cpu().tolist(),  # Average over batch
            }

            stats["per_layer"].append(layer_stats)

        return stats

    def visualize_attention_heatmap(
        self,
        attention: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        prefix_len: int,
        anchor_len: int = 0,
        bos_len: int = 0,
        example_id: str = "0",
        save_name: Optional[str] = None,
    ):
        """
        Create attention heatmap for a specific layer and head.

        Args:
            attention: [seq_len, seq_len] attention matrix
            layer_idx: Layer index
            head_idx: Head index
            prefix_len: Number of soft tokens
            anchor_len: Number of anchor tokens
            bos_len: BOS token count
            example_id: Example identifier
            save_name: Optional custom save name
        """
        attn_np = attention.cpu().numpy()
        seq_len = attn_np.shape[0]

        # Create labels for sequence positions
        labels = []
        for i in range(seq_len):
            if i < prefix_len:
                labels.append(f"S{i}")  # Soft token
            elif i < prefix_len + anchor_len:
                labels.append(f"A{i-prefix_len}")  # Anchor token
            elif i < prefix_len + anchor_len + bos_len:
                labels.append("BOS")
            else:
                labels.append(f"T{i-prefix_len-anchor_len-bos_len}")  # Text token

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            attn_np,
            xticklabels=labels,
            yticklabels=labels,
            cmap="viridis",
            cbar_kws={"label": "Attention Weight"},
            vmin=0,
            vmax=attn_np.max(),
            ax=ax,
        )

        ax.set_title(f"Attention Heatmap - Layer {layer_idx}, Head {head_idx}\n(S=Soft, A=Anchor, T=Text)")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        # Add vertical lines to separate regions
        ax.axvline(x=prefix_len, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(y=prefix_len, color='red', linestyle='--', linewidth=2, alpha=0.7)

        if anchor_len > 0:
            ax.axvline(x=prefix_len+anchor_len, color='orange', linestyle='--', linewidth=2, alpha=0.7)
            ax.axhline(y=prefix_len+anchor_len, color='orange', linestyle='--', linewidth=2, alpha=0.7)

        plt.tight_layout()

        if save_name is None:
            save_name = f"attn_ex{example_id}_layer{layer_idx}_head{head_idx}.png"

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved heatmap: {save_path}")

    def visualize_attention_to_soft_tokens(
        self,
        stats: Dict[str, Any],
        save_name: str = "soft_token_attention_by_layer.png",
    ):
        """
        Plot attention to soft tokens across layers and heads.

        Args:
            stats: Statistics from compute_soft_token_attention_stats
            save_name: Output filename
        """
        num_layers = stats["num_layers"]
        layers = list(range(num_layers))

        # Extract mean attention to soft tokens per layer
        soft_attn_means = [layer["soft_token_attn_mean"] for layer in stats["per_layer"]]
        soft_attn_stds = [layer["soft_token_attn_std"] for layer in stats["per_layer"]]
        anchor_attn_means = [layer["anchor_attn_mean"] for layer in stats["per_layer"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Mean attention by layer
        ax1.plot(layers, soft_attn_means, marker='o', label='Soft Tokens', linewidth=2)
        ax1.fill_between(
            layers,
            [m - s for m, s in zip(soft_attn_means, soft_attn_stds)],
            [m + s for m, s in zip(soft_attn_means, soft_attn_stds)],
            alpha=0.3,
        )
        if any(a > 0 for a in anchor_attn_means):
            ax1.plot(layers, anchor_attn_means, marker='s', label='Anchor Tokens', linewidth=2)

        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Mean Attention Weight', fontsize=12)
        ax1.set_title('Attention to Soft Tokens vs. Anchor Tokens by Layer', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Per-head attention for selected layers
        selected_layers = [0, num_layers // 2, num_layers - 1]
        for layer_idx in selected_layers:
            if layer_idx < num_layers:
                per_head = stats["per_layer"][layer_idx]["per_head_soft_attn"]
                ax2.plot(per_head, marker='o', label=f'Layer {layer_idx}', alpha=0.7)

        ax2.set_xlabel('Head Index', fontsize=12)
        ax2.set_ylabel('Mean Attention to Soft Tokens', fontsize=12)
        ax2.set_title('Per-Head Attention to Soft Tokens (Selected Layers)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved attention analysis: {save_path}")

    def compare_text_vs_latent_attention(
        self,
        text_stats: Dict[str, Any],
        latent_stats: Dict[str, Any],
        save_name: str = "text_vs_latent_attention.png",
    ):
        """
        Compare attention patterns between text baseline and latent conditioning.

        Args:
            text_stats: Attention statistics for text baseline
            latent_stats: Attention statistics for latent conditioning
            save_name: Output filename
        """
        num_layers = latent_stats["num_layers"]
        layers = list(range(num_layers))

        # For text baseline, we don't have "soft tokens", but we can compare
        # attention patterns to the prompt prefix
        latent_soft_attn = [layer["soft_token_attn_mean"] for layer in latent_stats["per_layer"]]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(layers, latent_soft_attn, marker='o', label='Latent: Soft Tokens', linewidth=2)

        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Mean Attention Weight', fontsize=12)
        ax.set_title('Attention to Soft Tokens (Latent Conditioning)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved comparison: {save_path}")


def main():
    args = parse_args()

    print("=" * 80)
    print("LatentWire Attention Pattern Analysis")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples}")
    print(f"Model: {args.model_key}")
    print(f"Output: {args.output_dir}")
    print()

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found: {args.ckpt}")

    config = json.loads((ckpt_path / "config.json").read_text())
    print("Loading models and encoder...")

    # Load encoder
    encoder = InterlinguaEncoder(
        d_z=config["d_z"],
        n_layers=config.get("encoder_layers", 6),
        n_heads=config.get("encoder_heads", 8),
        ff_mult=config.get("encoder_ff_mult", 4),
        latent_len=config["latent_len"],
    )
    encoder_state = torch.load(ckpt_path / "encoder.pt", map_location=args.device)
    encoder.load_state_dict(encoder_state)
    encoder = encoder.to(args.device).eval()

    # Load model wrapper
    model_id = config.get(f"{args.model_key}_id", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    wrapper_cfg = LMConfig(
        model_id=model_id,
        device=args.device,
        dtype=torch.bfloat16,
    )
    wrapper = LMWrapper(wrapper_cfg)

    # Load adapter
    adapter = Adapter(
        d_z=config["d_z"],
        d_model=wrapper.model.config.hidden_size,
        latent_length=config["latent_len"],
    )
    adapter_state = torch.load(
        ckpt_path / f"adapter_{args.model_key}.pt",
        map_location=args.device
    )
    adapter.load_state_dict(adapter_state)
    adapter = adapter.to(args.device).eval()

    print(f"Loaded {model_id}")
    print()

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    examples = load_examples(args.dataset, split="validation", samples=args.samples)
    print(f"Loaded {len(examples)} examples")
    print()

    # Set up extractors and analyzers
    extractor = AttentionExtractor(wrapper)
    analyzer = AttentionAnalyzer(args.output_dir)

    # Parse layer and head indices
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    head_indices = [int(x.strip()) for x in args.heads.split(",")]

    print("Analyzing attention patterns...")
    print()

    # Analyze first N examples
    all_stats = []

    for idx, ex in enumerate(examples[:min(args.samples, len(examples))]):
        print(f"Example {idx + 1}/{min(args.samples, len(examples))}: {ex['question'][:60]}...")

        # Encode question
        question_bytes = ex["question"].encode("utf-8")
        byte_ids = torch.tensor(list(question_bytes), dtype=torch.long).unsqueeze(0).to(args.device)

        with torch.no_grad():
            z = encoder(byte_ids)  # [1, M, d_z]
            prefix_embeds = adapter(z)  # [1, M, d_model]

        # Extract attention for first token generation
        anchor_text = "Answer: "
        result = extractor.extract_first_token_attention(
            prefix_embeds,
            anchor_token_text=anchor_text,
            append_bos_after_prefix=True,
        )

        # Compute statistics
        stats = analyzer.compute_soft_token_attention_stats(
            result["attentions"],
            prefix_len=result["prefix_len"],
            anchor_len=result["anchor_len"],
            bos_len=result["bos_len"],
        )
        all_stats.append(stats)

        # Visualize selected layers and heads for first example
        if idx == 0:
            attentions = result["attentions"]
            for layer_idx in layer_indices:
                if layer_idx >= len(attentions):
                    continue
                attn = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]

                for head_idx in head_indices:
                    if head_idx >= attn.size(0):
                        continue
                    analyzer.visualize_attention_heatmap(
                        attn[head_idx],
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        prefix_len=result["prefix_len"],
                        anchor_len=result["anchor_len"],
                        bos_len=result["bos_len"],
                        example_id=str(idx),
                    )

        print(f"  Soft token attention (layer 0): {stats['per_layer'][0]['soft_token_attn_mean']:.4f}")
        print(f"  Soft token attention (layer -1): {stats['per_layer'][-1]['soft_token_attn_mean']:.4f}")
        print()

    # Aggregate statistics across examples
    print("Aggregating statistics across examples...")
    num_layers = all_stats[0]["num_layers"]

    aggregated_stats = {
        "num_layers": num_layers,
        "per_layer": [],
    }

    for layer_idx in range(num_layers):
        layer_soft_attn_means = [stats["per_layer"][layer_idx]["soft_token_attn_mean"] for stats in all_stats]
        layer_anchor_attn_means = [stats["per_layer"][layer_idx]["anchor_attn_mean"] for stats in all_stats]

        # Average per-head attention across examples
        per_head_lists = [stats["per_layer"][layer_idx]["per_head_soft_attn"] for stats in all_stats]
        num_heads = len(per_head_lists[0])
        per_head_avg = [
            np.mean([per_head[h] for per_head in per_head_lists])
            for h in range(num_heads)
        ]

        aggregated_stats["per_layer"].append({
            "layer_idx": layer_idx,
            "num_heads": num_heads,
            "soft_token_attn_mean": np.mean(layer_soft_attn_means),
            "soft_token_attn_std": np.std(layer_soft_attn_means),
            "soft_token_attn_min": np.min(layer_soft_attn_means),
            "soft_token_attn_max": np.max(layer_soft_attn_means),
            "anchor_attn_mean": np.mean(layer_anchor_attn_means),
            "per_head_soft_attn": per_head_avg,
        })

    # Save aggregated statistics
    stats_path = Path(args.output_dir) / "attention_stats.json"
    with open(stats_path, "w") as f:
        json.dump(aggregated_stats, f, indent=2)
    print(f"Saved aggregated statistics: {stats_path}")
    print()

    # Create visualizations
    print("Creating visualizations...")
    analyzer.visualize_attention_to_soft_tokens(aggregated_stats)
    print()

    print("=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
