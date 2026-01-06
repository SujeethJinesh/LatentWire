#!/usr/bin/env python
# telepathy/phase1_calibration.py
"""
Phase 1: Statistical Calibration for Latent Bridge.

Collects distribution statistics from source (Llama) and target (Mistral) models
to enable proper normalization during latent transfer.

This solves the "Magnitude Shock" problem:
- Llama hidden states have different scale/distribution than Mistral embeddings
- Without calibration, direct injection causes gradient explosion
"""
import argparse
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibration: Collect statistics for Latent Bridge normalization"
    )
    parser.add_argument(
        "--source_model",
        type=str,
        required=True,
        help="Source model ID (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--target_model",
        type=str,
        required=True,
        help="Target model ID (e.g., mistralai/Mistral-7B-Instruct-v0.3)"
    )
    parser.add_argument(
        "--source_layer",
        type=int,
        default=20,
        help="Layer to extract hidden states from source model"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples for calibration"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for calibration"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="stats.pt",
        help="Output file for statistics"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Phase 1: Statistical Calibration")
    print("=" * 60)
    print(f"Source: {args.source_model}")
    print(f"Target: {args.target_model}")
    print(f"Source layer: {args.source_layer}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output_file}")
    print("=" * 60)

    # Load Source Model
    print(f"\n[1/4] Loading Source Model: {args.source_model}...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()
    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    print(f"  Hidden size: {src_model.config.hidden_size}")
    print(f"  Num layers: {src_model.config.num_hidden_layers}")

    # Load Target Model
    print(f"\n[2/4] Loading Target Model: {args.target_model}...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token
    print(f"  Hidden size: {tgt_model.config.hidden_size}")
    print(f"  Num layers: {tgt_model.config.num_hidden_layers}")

    # Load Dataset
    print("\n[3/4] Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main", split="train")
    print(f"  Total samples available: {len(ds)}")

    # Collect Statistics
    print(f"\n[4/4] Running Calibration ({args.num_samples} samples)...")
    src_stats = []
    tgt_stats = []

    with torch.no_grad():
        for i in tqdm(range(0, args.num_samples, args.batch_size), desc="Calibrating"):
            batch_end = min(i + args.batch_size, args.num_samples)
            batch_data = ds[i:batch_end]
            prompts = batch_data["question"]

            # Source Statistics: Extract hidden states from specified layer
            inputs = src_tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to("cuda:0")

            out = src_model(**inputs, output_hidden_states=True)
            h_src = out.hidden_states[args.source_layer].float()

            # Compute mean over non-padded positions
            mask = inputs["attention_mask"].unsqueeze(-1)
            active_h = (h_src * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            src_stats.append(active_h.cpu())

            # Target Statistics: Extract embedding layer activations
            t_inputs = tgt_tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to("cuda:0")

            t_embeds = tgt_model.get_input_embeddings()(t_inputs.input_ids).float()
            t_mask = t_inputs["attention_mask"].unsqueeze(-1)
            active_t = (t_embeds * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp(min=1)
            tgt_stats.append(active_t.cpu())

    # Compute final statistics
    src_all = torch.cat(src_stats, dim=0)
    tgt_all = torch.cat(tgt_stats, dim=0)

    l_mean = src_all.mean(dim=0)
    l_std = src_all.std(dim=0) + 1e-6
    m_mean = tgt_all.mean(dim=0)
    m_std = tgt_all.std(dim=0) + 1e-6

    # Calculate target embedding RMS (for scale reference)
    t_weights = tgt_model.get_input_embeddings().weight.float().cpu()
    target_rms = t_weights.pow(2).mean(dim=1).sqrt().median().item()

    # Report results
    print("\n" + "=" * 60)
    print("Calibration Results")
    print("=" * 60)
    print(f"Source (Layer {args.source_layer}):")
    print(f"  Mean norm: {l_mean.norm().item():.4f}")
    print(f"  Std mean:  {l_std.mean().item():.4f}")
    print(f"Target (Embeddings):")
    print(f"  Mean norm: {m_mean.norm().item():.4f}")
    print(f"  Std mean:  {m_std.mean().item():.4f}")
    print(f"  RMS:       {target_rms:.4f}")
    print(f"Scale ratio: {m_mean.norm().item() / l_mean.norm().item():.2f}x")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save statistics
    torch.save({
        "l_mean": l_mean,
        "l_std": l_std,
        "m_mean": m_mean,
        "m_std": m_std,
        "target_rms": target_rms,
        "source_layer": args.source_layer,
        "source_model": args.source_model,
        "target_model": args.target_model,
        "num_samples": args.num_samples,
    }, args.output_file)

    print(f"\nSaved statistics to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
