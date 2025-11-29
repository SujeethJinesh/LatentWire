#!/usr/bin/env python
# telepathy/eval_telepathy.py
"""
Telepathy Evaluation: Verify if cross-model latent communication actually works.

This script tests on the HELD-OUT test set (not training data) to detect:
1. Success: Mistral correctly answers math problems via soft tokens
2. Partial Success: Topic transmitted but details lost
3. Posterior Collapse: Model ignores bridge, outputs generic text
4. Hallucination: Vectors map to unrelated concepts

Usage:
    python telepathy/eval_telepathy.py --checkpoint runs/telepathy_*/bridge_final.pt
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Import all bridge versions
from latent_bridge import LatentBridge
try:
    from latent_bridge_v3 import LatentBridgeV3
except ImportError:
    LatentBridgeV3 = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Telepathy Bridge")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to bridge checkpoint (bridge_final.pt)"
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Path to stats.pt (defaults to same dir as checkpoint)"
    )
    parser.add_argument(
        "--source_model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument(
        "--source_layer",
        type=int,
        default=20,
        help="Layer to extract hidden states from"
    )
    parser.add_argument(
        "--soft_tokens",
        type=int,
        default=64
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of test samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to checkpoint dir)"
    )
    parser.add_argument(
        "--bridge_version",
        type=int,
        default=None,
        help="Bridge version (1, 2, or 3). Auto-detected from checkpoint name if not specified."
    )
    return parser.parse_args()


def extract_final_answer(text: str) -> str | None:
    """Extract the final numerical answer from GSM8K format."""
    # GSM8K answers end with #### <number>
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            return parts[-1].strip().split()[0] if parts[-1].strip() else None
    # Try to find last number in text
    import re
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    return numbers[-1] if numbers else None


def main():
    args = parse_args()
    start_time = time.time()

    # Resolve paths
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if args.stats_path is None:
        args.stats_path = os.path.join(checkpoint_dir, "stats.pt")
    if args.output_dir is None:
        args.output_dir = checkpoint_dir

    print("=" * 70)
    print("Telepathy Evaluation: Cross-Model Latent Communication Test")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Stats: {args.stats_path}")
    print(f"Samples: {args.num_samples}")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load Source Model
    print(f"\n[1/4] Loading Source Model: {args.source_model}...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token

    # Load Target Model
    print(f"[2/4] Loading Target Model: {args.target_model}...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Detect bridge version from checkpoint name if not specified
    bridge_version = args.bridge_version
    if bridge_version is None:
        if "v3" in args.checkpoint:
            bridge_version = 3
        elif "v2" in args.checkpoint:
            bridge_version = 2
        else:
            bridge_version = 1

    print(f"[3/4] Loading Bridge V{bridge_version} from {args.checkpoint}...")

    class BridgeArgs:
        stats_path = args.stats_path
        soft_tokens = args.soft_tokens
        heads = args.heads
        depth = args.depth

    # Calculate target RMS for V3
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

    if bridge_version == 3:
        if LatentBridgeV3 is None:
            raise ImportError("LatentBridgeV3 not found. Check latent_bridge_v3.py exists.")
        bridge = LatentBridgeV3(
            BridgeArgs(),
            src_model.config.hidden_size,
            tgt_model.config.hidden_size,
            target_rms=target_rms
        )
    else:
        bridge = LatentBridge(
            BridgeArgs(),
            src_model.config.hidden_size,
            tgt_model.config.hidden_size
        )
    bridge.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    bridge.to(device).bfloat16().eval()

    num_params = sum(p.numel() for p in bridge.parameters())
    print(f"  Bridge parameters: {num_params:,}")

    # Load Test Data (HELD OUT - not used in training)
    print("[4/4] Loading GSM8K TEST split (held-out data)...")
    ds = load_dataset("gsm8k", "main", split="test")
    print(f"  Test set size: {len(ds)}")

    # Select samples spread across the test set
    step = max(1, len(ds) // args.num_samples)
    indices = list(range(0, min(len(ds), args.num_samples * step), step))[:args.num_samples]

    print(f"\n{'=' * 70}")
    print("STARTING TELEPATHY EVALUATION")
    print(f"{'=' * 70}")

    results = []
    correct = 0
    partial = 0

    for idx, i in enumerate(indices):
        question = ds[i]['question']
        ground_truth = ds[i]['answer']
        gt_answer = extract_final_answer(ground_truth)

        print(f"\n[Test {idx+1}/{len(indices)}] Sample {i}")
        print("-" * 70)
        print(f"QUESTION (Seen by Llama only):\n{question[:500]}...")

        # 1. Source Forward - Extract Llama's "thoughts"
        src_input_text = f"Question: {question}\nAnswer:"
        src_inputs = src_tok(
            src_input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer].bfloat16()

        # 2. Bridge Forward - Compress to soft tokens
        with torch.no_grad():
            soft_tokens = bridge(src_h, src_inputs.attention_mask)

        # 3. Target Generation - Mistral receives only soft tokens
        primer = "Answer: "  # Must match training primer
        primer_inputs = tgt_tok(primer, return_tensors="pt").to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_inputs.input_ids)

        # Combine: [Primer Embeddings] + [Soft Tokens from Bridge]
        inputs_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
        attention_mask = torch.ones(1, inputs_embeds.shape[1], device=device)

        print("Mistral generating from soft tokens...", end=" ", flush=True)
        gen_start = time.time()

        with torch.no_grad():
            output_ids = tgt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tgt_tok.eos_token_id,
                eos_token_id=tgt_tok.eos_token_id,
            )

        gen_time = time.time() - gen_start
        output_text = tgt_tok.decode(output_ids[0], skip_special_tokens=True)
        pred_answer = extract_final_answer(output_text)

        print(f"Done ({gen_time:.1f}s)")

        # Evaluate
        is_correct = pred_answer == gt_answer if pred_answer and gt_answer else False
        is_partial = (
            any(word in output_text.lower() for word in question.lower().split()[:5])
            if not is_correct else False
        )

        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        elif is_partial:
            partial += 1
            status = "~ PARTIAL"
        else:
            status = "✗ FAILED"

        print("-" * 70)
        print(f"MISTRAL OUTPUT:\n{output_text[:600]}")
        print("-" * 70)
        print(f"GROUND TRUTH ANSWER: {gt_answer}")
        print(f"PREDICTED ANSWER:    {pred_answer}")
        print(f"STATUS: {status}")
        print("=" * 70)

        results.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "gt_answer": gt_answer,
            "output": output_text,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "partial": is_partial,
            "gen_time": gen_time,
        })

    # Summary
    elapsed = time.time() - start_time
    accuracy = correct / len(indices) * 100
    partial_rate = partial / len(indices) * 100

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples:    {len(indices)}")
    print(f"Correct:          {correct} ({accuracy:.1f}%)")
    print(f"Partial:          {partial} ({partial_rate:.1f}%)")
    print(f"Failed:           {len(indices) - correct - partial}")
    print(f"Total time:       {elapsed/60:.1f} minutes")
    print("=" * 70)

    # Diagnosis
    print("\nDIAGNOSIS:")
    if accuracy > 10:
        print("  ✓ SUCCESS - Telepathy is working! Mistral understood the soft tokens.")
    elif partial_rate > 30:
        print("  ~ PARTIAL SUCCESS - Bridge transmits topic but loses details.")
        print("    Recommendation: Increase soft_tokens or add more Perceiver layers.")
    elif accuracy == 0 and partial_rate < 10:
        # Check for collapse patterns
        outputs_similar = len(set(r['output'][:100] for r in results)) < len(results) / 2
        if outputs_similar:
            print("  ✗ POSTERIOR COLLAPSE - Model ignores bridge, outputs generic text.")
            print("    Recommendation: Increase recon_weight or add InfoNCE loss.")
        else:
            print("  ✗ ALIGNMENT FAILURE - Vectors map to random concepts.")
            print("    Recommendation: Check calibration stats, try different source layer.")
    else:
        print("  ? INCONCLUSIVE - Need more samples or investigation.")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "checkpoint": args.checkpoint,
                "source_model": args.source_model,
                "target_model": args.target_model,
                "num_samples": len(indices),
            },
            "summary": {
                "correct": correct,
                "partial": partial,
                "accuracy": accuracy,
                "partial_rate": partial_rate,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
