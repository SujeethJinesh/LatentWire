#!/usr/bin/env python
# telepathy/eval_telepathy.py
"""
Telepathy Evaluation with Text Baseline Comparison.

Runs two parallel inference passes for every question:
1. BASELINE (Text): Mistral reads the question directly (upper bound)
2. TELEPATHY (Vectors): Llama's vectors through Bridge to Mistral

This proves whether latent communication preserves semantic content.

Usage:
    python telepathy/eval_telepathy.py --checkpoint runs/telepathy_*/bridge_v5_final.pt
"""
import argparse
import json
import os
import re
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
try:
    from latent_bridge_v7 import LatentBridgeV7
except ImportError:
    LatentBridgeV7 = None
try:
    from latent_bridge_v8 import LatentBridgeV8
except ImportError:
    LatentBridgeV8 = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Telepathy Bridge vs Text Baseline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to bridge checkpoint")
    parser.add_argument("--stats_path", type=str, default=None, help="Path to stats.pt")
    parser.add_argument("--source_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=16, help="Layer to extract hidden states from")
    parser.add_argument("--soft_tokens", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bridge_version", type=int, default=None, help="Auto-detected if not specified")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip text baseline (faster)")
    return parser.parse_args()


def extract_final_answer(text: str) -> str | None:
    """Extract numerical answer from GSM8K format."""
    # GSM8K answers end with #### <number>
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            answer_part = parts[-1].strip()
            if answer_part:
                return answer_part.split()[0]
    # Fallback: find last number
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
    print("Telepathy Evaluation: Text vs Latent Comparison")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source Layer: {args.source_layer}")
    print(f"Soft Tokens: {args.soft_tokens}")
    print(f"Samples: {args.num_samples}")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load Source Model (Llama)
    print(f"\n[1/4] Loading Source Model: {args.source_model}...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token

    # Load Target Model (Mistral)
    print(f"[2/4] Loading Target Model: {args.target_model}...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Detect bridge version
    bridge_version = args.bridge_version
    if bridge_version is None:
        if "v8" in args.checkpoint:
            bridge_version = 8
        elif "v7" in args.checkpoint:
            bridge_version = 7
        elif "v5" in args.checkpoint or "v4" in args.checkpoint or "v3" in args.checkpoint:
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

    if bridge_version == 8:
        if LatentBridgeV8 is None:
            raise ImportError("LatentBridgeV8 not found. Check latent_bridge_v8.py exists.")
        bridge = LatentBridgeV8(
            BridgeArgs(),
            src_model.config.hidden_size,
            tgt_model.config.hidden_size,
            target_rms=target_rms
        )
    elif bridge_version == 7:
        if LatentBridgeV7 is None:
            raise ImportError("LatentBridgeV7 not found. Check latent_bridge_v7.py exists.")
        bridge = LatentBridgeV7(
            BridgeArgs(),
            src_model.config.hidden_size,
            tgt_model.config.hidden_size,
            target_rms=target_rms
        )
    elif bridge_version == 3:
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

    # Load Test Data
    print("[4/4] Loading GSM8K TEST split...")
    ds = load_dataset("gsm8k", "main", split="test")
    print(f"  Test set size: {len(ds)}")

    # Select samples spread across test set
    step = max(1, len(ds) // args.num_samples)
    indices = list(range(0, min(len(ds), args.num_samples * step), step))[:args.num_samples]

    print(f"\n{'=' * 70}")
    print("STARTING EVALUATION: TEXT vs TELEPATHY")
    print(f"{'=' * 70}")

    results = []
    baseline_correct = 0
    telepathy_correct = 0
    telepathy_partial = 0

    for idx, i in enumerate(indices):
        question = ds[i]['question']
        ground_truth = ds[i]['answer']
        gt_answer = extract_final_answer(ground_truth)

        print(f"\n[Test {idx+1}/{len(indices)}] Sample {i}")
        print("-" * 70)
        print(f"QUESTION:\n{question[:300]}...")
        print(f"GROUND TRUTH: {gt_answer}")
        print("-" * 70)

        result = {
            "index": i,
            "question": question,
            "gt_answer": gt_answer,
        }

        # =====================================================================
        # BASELINE: Text-to-Text (Mistral reads question directly)
        # =====================================================================
        if not args.skip_baseline:
            text_input = f"Question: {question}\nAnswer:"
            base_inputs = tgt_tok(text_input, return_tensors="pt", truncation=True, max_length=1024).to(device)

            with torch.no_grad():
                base_out = tgt_model.generate(
                    **base_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tgt_tok.eos_token_id,
                    eos_token_id=tgt_tok.eos_token_id,
                )

            base_full = tgt_tok.decode(base_out[0], skip_special_tokens=True)
            base_resp = base_full.split("Answer:")[-1].strip() if "Answer:" in base_full else base_full
            base_pred = extract_final_answer(base_resp)
            base_correct = (base_pred == gt_answer) if base_pred and gt_answer else False

            if base_correct:
                baseline_correct += 1

            result["baseline"] = {
                "response": base_resp[:500],
                "pred": base_pred,
                "correct": base_correct,
            }
            print(f"BASELINE (Text):     {base_pred} {'✓' if base_correct else '✗'}")

        # =====================================================================
        # TELEPATHY: Latent-to-Text (Llama vectors through Bridge to Mistral)
        # =====================================================================
        src_input_text = f"Question: {question}\nAnswer:"
        src_inputs = src_tok(src_input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # 1. Extract Llama's hidden states
        with torch.no_grad():
            src_out = src_model(**src_inputs, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer].bfloat16()

        # 2. Bridge: Compress to soft tokens
        with torch.no_grad():
            bridge_out = bridge(src_h, src_inputs.attention_mask)
            # V7/V8 return tuple (scaled, raw), earlier versions return just soft_tokens
            if isinstance(bridge_out, tuple):
                soft_tokens = bridge_out[0]  # Use scaled output
            else:
                soft_tokens = bridge_out

        # 3. Mistral generates from soft tokens
        primer = "Answer: "  # Must match training
        primer_inputs = tgt_tok(primer, return_tensors="pt").to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_inputs.input_ids)

        # Combine: [Primer] + [Soft Tokens]
        inputs_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
        attention_mask = torch.ones(1, inputs_embeds.shape[1], device=device)

        with torch.no_grad():
            tele_out = tgt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
                eos_token_id=tgt_tok.eos_token_id,
            )

        tele_resp = tgt_tok.decode(tele_out[0], skip_special_tokens=True)
        tele_pred = extract_final_answer(tele_resp)
        tele_correct = (tele_pred == gt_answer) if tele_pred and gt_answer else False

        # Check for partial success (topic/entity transfer)
        question_words = set(question.lower().split()[:10])
        output_words = set(tele_resp.lower().split())
        tele_partial = len(question_words & output_words) >= 2 if not tele_correct else False

        if tele_correct:
            telepathy_correct += 1
        elif tele_partial:
            telepathy_partial += 1

        result["telepathy"] = {
            "response": tele_resp[:500],
            "pred": tele_pred,
            "correct": tele_correct,
            "partial": tele_partial,
        }

        print(f"TELEPATHY (Vectors): {tele_pred} {'✓' if tele_correct else '~' if tele_partial else '✗'}")
        print(f"  Output: {tele_resp[:200]}...")

        results.append(result)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    n = len(indices)

    baseline_acc = baseline_correct / n * 100 if not args.skip_baseline else 0
    telepathy_acc = telepathy_correct / n * 100
    telepathy_partial_rate = telepathy_partial / n * 100

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {n}")
    print("-" * 70)
    if not args.skip_baseline:
        print(f"BASELINE (Text→Mistral):     {baseline_correct}/{n} ({baseline_acc:.1f}%)")
    print(f"TELEPATHY (Llama→Bridge→Mistral):")
    print(f"  Correct:  {telepathy_correct}/{n} ({telepathy_acc:.1f}%)")
    print(f"  Partial:  {telepathy_partial}/{n} ({telepathy_partial_rate:.1f}%)")
    print(f"  Failed:   {n - telepathy_correct - telepathy_partial}/{n}")
    print("-" * 70)
    if not args.skip_baseline and baseline_correct > 0:
        retention = telepathy_correct / baseline_correct * 100 if baseline_correct > 0 else 0
        print(f"RETENTION: {retention:.1f}% of baseline performance")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("=" * 70)

    # Diagnosis
    print("\nDIAGNOSIS:")
    if telepathy_acc > 10:
        print("  ✓ SUCCESS - Telepathy is working! Latent communication preserved.")
    elif telepathy_acc > 0:
        print("  ~ PROGRESS - Some correct answers. Need more capacity or training.")
    elif telepathy_partial_rate > 50:
        print("  ~ PARTIAL - Topic transfers but entities scrambled (lossy compression).")
        print("    Try: More soft tokens, lower source layer, stronger anchor weight.")
    else:
        print("  ✗ FAILURE - Semantic content not transferring.")
        print("    Check: Calibration stats, primer consistency, bridge architecture.")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "checkpoint": args.checkpoint,
                "source_model": args.source_model,
                "target_model": args.target_model,
                "source_layer": args.source_layer,
                "soft_tokens": args.soft_tokens,
                "num_samples": n,
            },
            "summary": {
                "baseline_correct": baseline_correct if not args.skip_baseline else None,
                "baseline_accuracy": baseline_acc if not args.skip_baseline else None,
                "telepathy_correct": telepathy_correct,
                "telepathy_partial": telepathy_partial,
                "telepathy_accuracy": telepathy_acc,
                "telepathy_partial_rate": telepathy_partial_rate,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
