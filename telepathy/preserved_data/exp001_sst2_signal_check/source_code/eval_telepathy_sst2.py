#!/usr/bin/env python
# telepathy/eval_telepathy_sst2.py
"""
Phase 16: SST-2 Evaluation (Continuous Version)

Evaluates bridge on sentiment classification.
Success criteria:
  - Accuracy > 50%: Bridge transmits some information
  - Accuracy > 70%: Bridge is working
  - Accuracy > 85%: Bridge is excellent
  - Accuracy ~ 50%: Bridge is broken (random chance)

Uses CONTINUOUS soft tokens (not VQ).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os

from latent_bridge_v15 import LatentBridgeV15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=872)  # Full validation set
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--bf16", action="store_true", default=True)
    # Continuous mode (no VQ/FSQ)
    parser.add_argument("--use_fsq", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Phase 16 Evaluation: SST-2 Sentiment Classification")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print("")
    print("Success Criteria:")
    print("  - Accuracy > 50%: Bridge transmits info")
    print("  - Accuracy > 70%: Bridge is working")
    print("  - Accuracy > 85%: Bridge is excellent")
    print("  - Accuracy ~ 50%: Bridge is broken")
    print("=" * 60)
    print("")

    # Load models
    print("Loading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()

    # Load bridge (CONTINUOUS, not VQ)
    bridge = LatentBridgeV15(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    bridge.load_state_dict(checkpoint)
    bridge.to(DEVICE)
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load SST-2 validation set
    ds = load_dataset("glue", "sst2", split="validation")
    num_samples = min(args.num_samples, len(ds))

    print(f"\nEvaluating {num_samples} samples...")
    print("-" * 60)

    correct = 0
    total = 0
    results = []

    # Track predictions for analysis
    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0

    for i in tqdm(range(num_samples), desc="Evaluating"):
        item = ds[i]
        text = item['sentence']
        label = "positive" if item['label'] == 1 else "negative"

        # Source
        src_input = f"Review: {text}\nSentiment:"
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            # Target: [Primer] + [Soft Tokens] -> Generate
            primer = "Sentiment:"
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(DEVICE)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=DEVICE, dtype=torch.long)

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

        # Check prediction
        pred = "positive" if "positive" in output else ("negative" if "negative" in output else "unknown")
        is_correct = (label in output)

        if is_correct:
            correct += 1
        total += 1

        # Track per-class accuracy
        if label == "positive":
            positive_total += 1
            if is_correct:
                positive_correct += 1
        else:
            negative_total += 1
            if is_correct:
                negative_correct += 1

        results.append({
            "idx": i,
            "text": text[:80] + "..." if len(text) > 80 else text,
            "label": label,
            "prediction": pred,
            "output": output[:50],
            "correct": is_correct
        })

        # Print first 10 samples
        if i < 10:
            status = "CORRECT" if is_correct else "wrong"
            print(f"\n[{i}] {status}")
            print(f"  Text: {text[:60]}...")
            print(f"  GT: {label} | Pred: {output[:30]}")

    # Summary
    accuracy = 100 * correct / total
    pos_acc = 100 * positive_correct / max(positive_total, 1)
    neg_acc = 100 * negative_correct / max(negative_total, 1)

    print("\n" + "=" * 60)
    print("PHASE 16 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nOVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Positive: {positive_correct}/{positive_total} ({pos_acc:.1f}%)")
    print(f"  Negative: {negative_correct}/{negative_total} ({neg_acc:.1f}%)")
    print("")

    # Interpretation
    if accuracy < 55:
        print("=" * 60)
        print("FAILURE: Accuracy near random (50%)")
        print("The bridge is NOT transmitting meaningful information.")
        print("Possible causes:")
        print("  - VQ codebook collapsed")
        print("  - Bridge architecture issue")
        print("  - Training didn't converge")
        print("=" * 60)
    elif accuracy < 70:
        print("=" * 60)
        print("PARTIAL SUCCESS: Some information transfers")
        print("The bridge is learning but not yet effective.")
        print("Consider: More training, larger model, tune hyperparameters")
        print("=" * 60)
    elif accuracy < 85:
        print("=" * 60)
        print("SUCCESS: Bridge is working!")
        print("Semantic information transfers through the bridge.")
        print("Next: Try harder tasks (topic classification, then QA)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXCELLENT: Bridge works very well!")
        print("Ready for more challenging tasks.")
        print("Next: Scale up to multi-step reasoning (GSM8K)")
        print("=" * 60)

    # Save results
    output_path = os.path.join(args.output_dir, "eval_sst2_results.json")
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "samples": results[:20]  # Save first 20 for inspection
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
