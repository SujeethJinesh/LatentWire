#!/usr/bin/env python
# telepathy/eval_telepathy_agnews.py
"""
Phase 17: AG News Evaluation

Evaluates bridge on 4-class news topic classification.
Success criteria:
  - Random baseline: 25%
  - Accuracy > 50%: Bridge works for multi-class
  - Accuracy > 70%: Bridge is excellent
  - Accuracy matches Mistral text: Perfect transfer

Uses optimal config from SST-2 ablation:
- Layer 31, 8 soft tokens, continuous mode
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os

from latent_bridge_v15 import LatentBridgeV15

# AG News class labels
AGNEWS_LABELS = ["world", "sports", "business", "science"]

# Permissive matching for science/tech (AG News uses "Sci/Tech")
SCIENCE_SYNONYMS = ["science", "technology", "tech", "sci/tech", "scitech"]


def check_label_match(label, output):
    """Check if label matches output, with permissive matching for science."""
    if label == "science":
        return any(syn in output for syn in SCIENCE_SYNONYMS)
    return label in output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=31)
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_fsq", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Phase 17 Evaluation: AG News 4-Class Classification")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {', '.join(AGNEWS_LABELS)}")
    print("")
    print("Success Criteria:")
    print("  - Random baseline: 25%")
    print("  - Accuracy > 50%: Bridge works for multi-class")
    print("  - Accuracy > 70%: Bridge is excellent")
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

    # Load bridge
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

    # Load AG News test set
    ds = load_dataset("ag_news", split="test")
    num_samples = min(args.num_samples, len(ds))

    print(f"\nEvaluating {num_samples} samples...")
    print("-" * 60)

    correct = 0
    total = 0
    results = []

    # Track per-class accuracy
    class_correct = {l: 0 for l in AGNEWS_LABELS}
    class_total = {l: 0 for l in AGNEWS_LABELS}

    for i in tqdm(range(num_samples), desc="Evaluating"):
        item = ds[i]
        text = item['text']
        label = AGNEWS_LABELS[item['label']]

        # Source (same prompt format as text baseline for fair comparison)
        src_input = f"Article: {text[:256]}\nTopic (world, sports, business, or science):"
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            # Target: [Primer] + [Soft Tokens] -> Generate
            primer = "Topic:"
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

        # Check prediction - find which label appears in output
        pred = "unknown"
        for candidate in AGNEWS_LABELS:
            if check_label_match(candidate, output):
                pred = candidate
                break

        is_correct = check_label_match(label, output)

        if is_correct:
            correct += 1
            class_correct[label] += 1
        total += 1
        class_total[label] += 1

        results.append({
            "idx": i,
            "text": text[:80] + "..." if len(text) > 80 else text,
            "label": label,
            "prediction": pred,
            "output": output[:50],
            "correct": is_correct
        })

        # Print first 12 samples (3 per class ideally)
        if i < 12:
            status = "CORRECT" if is_correct else "wrong"
            print(f"\n[{i}] {status}")
            print(f"  Text: {text[:60]}...")
            print(f"  GT: {label} | Pred: {output[:30]}")

    # Summary
    accuracy = 100 * correct / total

    print("\n" + "=" * 60)
    print("PHASE 17 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nOVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print(f"\nPer-class breakdown:")
    for label in AGNEWS_LABELS:
        if class_total[label] > 0:
            class_acc = 100 * class_correct[label] / class_total[label]
            print(f"  {label:10}: {class_correct[label]:4}/{class_total[label]:4} ({class_acc:.1f}%)")

    # Interpretation
    print("")
    if accuracy < 30:
        print("=" * 60)
        print("FAILURE: Accuracy near random (25%)")
        print("The bridge is NOT transmitting meaningful multi-class info.")
        print("=" * 60)
    elif accuracy < 50:
        print("=" * 60)
        print("PARTIAL SUCCESS: Some multi-class info transfers")
        print("Better than random, but room for improvement.")
        print("=" * 60)
    elif accuracy < 70:
        print("=" * 60)
        print("SUCCESS: Bridge works for multi-class!")
        print("Good transfer of topic information.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXCELLENT: Bridge works very well for 4-class!")
        print("Strong multi-class transfer. Ready for harder tasks.")
        print("=" * 60)

    # Save results
    output_path = os.path.join(args.output_dir, "eval_agnews_results.json")
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class": {
            label: {
                "correct": class_correct[label],
                "total": class_total[label],
                "accuracy": 100 * class_correct[label] / max(class_total[label], 1)
            }
            for label in AGNEWS_LABELS
        },
        "samples": results[:20]
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
