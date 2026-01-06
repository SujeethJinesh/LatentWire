#!/usr/bin/env python
# telepathy/eval_telepathy_trec.py
"""
TREC Question Classification Evaluation

Evaluates bridge on 6-class question classification task.
Success criteria:
  - Random baseline: ~16.7% (1/6)
  - Accuracy > 30%: Bridge transmits some information
  - Accuracy > 50%: Bridge is working well
  - Accuracy > 70%: Bridge is excellent
  - Accuracy ~ 16.7%: Bridge is broken (random chance)

Dataset: TREC Question Classification
- Test set: 500 samples
- Classes: ABBR, ENTY, DESC, HUM, LOC, NUM

Uses CONTINUOUS soft tokens (not VQ).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os
import numpy as np
from scipy import stats

from latent_bridge_v15 import LatentBridgeV15

# TREC class labels
TREC_LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]

# Label descriptions for context
LABEL_DESCRIPTIONS = {
    "ABBR": "abbreviation",
    "ENTY": "entity",
    "DESC": "description",
    "HUM": "human",
    "LOC": "location",
    "NUM": "numeric"
}


def check_label_match(label, output):
    """
    Check if label matches output.
    Try both uppercase (ABBR) and full description (abbreviation).
    """
    output_lower = output.lower()
    # Check exact label match
    if label.lower() in output_lower:
        return True
    # Check description match
    if LABEL_DESCRIPTIONS[label] in output_lower:
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=31)
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=500)  # Full test set
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_fsq", action="store_true", default=False)
    return parser.parse_args()


def compute_confidence_interval(correct, total, confidence=0.95):
    """Compute Wilson score confidence interval for accuracy."""
    if total == 0:
        return 0.0, 0.0

    p = correct / total
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    return max(0, center - margin), min(1, center + margin)


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TREC Question Classification Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {', '.join(TREC_LABELS)}")
    print("")
    print("Success Criteria:")
    print("  - Random baseline: ~16.7% (1/6)")
    print("  - Accuracy > 30%: Bridge transmits info")
    print("  - Accuracy > 50%: Bridge is working well")
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

    # Load TREC test set
    ds = load_dataset("SetFit/TREC-QC", split="test")
    num_samples = min(args.num_samples, len(ds))

    # Map label IDs to names
    LABEL_MAP = {0: "ABBR", 1: "ENTY", 2: "DESC", 3: "HUM", 4: "LOC", 5: "NUM"}

    print(f"\nEvaluating {num_samples} samples...")
    print("-" * 60)

    correct = 0
    total = 0
    results = []

    # Track per-class accuracy
    class_correct = {l: 0 for l in TREC_LABELS}
    class_total = {l: 0 for l in TREC_LABELS}

    # Track confusion matrix
    confusion = {true_label: {pred_label: 0 for pred_label in TREC_LABELS} for true_label in TREC_LABELS}

    for i in tqdm(range(num_samples), desc="Evaluating"):
        item = ds[i]
        question = item['text']
        label_id = item['label_coarse']
        label = LABEL_MAP[label_id]

        # Source prompt (same format as text baseline for fair comparison)
        # Include all class options in the prompt
        src_input = f"Question: {question}\nCategory (ABBR, ENTY, DESC, HUM, LOC, or NUM):"

        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            # Target: [Primer] + [Soft Tokens] -> Generate
            primer = "Category:"
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
        for candidate in TREC_LABELS:
            if check_label_match(candidate, output):
                pred = candidate
                break

        is_correct = check_label_match(label, output)

        # Update confusion matrix
        confusion[label][pred] += 1

        if is_correct:
            correct += 1
            class_correct[label] += 1
        total += 1
        class_total[label] += 1

        results.append({
            "idx": i,
            "question": question[:80] + "..." if len(question) > 80 else question,
            "label": label,
            "prediction": pred,
            "output": output[:50],
            "correct": is_correct
        })

        # Print first 12 samples (2 per class ideally)
        if i < 12:
            status = "CORRECT" if is_correct else "wrong"
            print(f"\n[{i}] {status}")
            print(f"  Question: {question[:60]}...")
            print(f"  GT: {label} | Pred: {output[:30]}")

    # Compute overall statistics
    accuracy = 100 * correct / total
    lower_ci, upper_ci = compute_confidence_interval(correct, total)

    print("\n" + "=" * 60)
    print("TREC EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nOVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  95% CI: [{100*lower_ci:.1f}%, {100*upper_ci:.1f}%]")

    print(f"\nPer-class breakdown:")
    for label in TREC_LABELS:
        if class_total[label] > 0:
            class_acc = 100 * class_correct[label] / class_total[label]
            # Compute per-class CI
            cl_lower, cl_upper = compute_confidence_interval(class_correct[label], class_total[label])
            print(f"  {label:4}: {class_correct[label]:3}/{class_total[label]:3} ({class_acc:.1f}%) CI: [{100*cl_lower:.1f}%, {100*cl_upper:.1f}%]")

    # Print confusion matrix (top 3 confusions for each class)
    print(f"\nTop confusions (true -> predicted):")
    for true_label in TREC_LABELS:
        if class_total[true_label] == 0:
            continue
        # Get top confusions (excluding correct predictions)
        confusions_list = [(pred, count) for pred, count in confusion[true_label].items()
                          if pred != true_label and count > 0]
        confusions_list.sort(key=lambda x: x[1], reverse=True)
        if confusions_list:
            top_3 = confusions_list[:3]
            conf_str = ", ".join([f"{pred}({cnt})" for pred, cnt in top_3])
            print(f"  {true_label}: {conf_str}")

    # Interpretation
    print("")
    if accuracy < 20:
        print("=" * 60)
        print("FAILURE: Accuracy near random (~16.7%)")
        print("The bridge is NOT transmitting meaningful information.")
        print("Possible causes:")
        print("  - Bridge architecture issue")
        print("  - Training didn't converge")
        print("  - Task may require more capacity")
        print("=" * 60)
    elif accuracy < 30:
        print("=" * 60)
        print("PARTIAL SUCCESS: Some information transfers")
        print("Better than random but needs improvement.")
        print("=" * 60)
    elif accuracy < 50:
        print("=" * 60)
        print("SUCCESS: Bridge transmits meaningful info!")
        print("Decent performance on 6-class classification.")
        print("=" * 60)
    elif accuracy < 70:
        print("=" * 60)
        print("EXCELLENT: Bridge works well!")
        print("Strong 6-class classification performance.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("OUTSTANDING: Bridge works very well!")
        print("Near-optimal 6-class transfer.")
        print("=" * 60)

    # Save results
    output_path = os.path.join(args.output_dir, "eval_trec_results.json")
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confidence_interval_95": {
            "lower": 100 * lower_ci,
            "upper": 100 * upper_ci
        },
        "per_class": {
            label: {
                "correct": class_correct[label],
                "total": class_total[label],
                "accuracy": 100 * class_correct[label] / max(class_total[label], 1),
                "confidence_interval_95": {
                    "lower": 100 * compute_confidence_interval(class_correct[label], class_total[label])[0],
                    "upper": 100 * compute_confidence_interval(class_correct[label], class_total[label])[1]
                }
            }
            for label in TREC_LABELS
        },
        "confusion_matrix": confusion,
        "samples": results[:20]  # Save first 20 for inspection
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
