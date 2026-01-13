#!/usr/bin/env python
# telepathy/eval_telepathy.py
"""
Unified Telepathy Evaluation Script

Evaluates trained bridge checkpoints on classification tasks.
Supports multiple datasets: SST-2 (2-class), AG News (4-class), TREC (6-class), Banking77 (77-class)

Usage:
    python telepathy/eval_telepathy.py --checkpoint runs/sst2/bridge_sst2.pt --dataset sst2
    python telepathy/eval_telepathy.py --checkpoint runs/agnews/bridge_agnews.pt --dataset agnews
    python telepathy/eval_telepathy.py --checkpoint runs/trec/bridge_trec.pt --dataset trec --num_samples 500

Key Features:
- Per-class accuracy breakdown
- Confidence intervals (Wilson score)
- Confusion matrix for multi-class
- JSON results export
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import numpy as np
from scipy import stats

from latentwire import LatentBridge


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    "sst2": {
        "load_args": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
        "labels": ["negative", "positive"],
        "num_classes": 2,
        "eval_split": "validation",
        "max_length": 128,
        "prompt_template": "Review: {text}\nSentiment (positive or negative):",
        "primer": "Sentiment:",
        "random_baseline": 50.0,
    },
    "agnews": {
        "load_args": ("ag_news",),
        "text_field": "text",
        "label_field": "label",
        "labels": ["world", "sports", "business", "science"],
        "num_classes": 4,
        "eval_split": "test",
        "max_length": 256,
        "prompt_template": "Article: {text}\nTopic (world, sports, business, or science):",
        "primer": "Topic:",
        "random_baseline": 25.0,
        "label_synonyms": {
            "science": ["science", "technology", "tech", "sci/tech", "scitech"],
        },
    },
    "trec": {
        "load_args": ("CogComp/trec",),
        "text_field": "text",
        "label_field": "coarse_label",
        "labels": ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
        "label_descriptions": {
            "ABBR": "abbreviation",
            "DESC": "description",
            "ENTY": "entity",
            "HUM": "human",
            "LOC": "location",
            "NUM": "numeric",
        },
        "num_classes": 6,
        "eval_split": "test",
        "max_length": 128,
        "prompt_template": "Question: {text}\nCategory (ABBR, ENTY, DESC, HUM, LOC, or NUM):",
        "primer": "Category:",
        "random_baseline": 16.7,
    },
    "banking77": {
        "load_args": ("banking77",),
        "text_field": "text",
        "label_field": "label",
        "labels": None,  # Will be populated from dataset
        "num_classes": 77,
        "eval_split": "test",
        "max_length": 128,
        "prompt_template": "Query: {text}\nIntent:",
        "primer": "Intent:",
        "random_baseline": 1.3,
    },
}


def check_label_match(label, output, config):
    """Check if label matches output, with permissive matching."""
    output_lower = output.lower()

    # Check synonyms if defined
    if "label_synonyms" in config and label in config["label_synonyms"]:
        return any(syn in output_lower for syn in config["label_synonyms"][label])

    # Check label descriptions (e.g., TREC)
    if "label_descriptions" in config and label in config["label_descriptions"]:
        if config["label_descriptions"][label] in output_lower:
            return True

    return label.lower() in output_lower


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


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Telepathy Evaluation")

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to bridge checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["sst2", "agnews", "trec", "banking77"],
                       help="Dataset to evaluate on")

    # Model configuration
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=31)

    # Bridge architecture (must match training)
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)

    # Evaluation settings
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None = full dataset)")
    parser.add_argument("--output_dir", default=".",
                       help="Directory to save results")

    # Other settings
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_fsq", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"TELEPATHY EVALUATION: {args.dataset.upper()}")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {config['num_classes']}")
    print("")
    print("Success Criteria:")
    print(f"  - Random baseline: {config['random_baseline']:.1f}%")
    print("  - Accuracy > 2x random: Bridge transmits info")
    print("  - Accuracy > 3x random: Bridge is working well")
    print("=" * 60)
    print("")

    # Load models
    print("Loading models...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": device}
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
    bridge = LatentBridge(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    bridge.load_state_dict(checkpoint)
    bridge.to(device)
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load dataset
    if len(config["load_args"]) == 2:
        ds = load_dataset(config["load_args"][0], config["load_args"][1],
                         split=config["eval_split"], trust_remote_code=True)
    else:
        ds = load_dataset(config["load_args"][0],
                         split=config["eval_split"], trust_remote_code=True)

    # Get labels from dataset if not predefined
    labels = config["labels"]
    if labels is None:
        labels = ds.features[config["label_field"]].names
        config["labels"] = labels

    num_samples = min(args.num_samples, len(ds)) if args.num_samples else len(ds)

    print(f"\nEvaluating {num_samples} samples...")
    print("-" * 60)

    correct = 0
    total = 0
    results = []

    # Per-class tracking
    class_correct = {l: 0 for l in labels}
    class_total = {l: 0 for l in labels}

    # Confusion matrix
    confusion = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}

    for i in tqdm(range(num_samples), desc="Evaluating"):
        item = ds[i]
        text = item[config["text_field"]]
        label_idx = item[config["label_field"]]
        label = labels[label_idx]

        src_input = config["prompt_template"].format(text=text[:256])

        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True,
                            max_length=config["max_length"]).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()
            soft_tokens, _, _, _ = bridge(src_h, src_enc.attention_mask)

            # Target: [Primer] + [Soft Tokens] -> Generate
            primer = config["primer"]
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

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
        for candidate in labels:
            if check_label_match(candidate, output, config):
                pred = candidate
                break

        is_correct = check_label_match(label, output, config)

        # Update confusion matrix
        if pred in confusion[label]:
            confusion[label][pred] += 1
        else:
            confusion[label]["unknown"] = confusion[label].get("unknown", 0) + 1

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

        # Print first few samples
        if i < 10:
            status = "CORRECT" if is_correct else "wrong"
            print(f"\n[{i}] {status}")
            print(f"  Text: {text[:60]}...")
            print(f"  GT: {label} | Pred: {output[:30]}")

    # Compute overall statistics
    accuracy = 100 * correct / total
    lower_ci, upper_ci = compute_confidence_interval(correct, total)

    print("\n" + "=" * 60)
    print(f"{args.dataset.upper()} EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nOVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  95% CI: [{100*lower_ci:.1f}%, {100*upper_ci:.1f}%]")
    print(f"  Random baseline: {config['random_baseline']:.1f}%")

    print(f"\nPer-class breakdown:")
    for label in labels:
        if class_total[label] > 0:
            class_acc = 100 * class_correct[label] / class_total[label]
            cl_lower, cl_upper = compute_confidence_interval(class_correct[label], class_total[label])
            print(f"  {label:15}: {class_correct[label]:4}/{class_total[label]:4} ({class_acc:.1f}%) CI: [{100*cl_lower:.1f}%, {100*cl_upper:.1f}%]")

    # Print top confusions for multi-class
    if config["num_classes"] > 2:
        print(f"\nTop confusions (true -> predicted):")
        for true_label in labels:
            if class_total[true_label] == 0:
                continue
            confusions_list = [(pred, count) for pred, count in confusion[true_label].items()
                              if pred != true_label and count > 0]
            confusions_list.sort(key=lambda x: x[1], reverse=True)
            if confusions_list:
                top_3 = confusions_list[:3]
                conf_str = ", ".join([f"{pred}({cnt})" for pred, cnt in top_3])
                print(f"  {true_label}: {conf_str}")

    # Interpretation
    print("")
    ratio = accuracy / config['random_baseline']
    if ratio < 1.5:
        print("=" * 60)
        print("FAILURE: Accuracy near random")
        print("The bridge is NOT transmitting meaningful information.")
        print("=" * 60)
    elif ratio < 2.5:
        print("=" * 60)
        print("PARTIAL SUCCESS: Some information transfers")
        print("Better than random but room for improvement.")
        print("=" * 60)
    elif ratio < 4.0:
        print("=" * 60)
        print("SUCCESS: Bridge transmits meaningful info!")
        print(f"Accuracy is {ratio:.1f}x random baseline.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXCELLENT: Bridge works very well!")
        print(f"Accuracy is {ratio:.1f}x random baseline.")
        print("=" * 60)

    # Save results
    output_path = os.path.join(args.output_dir, f"eval_{args.dataset}_results.json")
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_samples": num_samples,
        "num_classes": config["num_classes"],
        "random_baseline": config["random_baseline"],
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
            for label in labels
        },
        "confusion_matrix": confusion,
        "samples": results[:20]  # Save first 20 for inspection
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
