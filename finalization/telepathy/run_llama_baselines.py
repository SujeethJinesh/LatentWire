#!/usr/bin/env python3
"""
Llama 3.1 8B Zero-Shot Baseline Evaluations

Runs zero-shot evaluation on SST-2 and AG News using Llama 3.1 8B Instruct.
This provides baseline comparison metrics for the telepathy experiments.

Usage:
    python telepathy/run_llama_baselines.py --datasets sst2 agnews --output_dir runs/llama_baselines
    python telepathy/run_llama_baselines.py --datasets sst2 --eval_samples 500
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import random
import numpy as np
from datetime import datetime


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET CONFIGS
# =============================================================================

DATASET_CONFIGS = {
    "sst2": {
        "hf_name": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "task_prompt": "Is the sentiment positive or negative?",
        "random_chance": 50.0,
    },
    "agnews": {
        "hf_name": ("ag_news",),
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Classify the topic: World, Sports, Business, or Sci/Tech.",
        "random_chance": 25.0,
    },
    "trec": {
        "hf_name": ("trec",),
        "text_field": "text",
        "label_field": "coarse_label",
        "label_map": {
            0: "abbreviation",
            1: "entity",
            2: "description",
            3: "human",
            4: "location",
            5: "number"
        },
        "num_classes": 6,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Classify the question type: abbreviation, entity, description, human, location, or number.",
        "random_chance": 16.7,
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(dataset_name, split="train", max_samples=None):
    """Load dataset."""
    config = DATASET_CONFIGS[dataset_name]
    if len(config["hf_name"]) == 2:
        ds = load_dataset(config["hf_name"][0], config["hf_name"][1], split=split)
    else:
        ds = load_dataset(config["hf_name"][0], split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def get_label_tokens(tokenizer, dataset_name):
    """Get token IDs for each label."""
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = {}
    for idx, label in config["label_map"].items():
        # Get first token of the label
        tokens = tokenizer.encode(label, add_special_tokens=False)
        label_tokens[idx] = tokens[0]
    return label_tokens


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def eval_llama_zeroshot(model, tokenizer, eval_ds, dataset_name, device):
    """Evaluate Llama zero-shot baseline.

    This uses the same prompt format and evaluation logic as the unified comparison script.
    """
    config = DATASET_CONFIGS[dataset_name]
    label_tokens = get_label_tokens(tokenizer, dataset_name)

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for item in tqdm(eval_ds, desc=f"Eval Llama Zero-shot on {dataset_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            # Same prompt format as unified_comparison.py
            prompt = f"Text: {text[:256]}\n\n{config['task_prompt']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[0, -1]

            # Get logits for each label token
            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            pred = label_logits.argmax().item()

            # Store prediction
            predictions.append({
                "text": text,
                "true_label": label,
                "true_label_name": config["label_map"][label],
                "predicted_label": pred,
                "predicted_label_name": config["label_map"][pred],
                "correct": pred == label,
                "label_logits": {config["label_map"][i]: label_logits[i].item() for i in range(len(label_tokens))}
            })

            if pred == label:
                correct += 1
            total += 1

    return {
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Llama 3.1 8B zero-shot baselines")
    parser.add_argument("--datasets", nargs="+", default=["sst2", "agnews"],
                       choices=["sst2", "agnews", "trec"],
                       help="Datasets to evaluate on")
    parser.add_argument("--output_dir", default="runs/llama_baselines",
                       help="Output directory for results")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Llama model to use")
    parser.add_argument("--eval_samples", type=int, default=None,
                       help="Number of samples to evaluate (None = full dataset)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save individual predictions to JSON")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("LLAMA 3.1 8B ZERO-SHOT BASELINE EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Eval samples: {args.eval_samples if args.eval_samples else 'full dataset'}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded successfully")

    # Results container
    all_results = {
        "meta": {
            "timestamp": timestamp,
            "model": args.model,
            "datasets": args.datasets,
            "eval_samples": args.eval_samples,
            "seed": args.seed,
        },
        "results": {},
    }

    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")

        config = DATASET_CONFIGS[dataset_name]

        # Load data
        print(f"Loading {dataset_name} dataset...")
        eval_ds = load_data(dataset_name, config["eval_split"], max_samples=args.eval_samples)
        print(f"Loaded {len(eval_ds)} evaluation samples")

        # Evaluate
        print(f"Evaluating Llama zero-shot on {dataset_name}...")
        results = eval_llama_zeroshot(model, tokenizer, eval_ds, dataset_name, device)

        # Add metadata
        results["random_chance"] = config["random_chance"]
        results["dataset_config"] = {
            "num_classes": config["num_classes"],
            "label_map": config["label_map"],
            "task_prompt": config["task_prompt"],
        }

        all_results["results"][dataset_name] = results

        # Print summary
        print(f"\n{'='*50}")
        print(f"RESULTS: {dataset_name.upper()}")
        print(f"{'='*50}")
        print(f"  Accuracy:       {results['accuracy']:.2f}%")
        print(f"  Correct:        {results['correct']}/{results['total']}")
        print(f"  Random chance:  {config['random_chance']:.1f}%")
        print(f"{'='*50}")

        # Save per-dataset predictions if requested
        if args.save_predictions:
            pred_path = f"{args.output_dir}/{dataset_name}_predictions_{timestamp}.json"
            with open(pred_path, "w") as f:
                json.dump({
                    "meta": all_results["meta"],
                    "dataset": dataset_name,
                    "results": results
                }, f, indent=2)
            print(f"  Predictions saved to: {pred_path}")

    # Save summary results
    summary_path = f"{args.output_dir}/llama_baseline_results_{timestamp}.json"

    # Remove predictions from summary to keep file small
    summary_results = {
        "meta": all_results["meta"],
        "results": {}
    }
    for dataset_name, results in all_results["results"].items():
        summary_results["results"][dataset_name] = {
            k: v for k, v in results.items() if k != "predictions"
        }
        summary_results["results"][dataset_name]["num_predictions"] = len(results.get("predictions", []))

    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Summary results saved to: {summary_path}")

    # Final summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Dataset':<15} {'Accuracy':<12} {'Correct':<12} {'Random Chance':<15}")
    print("-"*70)
    for dataset_name in args.datasets:
        results = all_results["results"][dataset_name]
        config = DATASET_CONFIGS[dataset_name]
        print(f"{dataset_name.upper():<15} {results['accuracy']:<12.2f} "
              f"{results['correct']}/{results['total']:<8} {config['random_chance']:<15.1f}")
    print("="*70)


if __name__ == "__main__":
    main()
