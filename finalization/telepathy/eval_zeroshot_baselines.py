#!/usr/bin/env python3
"""
Zero-Shot Baseline Evaluation for Llama and Mistral.

Generates explicit JSON result files for all zero-shot baselines.
This addresses the audit finding that zero-shot results lacked proper JSON files.

Datasets evaluated:
- SST-2 (GLUE): Binary sentiment classification
- AG News: 4-class topic classification
- TREC: 6-class question type classification
- Banking77: 77-class intent classification

Usage:
    python eval_zeroshot_baselines.py --output_dir runs/zeroshot_baselines
"""

import argparse
import json
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# Dataset configurations
DATASET_CONFIGS = {
    "sst2": {
        "hf_path": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
        "eval_split": "validation",
        "labels": ["negative", "positive"],
        "prompt_template": "Classify the sentiment of this review as positive or negative.\n\nReview: {text}\n\nSentiment:",
        "random_chance": 50.0,
    },
    "agnews": {
        "hf_path": ("ag_news",),
        "text_field": "text",
        "label_field": "label",
        "eval_split": "test",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "prompt_template": "Classify the topic of this article as World, Sports, Business, or Sci/Tech.\n\nArticle: {text}\n\nTopic:",
        "random_chance": 25.0,
    },
    "trec": {
        "hf_path": ("trec",),
        "text_field": "text",
        "label_field": "coarse_label",
        "eval_split": "test",
        "labels": ["abbreviation", "entity", "description", "human", "location", "number"],
        "label_descriptions": {
            "abbreviation": "Questions about abbreviations or expressions (ABBR)",
            "entity": "Questions about entities like animals, colors, foods (ENTY)",
            "description": "Questions asking for definitions or descriptions (DESC)",
            "human": "Questions about humans or groups (HUM)",
            "location": "Questions about locations or places (LOC)",
            "number": "Questions about numbers, dates, or quantities (NUM)",
        },
        "prompt_template": "What type of answer does this question expect?\n\nTypes:\n- abbreviation: Questions about abbreviations\n- entity: Questions about things (animals, colors, etc.)\n- description: Questions asking for definitions\n- human: Questions about people\n- location: Questions about places\n- number: Questions about quantities or dates\n\nQuestion: {text}\n\nAnswer type:",
        "random_chance": 16.7,
    },
    "banking77": {
        "hf_path": ("PolyAI/banking77",),
        "text_field": "text",
        "label_field": "label",
        "eval_split": "test",
        "labels": None,  # 77 labels, loaded dynamically
        "prompt_template": "Classify this banking query intent.\n\nQuery: {text}\n\nIntent:",
        "random_chance": 1.3,
    },
}


def load_model(model_id, device):
    """Load model and tokenizer."""
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def evaluate_zeroshot(model, tokenizer, dataset, config, device, max_samples=200):
    """Evaluate zero-shot classification accuracy."""
    labels = config["labels"]

    correct = 0
    total = 0
    predictions = []

    for i, item in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)), desc="Evaluating")):
        if i >= max_samples:
            break

        text = item[config["text_field"]]
        true_label_id = item[config["label_field"]]
        true_label = labels[true_label_id] if labels else str(true_label_id)

        # Format prompt
        prompt = config["prompt_template"].format(text=text[:500])  # Truncate long texts

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        # Match prediction to label
        pred_label = None
        for label in labels:
            if label.lower() in response:
                pred_label = label
                break

        # Check correctness
        is_correct = pred_label is not None and pred_label.lower() == true_label.lower()
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "true_label": true_label,
            "predicted": pred_label,
            "response": response[:50],
            "correct": is_correct,
        })

    accuracy = 100 * correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "sample_predictions": predictions[:10],  # Save first 10 for inspection
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/zeroshot_baselines")
    parser.add_argument("--datasets", nargs="+", default=["sst2", "agnews", "trec", "banking77"])
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models to evaluate
    models = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    }

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_samples": args.max_samples,
            "datasets": args.datasets,
        },
        "results": {},
    }

    # Load models
    loaded_models = {}
    for name, model_id in models.items():
        loaded_models[name] = load_model(model_id, device)

    # Evaluate each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating {dataset_name.upper()}")
        print(f"{'='*60}")

        config = DATASET_CONFIGS[dataset_name]

        # Load dataset
        if len(config["hf_path"]) == 2:
            dataset = load_dataset(config["hf_path"][0], config["hf_path"][1], trust_remote_code=True)
        else:
            dataset = load_dataset(config["hf_path"][0], trust_remote_code=True)

        eval_data = dataset[config["eval_split"]]

        # Get labels for banking77
        if config["labels"] is None:
            # Banking77 has 77 intent labels
            unique_labels = sorted(set(eval_data[config["label_field"]]))
            config["labels"] = [f"intent_{i}" for i in unique_labels]

        dataset_results = {
            "random_chance": config["random_chance"],
            "num_classes": len(config["labels"]),
            "models": {},
        }

        # Evaluate each model
        for model_name, (model, tokenizer) in loaded_models.items():
            print(f"\n--- {model_name.upper()} ---")

            results = evaluate_zeroshot(
                model, tokenizer, eval_data, config, device, args.max_samples
            )

            dataset_results["models"][model_name] = results
            print(f"Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")

        all_results["results"][dataset_name] = dataset_results

        # Save per-dataset results
        dataset_output = os.path.join(args.output_dir, f"zeroshot_{dataset_name}.json")
        with open(dataset_output, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "timestamp": datetime.now().isoformat(),
                **dataset_results,
            }, f, indent=2)
        print(f"Saved: {dataset_output}")

    # Save combined results
    combined_output = os.path.join(args.output_dir, "zeroshot_all_baselines.json")
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("ZERO-SHOT BASELINE SUMMARY")
    print(f"{'='*60}")

    for dataset_name, results in all_results["results"].items():
        print(f"\n{dataset_name.upper()} (random: {results['random_chance']:.1f}%):")
        for model_name, model_results in results["models"].items():
            print(f"  {model_name}: {model_results['accuracy']:.1f}%")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
