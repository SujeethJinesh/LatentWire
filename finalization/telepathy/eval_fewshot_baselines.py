#!/usr/bin/env python3
"""
Few-shot baseline evaluation for Llama and Mistral.

Addresses reviewer concern: "Zero-shot baselines may be artificially weak"

Usage:
    python eval_fewshot_baselines.py --dataset sst2 --shots 5
"""

import argparse
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
    configs = {
        "sst2": {
            "hf_name": ("glue", "sst2"),
            "text_field": "sentence",
            "label_field": "label",
            "label_map": {0: "negative", 1: "positive"},
            "train_split": "train",
            "eval_split": "validation",
            "task_prompt": "Classify the sentiment as positive or negative.",
        },
        "agnews": {
            "hf_name": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            "train_split": "train",
            "eval_split": "test",
            "task_prompt": "Classify the topic as World, Sports, Business, or Sci/Tech.",
        },
        "trec": {
            "hf_name": ("trec",),
            "text_field": "text",
            "label_field": "coarse_label",
            "label_map": {0: "ABBR", 1: "ENTY", 2: "DESC", 3: "HUM", 4: "LOC", 5: "NUM"},
            "train_split": "train",
            "eval_split": "test",
            "task_prompt": "Classify the question type as ABBR, ENTY, DESC, HUM, LOC, or NUM.",
        },
    }
    return configs[dataset_name]


def build_fewshot_prompt(examples, test_text, config, include_cot=False):
    """Build a few-shot prompt with examples."""
    label_map = config["label_map"]
    task = config["task_prompt"]

    prompt = f"{task}\n\n"

    # Add examples
    for ex in examples:
        text = ex[config["text_field"]]
        label = label_map[ex[config["label_field"]]]
        prompt += f"Text: {text}\nLabel: {label}\n\n"

    # Add test example
    prompt += f"Text: {test_text}\nLabel:"
    return prompt


def sample_fewshot_examples(train_ds, config, num_shots, seed):
    """Sample balanced few-shot examples."""
    random.seed(seed)
    label_map = config["label_map"]
    num_classes = len(label_map)
    shots_per_class = num_shots // num_classes

    examples = []
    for label_id in label_map.keys():
        class_examples = [ex for ex in train_ds if ex[config["label_field"]] == label_id]
        sampled = random.sample(class_examples, min(shots_per_class, len(class_examples)))
        examples.extend(sampled)

    random.shuffle(examples)
    return examples[:num_shots]


def evaluate_model(model, tokenizer, eval_ds, config, fewshot_examples, device, max_samples=200):
    """Evaluate a model with few-shot prompting."""
    label_map = config["label_map"]
    label_names = list(label_map.values())

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(tqdm(eval_ds, desc="Evaluating", total=min(max_samples, len(eval_ds)))):
        if i >= max_samples:
            break

        text = item[config["text_field"]]
        true_label = label_map[item[config["label_field"]]]

        prompt = build_fewshot_prompt(fewshot_examples, text, config)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        # Check if any label appears in response
        pred_label = None
        for label in label_names:
            if label.lower() in response:
                pred_label = label
                break

        is_correct = pred_label and pred_label.lower() == true_label.lower()
        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            print(f"[{i}] True: {true_label}, Pred: {response[:30]}, Correct: {is_correct}")

    accuracy = 100 * correct / total
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sst2", "agnews", "trec"], required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="runs/fewshot_baselines")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_dataset_config(args.dataset)

    # Load dataset
    print(f"Loading {args.dataset}...")
    train_ds = load_dataset(*config["hf_name"], split=config["train_split"])
    eval_ds = load_dataset(*config["hf_name"], split=config["eval_split"])

    # Models to evaluate
    models_to_test = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral"),
    ]

    all_results = {}

    for model_id, model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} ({args.shots}-shot)")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        seed_results = []
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            fewshot_examples = sample_fewshot_examples(train_ds, config, args.shots, seed)
            accuracy, correct, total = evaluate_model(
                model, tokenizer, eval_ds, config, fewshot_examples, device, args.max_samples
            )
            print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
            seed_results.append(accuracy)

        mean_acc = sum(seed_results) / len(seed_results)
        std_acc = (sum((x - mean_acc) ** 2 for x in seed_results) / len(seed_results)) ** 0.5

        all_results[model_name] = {
            "mean": mean_acc,
            "std": std_acc,
            "seeds": seed_results,
        }

        print(f"\n{model_name} {args.shots}-shot: {mean_acc:.1f}% ± {std_acc:.1f}%")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/fewshot_{args.dataset}_{args.shots}shot.json"

    results = {
        "experiment": f"fewshot_baseline_{args.dataset}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "shots": args.shots,
            "seeds": args.seeds,
            "max_samples": args.max_samples,
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Few-shot vs Bridge")
    print("=" * 60)

    bridge_results = {
        "sst2": 96.7,
        "agnews": 90.7,
        "trec": 95.3,
    }

    for model_name, res in all_results.items():
        bridge_acc = bridge_results.get(args.dataset, 0)
        gap = bridge_acc - res["mean"]
        print(f"{model_name} {args.shots}-shot: {res['mean']:.1f}% ± {res['std']:.1f}%")
        print(f"  vs Bridge ({bridge_acc}%): {'+' if gap > 0 else ''}{gap:.1f}pp")


if __name__ == "__main__":
    main()
