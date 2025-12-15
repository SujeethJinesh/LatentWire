#!/usr/bin/env python3
"""
Zero-Shot and Text-Relay Reasoning Baseline Evaluation.

Generates explicit JSON result files for all reasoning baselines:
- Llama 0-shot: Direct Llama inference
- Mistral 0-shot: Direct Mistral inference
- Text-Relay: Llama generates summary -> Mistral classifies

Benchmarks evaluated:
- BoolQ: Yes/No reading comprehension (2-way)
- PIQA: Physical intuition QA (2-way)
- CommonsenseQA: Commonsense reasoning (5-way)

Usage:
    python eval_zeroshot_reasoning.py --output_dir runs/zeroshot_reasoning
    python eval_zeroshot_reasoning.py --include_text_relay  # Also run text-relay
"""

import argparse
import json
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# Reasoning benchmark configurations
BENCHMARK_CONFIGS = {
    "boolq": {
        "hf_path": "google/boolq",
        "hf_config": None,
        "eval_split": "validation",
        "num_choices": 2,
        "labels": ["No", "Yes"],
        "get_text": lambda x: f"Passage: {x['passage'][:500]}\n\nQuestion: {x['question']}",
        "get_label": lambda x: 1 if x["answer"] else 0,
        "prompt_template": "{text}\n\nAnswer with Yes or No:",
        "random_chance": 50.0,
        "description": "Yes/No reading comprehension",
    },
    "piqa": {
        "hf_path": "ybisk/piqa",
        "hf_config": None,
        "eval_split": "validation",
        "num_choices": 2,
        "labels": ["1", "2"],
        "get_text": lambda x: f"Goal: {x['goal']}\n\nSolution 1: {x['sol1']}\nSolution 2: {x['sol2']}",
        "get_label": lambda x: x["label"],
        "prompt_template": "{text}\n\nWhich solution is better? Answer 1 or 2:",
        "random_chance": 50.0,
        "description": "Physical intuition QA",
    },
    "commonsenseqa": {
        "hf_path": "tau/commonsense_qa",
        "hf_config": None,
        "eval_split": "validation",
        "num_choices": 5,
        "labels": ["A", "B", "C", "D", "E"],
        "get_text": lambda x: format_commonsenseqa(x),
        "get_label": lambda x: ["A", "B", "C", "D", "E"].index(x["answerKey"]),
        "prompt_template": "{text}\n\nAnswer:",
        "random_chance": 20.0,
        "description": "Commonsense reasoning (5-way)",
    },
}


def format_commonsenseqa(item):
    """Format CommonsenseQA item with choices."""
    text = f"Question: {item['question']}\n\nChoices:\n"
    for label, choice in zip(item["choices"]["label"], item["choices"]["text"]):
        text += f"{label}. {choice}\n"
    return text


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
    """Evaluate zero-shot accuracy on reasoning benchmark."""
    labels = config["labels"]

    correct = 0
    total = 0
    predictions = []

    for i, item in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)), desc="Evaluating")):
        if i >= max_samples:
            break

        text = config["get_text"](item)
        true_label_id = config["get_label"](item)
        true_label = labels[true_label_id]

        # Format prompt
        prompt = config["prompt_template"].format(text=text)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()

        # Match prediction to label
        pred_label = None
        response_lower = response.lower()

        # Try exact match first
        for label in labels:
            if label.lower() == response_lower or response_lower.startswith(label.lower()):
                pred_label = label
                break

        # Fallback: check if label is in response
        if pred_label is None:
            for label in labels:
                if label.lower() in response_lower:
                    pred_label = label
                    break

        # Check correctness
        is_correct = pred_label is not None and pred_label.lower() == true_label.lower()
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
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
        "sample_predictions": predictions[:10],
    }


def evaluate_text_relay(llama, llama_tok, mistral, mistral_tok, dataset, config, device, max_samples=200):
    """Evaluate text-relay: Llama summarizes -> Mistral classifies."""
    labels = config["labels"]

    correct = 0
    total = 0
    predictions = []

    for i, item in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)), desc="Text-Relay")):
        if i >= max_samples:
            break

        text = config["get_text"](item)
        true_label_id = config["get_label"](item)
        true_label = labels[true_label_id]

        # Step 1: Llama generates a summary/analysis
        llama_prompt = f"Analyze this and provide a brief summary of the key information:\n\n{text}\n\nSummary:"
        llama_inputs = llama_tok(llama_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            llama_outputs = llama.generate(
                **llama_inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=llama_tok.eos_token_id,
            )

        llama_summary = llama_tok.decode(llama_outputs[0][llama_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        llama_summary = llama_summary.strip()[:200]  # Truncate

        # Step 2: Mistral classifies based on Llama's summary
        mistral_prompt = config["prompt_template"].format(text=f"Context: {llama_summary}\n\n{text}")
        mistral_inputs = mistral_tok(mistral_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            mistral_outputs = mistral.generate(
                **mistral_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=mistral_tok.eos_token_id,
            )

        response = mistral_tok.decode(mistral_outputs[0][mistral_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()

        # Match prediction to label
        pred_label = None
        response_lower = response.lower()

        for label in labels:
            if label.lower() == response_lower or response_lower.startswith(label.lower()):
                pred_label = label
                break

        if pred_label is None:
            for label in labels:
                if label.lower() in response_lower:
                    pred_label = label
                    break

        is_correct = pred_label is not None and pred_label.lower() == true_label.lower()
        if is_correct:
            correct += 1
        total += 1

        predictions.append({
            "true_label": true_label,
            "predicted": pred_label,
            "llama_summary": llama_summary[:50],
            "response": response[:50],
            "correct": is_correct,
        })

    accuracy = 100 * correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "sample_predictions": predictions[:10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/zeroshot_reasoning")
    parser.add_argument("--benchmarks", nargs="+", default=["boolq", "piqa", "commonsenseqa"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--include_text_relay", action="store_true", help="Also evaluate text-relay baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Models to evaluate
    models = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    }

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_samples": args.max_samples,
            "benchmarks": args.benchmarks,
        },
        "results": {},
    }

    # Load models
    loaded_models = {}
    for name, model_id in models.items():
        loaded_models[name] = load_model(model_id, device)

    # Evaluate each benchmark
    for benchmark_name in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating {benchmark_name.upper()}")
        print(f"{'='*60}")

        config = BENCHMARK_CONFIGS[benchmark_name]
        print(f"Description: {config['description']}")
        print(f"Random chance: {config['random_chance']}%")

        # Load dataset
        try:
            if config["hf_config"]:
                dataset = load_dataset(config["hf_path"], config["hf_config"], trust_remote_code=True)
            else:
                dataset = load_dataset(config["hf_path"], trust_remote_code=True)

            eval_data = dataset[config["eval_split"]]
            print(f"Eval samples available: {len(eval_data)}")

        except Exception as e:
            print(f"ERROR loading {benchmark_name}: {e}")
            all_results["results"][benchmark_name] = {"error": str(e)}
            continue

        benchmark_results = {
            "random_chance": config["random_chance"],
            "num_choices": config["num_choices"],
            "description": config["description"],
            "models": {},
        }

        # Evaluate each model
        for model_name, (model, tokenizer) in loaded_models.items():
            print(f"\n--- {model_name.upper()} ---")

            results = evaluate_zeroshot(
                model, tokenizer, eval_data, config, device, args.max_samples
            )

            benchmark_results["models"][model_name] = results
            print(f"Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")

        # Evaluate text-relay if requested
        if args.include_text_relay:
            print(f"\n--- TEXT-RELAY (Llama->Mistral) ---")
            llama_model, llama_tokenizer = loaded_models["llama"]
            mistral_model, mistral_tokenizer = loaded_models["mistral"]

            text_relay_results = evaluate_text_relay(
                llama_model, llama_tokenizer,
                mistral_model, mistral_tokenizer,
                eval_data, config, device, args.max_samples
            )

            benchmark_results["models"]["text_relay"] = text_relay_results
            print(f"Accuracy: {text_relay_results['accuracy']:.1f}% ({text_relay_results['correct']}/{text_relay_results['total']})")

        all_results["results"][benchmark_name] = benchmark_results

        # Save per-benchmark results
        benchmark_output = os.path.join(args.output_dir, f"zeroshot_{benchmark_name}.json")
        with open(benchmark_output, "w") as f:
            json.dump({
                "benchmark": benchmark_name,
                "timestamp": datetime.now().isoformat(),
                **benchmark_results,
            }, f, indent=2)
        print(f"Saved: {benchmark_output}")

    # Save combined results
    combined_output = os.path.join(args.output_dir, "zeroshot_reasoning_all.json")
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("REASONING BASELINE SUMMARY")
    print(f"{'='*60}")

    if args.include_text_relay:
        print(f"\n{'Benchmark':<18} {'Llama':<10} {'Mistral':<10} {'Text-Relay':<12} {'Random':<8}")
        print("-" * 68)
    else:
        print(f"\n{'Benchmark':<20} {'Llama':<12} {'Mistral':<12} {'Random':<10}")
        print("-" * 54)

    for benchmark_name, results in all_results["results"].items():
        if "error" in results:
            print(f"{benchmark_name:<20} ERROR")
            continue

        llama_acc = results["models"].get("llama", {}).get("accuracy", 0)
        mistral_acc = results["models"].get("mistral", {}).get("accuracy", 0)
        random = results["random_chance"]

        if args.include_text_relay:
            text_relay_acc = results["models"].get("text_relay", {}).get("accuracy", 0)
            print(f"{benchmark_name:<18} {llama_acc:>6.1f}%   {mistral_acc:>6.1f}%   {text_relay_acc:>6.1f}%     {random:>6.1f}%")
        else:
            print(f"{benchmark_name:<20} {llama_acc:>6.1f}%     {mistral_acc:>6.1f}%     {random:>6.1f}%")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
