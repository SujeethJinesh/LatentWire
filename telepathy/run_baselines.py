#!/usr/bin/env python
# telepathy/run_baselines.py
"""
Unified Baselines Script

Runs various baseline methods for comparison with the telepathy bridge.
Supports: zero-shot, few-shot, LoRA fine-tuning, DoRA fine-tuning, prompt tuning, text relay, ensemble, and ridge regression.

Usage:
    python telepathy/run_baselines.py --baseline zeroshot --dataset sst2
    python telepathy/run_baselines.py --baseline fewshot --dataset agnews --shots 5
    python telepathy/run_baselines.py --baseline lora --dataset trec --rank 8
    python telepathy/run_baselines.py --baseline dora --dataset trec --rank 8
    python telepathy/run_baselines.py --baseline prompt_tuning --dataset sst2 --soft_tokens 8
    python telepathy/run_baselines.py --baseline text_relay --dataset sst2
    python telepathy/run_baselines.py --baseline ensemble --dataset sst2 --alphas 0.3 0.5 0.7
    python telepathy/run_baselines.py --baseline ridge_regression --dataset sst2 --lambda_values 0.1 1.0 10.0 100.0

Baseline Types:
- zeroshot: Direct prompting without examples (Llama and Mistral)
- fewshot: In-context learning with k examples
- lora: LoRA fine-tuning on Mistral
- dora: DoRA fine-tuning on Mistral (Weight-Decomposed Low-Rank Adaptation)
- prompt_tuning: Learnable soft prompts on Mistral (no sender model)
- text_relay: Fair cross-model text communication (Llama generates hint, Mistral classifies)
- ensemble: Ensemble of Llama and Mistral predictions (tests if bridge is "super-additive")
- ridge_regression: LatentMAS-style linear alignment from Llama hidden states to Mistral embeddings
"""
import os
import gc
import json
import random
import time
import argparse
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA/DoRA baselines unavailable.")

try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed. Ridge regression baseline unavailable.")


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    "sst2": {
        "load_args": ("glue", "sst2"),
        "text_field": "sentence",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "task_prompt": "Classify the sentiment as positive or negative.",
        "prompt_template": "Text: {text}\n\nIs this positive or negative?\nAnswer:",
        "random_baseline": 50.0,
    },
    "agnews": {
        "load_args": ("ag_news",),
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Classify the topic as World, Sports, Business, or Sci/Tech.",
        "prompt_template": "Text: {text}\n\n{task_prompt}\nAnswer:",
        "random_baseline": 25.0,
    },
    "trec": {
        "load_args": ("CogComp/trec",),
        "text_field": "text",
        "label_field": "coarse_label",
        "label_map": {0: "ABBR", 1: "DESC", 2: "ENTY", 3: "HUM", 4: "LOC", 5: "NUM"},
        "num_classes": 6,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Classify the question type as ABBR, ENTY, DESC, HUM, LOC, or NUM.",
        "prompt_template": "Question: {text}\n\n{task_prompt}\nAnswer:",
        "random_baseline": 16.7,
    },
    # =========================================================================
    # REASONING BENCHMARKS
    # =========================================================================
    "arc_easy": {
        "load_args": ("allenai/ai2_arc", "ARC-Easy"),
        "text_field": "question",
        "label_field": "answerKey",
        # ARC dataset uses both letter (A,B,C,D,E) and numeric (1,2,3,4,5) answer keys
        "label_map": {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "1": "A", "2": "B", "3": "C", "4": "D", "5": "E"},
        "label_from_key": True,  # Labels are string keys, not int indices
        "num_classes": 5,  # Some questions have 5 choices
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Answer A, B, C, D, or E.",
        "prompt_template": "Question: {text}\n\n{task_prompt}\nAnswer:",
        "random_baseline": 25.0,
    },
    "winogrande": {
        "load_args": ("allenai/winogrande", "winogrande_xl"),
        "text_field": "sentence",
        "label_field": "answer",
        "label_map": {"1": "1", "2": "2"},
        "label_from_key": True,
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "task_prompt": "Which option fits best? Answer 1 or 2.",
        "prompt_template": "Sentence: {text}\n\n{task_prompt}\nAnswer:",
        "random_baseline": 50.0,
    },
    "hellaswag": {
        "load_args": ("Rowan/hellaswag",),
        "text_field": "ctx",
        "label_field": "label",
        "label_map": {0: "0", 1: "1", 2: "2", 3: "3", "0": "0", "1": "1", "2": "2", "3": "3"},
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "validation",
        "task_prompt": "Which continuation is most likely? Answer 0, 1, 2, or 3.",
        "prompt_template": "Context: {text}\n\n{task_prompt}\nAnswer:",
        "random_baseline": 25.0,
    },
    "boolq": {
        "load_args": ("google/boolq",),
        "text_field": "question",
        "label_field": "answer",
        "label_map": {False: "No", True: "Yes", 0: "No", 1: "Yes"},
        "num_classes": 2,
        "train_split": "train",
        "eval_split": "validation",
        "task_prompt": "Answer Yes or No.",
        "prompt_template": "Question: {text}\n\n{task_prompt}\nAnswer:",
        "random_baseline": 50.0,
    },
}


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# ZERO-SHOT BASELINE
# =============================================================================

def run_zeroshot_baseline(args, config, device):
    """Run zero-shot evaluation on Llama and Mistral."""
    print("\n" + "=" * 70)
    print(f"ZERO-SHOT BASELINE: {args.dataset.upper()}")
    print("=" * 70)

    # Load evaluation data
    if len(config["load_args"]) == 2:
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    models_to_test = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral"),
    ]

    all_results = {}

    for model_id, model_name in models_to_test:
        print(f"\nEvaluating {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        # Get unique canonical labels (handles datasets with multiple keys mapping to same label)
        canonical_labels = sorted(set(config["label_map"].values()))
        label_tokens = {}
        for label_text in canonical_labels:
            tokens = tokenizer.encode(label_text, add_special_tokens=False)
            label_tokens[label_text] = tokens[0]

        correct = 0
        total = 0

        for item in tqdm(eval_ds, desc=f"Zero-shot {model_name}"):
            text = item[config["text_field"]]
            raw_label = item[config["label_field"]]
            # Normalize raw label to canonical form (e.g., "1" -> "A" for ARC)
            true_label = config["label_map"].get(raw_label, raw_label)

            prompt = config["prompt_template"].format(
                text=text[:256],
                task_prompt=config["task_prompt"]
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]

            # Get logits for each canonical label token
            label_logits = torch.stack([logits[label_tokens[lbl]] for lbl in canonical_labels])
            pred = label_logits.argmax().item()
            pred_label = canonical_labels[pred]

            if pred_label == true_label:
                correct += 1
            total += 1

        accuracy = 100 * correct / total
        all_results[model_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        print(f"{model_name}: {accuracy:.1f}% ({correct}/{total})")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return all_results


# =============================================================================
# FEW-SHOT BASELINE
# =============================================================================

def build_fewshot_prompt(examples, test_text, config):
    """Build a few-shot prompt with examples."""
    label_map = config["label_map"]
    task = config["task_prompt"]

    prompt = f"{task}\n\n"

    for ex in examples:
        text = ex[config["text_field"]]
        label = label_map[ex[config["label_field"]]]
        prompt += f"Text: {text[:200]}\nLabel: {label}\n\n"

    prompt += f"Text: {test_text[:200]}\nLabel:"
    return prompt


def sample_fewshot_examples(train_ds, config, num_shots, seed):
    """Sample balanced few-shot examples."""
    random.seed(seed)
    label_map = config["label_map"]
    num_classes = len(label_map)
    shots_per_class = max(1, num_shots // num_classes)

    examples = []
    for label_id in label_map.keys():
        class_examples = [ex for ex in train_ds if ex[config["label_field"]] == label_id]
        sampled = random.sample(class_examples, min(shots_per_class, len(class_examples)))
        examples.extend(sampled)

    random.shuffle(examples)
    return examples[:num_shots]


def run_fewshot_baseline(args, config, device):
    """Run few-shot evaluation."""
    print("\n" + "=" * 70)
    print(f"FEW-SHOT BASELINE ({args.shots}-shot): {args.dataset.upper()}")
    print("=" * 70)

    # Load data
    if len(config["load_args"]) == 2:
        train_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        train_ds = load_dataset(config["load_args"][0],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    models_to_test = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral"),
    ]

    all_results = {}

    for model_id, model_name in models_to_test:
        print(f"\nEvaluating {model_name} ({args.shots}-shot)...")

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        seed_results = []
        for seed in args.seeds:
            set_seed(seed)
            fewshot_examples = sample_fewshot_examples(train_ds, config, args.shots, seed)

            label_names = list(config["label_map"].values())
            correct = 0
            total = 0

            for item in tqdm(eval_ds, desc=f"Seed {seed}", leave=False):
                text = item[config["text_field"]]
                true_label = config["label_map"][item[config["label_field"]]]

                prompt = build_fewshot_prompt(fewshot_examples, text, config)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=10, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                response = response.strip().lower()

                pred_label = None
                for label in label_names:
                    if label.lower() in response:
                        pred_label = label
                        break

                if pred_label and pred_label.lower() == true_label.lower():
                    correct += 1
                total += 1

            accuracy = 100 * correct / total
            seed_results.append(accuracy)
            print(f"  Seed {seed}: {accuracy:.1f}%")

        mean_acc = np.mean(seed_results)
        std_acc = np.std(seed_results)
        all_results[model_name] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "seeds": seed_results,
        }
        print(f"{model_name} {args.shots}-shot: {mean_acc:.1f}% +/- {std_acc:.1f}%")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return all_results


# =============================================================================
# LORA BASELINE
# =============================================================================

def run_lora_baseline(args, config, device):
    """Run LoRA fine-tuning baseline."""
    if not PEFT_AVAILABLE:
        print("ERROR: peft library not installed. Run: pip install peft")
        return {}

    print("\n" + "=" * 70)
    print(f"LORA BASELINE (rank={args.rank}): {args.dataset.upper()}")
    print("=" * 70)

    # Load data
    if len(config["load_args"]) == 2:
        train_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        train_ds = load_dataset(config["load_args"][0],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    all_results = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.pad_token = tokenizer.eos_token

        # Configure LoRA
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

        # Prepare training data
        prompt_template = f"{config['task_prompt']}\n\nText: {{text}}\nLabel:"
        train_texts = []
        for item in train_ds:
            text = item[config["text_field"]]
            label = config["label_map"][item[config["label_field"]]]
            train_texts.append(f"{prompt_template.format(text=text[:200])} {label}")

        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()

        for epoch in range(args.epochs):
            indices = list(range(len(train_texts)))
            random.shuffle(indices)
            epoch_loss = 0

            pbar = tqdm(range(0, len(indices), args.batch_size), desc=f"Epoch {epoch+1}")
            for batch_start in pbar:
                batch_indices = indices[batch_start:batch_start + args.batch_size]
                batch_texts = [train_texts[i] for i in batch_indices]

                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=512).to(device)
                labels = inputs.input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate
        model.eval()
        label_names = list(config["label_map"].values())
        correct = 0
        total = 0

        for item in tqdm(eval_ds, desc="Evaluating"):
            text = item[config["text_field"]]
            true_label = config["label_map"][item[config["label_field"]]]

            prompt = prompt_template.format(text=text[:200])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            response = response.strip().lower()

            pred_label = None
            for label in label_names:
                if label.lower() in response:
                    pred_label = label
                    break

            if pred_label and pred_label.lower() == true_label.lower():
                correct += 1
            total += 1

        accuracy = 100 * correct / total
        all_results[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "trainable_params": trainable_params,
        }
        print(f"Seed {seed}: {accuracy:.1f}%")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute statistics
    accuracies = [r["accuracy"] for r in all_results.values()]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    all_results["summary"] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }
    print(f"\nLoRA (rank={args.rank}): {mean_acc:.1f}% +/- {std_acc:.1f}%")

    return all_results


# =============================================================================
# DORA BASELINE (Weight-Decomposed Low-Rank Adaptation)
# =============================================================================

def run_dora_baseline(args, config, device):
    """Run DoRA fine-tuning baseline.

    DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes weight updates into
    magnitude and direction components. The direction is handled by standard LoRA,
    while the magnitude is handled by a separate learnable parameter.

    DoRA has been shown to outperform LoRA, especially at lower ranks.
    Reference: https://arxiv.org/abs/2402.09353
    """
    if not PEFT_AVAILABLE:
        print("ERROR: peft library not installed. Run: pip install peft")
        return {}

    print("\n" + "=" * 70)
    print(f"DORA BASELINE (rank={args.rank}): {args.dataset.upper()}")
    print("=" * 70)

    # Load data
    if len(config["load_args"]) == 2:
        train_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        train_ds = load_dataset(config["load_args"][0],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    all_results = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.pad_token = tokenizer.eos_token

        # Configure DoRA (same as LoRA but with use_dora=True)
        dora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,  # Key difference: enable DoRA
        )
        model = get_peft_model(model, dora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

        # Prepare training data
        prompt_template = f"{config['task_prompt']}\n\nText: {{text}}\nLabel:"
        train_texts = []
        for item in train_ds:
            text = item[config["text_field"]]
            label = config["label_map"][item[config["label_field"]]]
            train_texts.append(f"{prompt_template.format(text=text[:200])} {label}")

        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()

        for epoch in range(args.epochs):
            indices = list(range(len(train_texts)))
            random.shuffle(indices)
            epoch_loss = 0

            pbar = tqdm(range(0, len(indices), args.batch_size), desc=f"Epoch {epoch+1}")
            for batch_start in pbar:
                batch_indices = indices[batch_start:batch_start + args.batch_size]
                batch_texts = [train_texts[i] for i in batch_indices]

                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=512).to(device)
                labels = inputs.input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate
        model.eval()
        label_names = list(config["label_map"].values())
        correct = 0
        total = 0

        for item in tqdm(eval_ds, desc="Evaluating"):
            text = item[config["text_field"]]
            true_label = config["label_map"][item[config["label_field"]]]

            prompt = prompt_template.format(text=text[:200])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            response = response.strip().lower()

            pred_label = None
            for label in label_names:
                if label.lower() in response:
                    pred_label = label
                    break

            if pred_label and pred_label.lower() == true_label.lower():
                correct += 1
            total += 1

        accuracy = 100 * correct / total
        all_results[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "trainable_params": trainable_params,
        }
        print(f"Seed {seed}: {accuracy:.1f}%")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute statistics
    accuracies = [r["accuracy"] for r in all_results.values()]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    all_results["summary"] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }
    print(f"\nDoRA (rank={args.rank}): {mean_acc:.1f}% +/- {std_acc:.1f}%")

    return all_results


# =============================================================================
# TEXT RELAY BASELINE (Fair Llama -> Mistral text communication)
# =============================================================================

def run_text_relay_baseline(args, config, device):
    """Run fair text relay baseline: Llama generates hint, Mistral classifies.

    This is a FAIR text-based cross-model communication baseline that:
    1. Sends input text to Llama with a prompt asking for a classification hint
    2. Gets Llama's generated hint (max 50 tokens)
    3. Sends the hint to Mistral for final classification
    4. Compares Mistral's output to ground truth

    This provides a proper comparison point for the latent bridge approach.
    """
    print("\n" + "=" * 70)
    print(f"TEXT RELAY BASELINE (Llama->Mistral): {args.dataset.upper()}")
    print("=" * 70)
    print("Llama generates text hint, Mistral classifies based on hint.")

    # Load evaluation data
    if len(config["load_args"]) == 2:
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    # Get label options string for prompts
    label_names = list(config["label_map"].values())
    label_options = ", ".join(label_names)

    all_results = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        # Load Llama (sender)
        print("Loading Llama (sender)...")
        llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_model.eval()

        # Load Mistral (receiver)
        print("Loading Mistral (receiver)...")
        mistral_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
        mistral_model.eval()

        correct = 0
        total = 0
        example_hints = []  # Store a few examples for logging

        for idx, item in enumerate(tqdm(eval_ds, desc=f"Text relay (seed {seed})")):
            text = item[config["text_field"]]
            true_label = config["label_map"][item[config["label_field"]]]

            # Step 1: Llama generates a hint
            llama_prompt = (
                f"Read this text and provide a brief hint about what category/sentiment "
                f"it belongs to (options: {label_options}): {text[:256]}\nHint:"
            )
            llama_inputs = llama_tokenizer(
                llama_prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                llama_outputs = llama_model.generate(
                    **llama_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=llama_tokenizer.eos_token_id,
                )

            hint = llama_tokenizer.decode(
                llama_outputs[0][llama_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Store first few examples for logging
            if idx < 3:
                example_hints.append({
                    "text": text[:100] + "...",
                    "hint": hint,
                    "true_label": true_label
                })

            # Step 2: Mistral classifies based on hint
            mistral_prompt = (
                f"Based on this hint: '{hint}'\n"
                f"Classify as {label_options}:\nAnswer:"
            )
            mistral_inputs = mistral_tokenizer(
                mistral_prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                mistral_outputs = mistral_model.generate(
                    **mistral_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=mistral_tokenizer.eos_token_id,
                )

            response = mistral_tokenizer.decode(
                mistral_outputs[0][mistral_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip().lower()

            # Match prediction to label
            pred_label = None
            for label in label_names:
                if label.lower() in response:
                    pred_label = label
                    break

            if pred_label and pred_label.lower() == true_label.lower():
                correct += 1
            total += 1

        accuracy = 100 * correct / total
        all_results[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "example_hints": example_hints,
        }
        print(f"Seed {seed}: {accuracy:.1f}% ({correct}/{total})")

        # Log example hints
        print("  Example hints generated by Llama:")
        for ex in example_hints:
            print(f"    Text: {ex['text']}")
            print(f"    Hint: {ex['hint']}")
            print(f"    True: {ex['true_label']}")
            print()

        # Clean up
        del llama_model, mistral_model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute statistics
    accuracies = [r["accuracy"] for r in all_results.values() if "accuracy" in r]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    all_results["summary"] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "random_baseline": config.get("random_baseline", 0),
    }
    print(f"\nText Relay (Llama->Mistral): {mean_acc:.1f}% +/- {std_acc:.1f}%")
    print(f"Random baseline: {config.get('random_baseline', 0):.1f}%")

    return all_results


# =============================================================================
# PROMPT TUNING BASELINE
# =============================================================================

class SoftPromptTuning(nn.Module):
    """Learnable soft prompts for a frozen LLM."""

    def __init__(self, num_tokens, embed_dim, target_rms=0.03):
        super().__init__()
        self.num_tokens = num_tokens
        self.soft_prompts = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)
        self.output_scale = nn.Parameter(torch.tensor(target_rms))

    def forward(self, batch_size):
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        rms = torch.sqrt((prompts ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        return (prompts / rms) * self.output_scale


def run_prompt_tuning_baseline(args, config, device):
    """Run prompt tuning baseline (no sender model)."""
    print("\n" + "=" * 70)
    print(f"PROMPT TUNING BASELINE ({args.soft_tokens} tokens): {args.dataset.upper()}")
    print("=" * 70)
    print("This proves whether Llama (sender) actually helps.")

    # Load data
    if len(config["load_args"]) == 2:
        train_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        train_ds = load_dataset(config["load_args"][0],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    all_results = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.pad_token = tokenizer.eos_token

        # Get target RMS
        with torch.no_grad():
            embeds = model.get_input_embeddings().weight.float()
            target_rms = embeds.pow(2).mean(dim=1).sqrt().median().item()

        # Initialize soft prompts
        embed_dim = model.config.hidden_size
        soft_prompt = SoftPromptTuning(args.soft_tokens, embed_dim, target_rms)
        soft_prompt = soft_prompt.bfloat16().to(device)
        soft_prompt.train()

        optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr=args.lr)

        # Primer
        primer = "Classify:"
        primer_enc = tokenizer(primer, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            primer_embeds = model.get_input_embeddings()(primer_enc.input_ids)

        # Training
        from torch.utils.data import DataLoader
        dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        iter_dl = iter(dl)

        labels_list = list(config["label_map"].values())
        pbar = tqdm(range(args.steps), desc="Training")

        for step in pbar:
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(dl)
                batch = next(iter_dl)

            # Handle both tensor and list label fields
            labels_raw = batch[config["label_field"]]
            labels_list = labels_raw.tolist() if hasattr(labels_raw, 'tolist') else labels_raw
            label_texts = [config["label_map"][l] for l in labels_list]
            # Get input texts from the batch
            text_field = config["text_field"]
            input_texts = [t[:200] for t in batch[text_field]]  # Truncate to 200 chars
            B = len(label_texts)

            # Get soft prompts
            prompts = soft_prompt(B)

            # Tokenize input texts and get embeddings
            with torch.no_grad():
                # Tokenize input text
                input_enc = tokenizer(input_texts, return_tensors="pt", padding=True,
                                     truncation=True, max_length=128, add_special_tokens=False).to(device)
                input_embeds = model.get_input_embeddings()(input_enc.input_ids)

                primer_batch = primer_embeds.expand(B, -1, -1)
                answer_texts = [f" {l}{tokenizer.eos_token}" for l in label_texts]
                answer_enc = tokenizer(answer_texts, return_tensors="pt", padding=True,
                                      truncation=True, max_length=16, add_special_tokens=False).to(device)
                answer_embeds = model.get_input_embeddings()(answer_enc.input_ids)

            # Concatenate: [soft_prompts] + [input_text] + [primer] + [answer]
            inputs_embeds = torch.cat([prompts, input_embeds, primer_batch, answer_embeds], dim=1)

            K = prompts.shape[1]
            T_len = input_embeds.shape[1]
            P_len = primer_batch.shape[1]

            # Labels: ignore soft prompts, input text, and primer; only predict answer
            ignore_prefix = torch.full((B, K + T_len + P_len), -100, dtype=torch.long, device=device)
            answer_labels = answer_enc.input_ids.clone()
            answer_labels[answer_enc.attention_mask == 0] = -100
            labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

            # Attention mask
            soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
            full_mask = torch.cat([soft_mask, input_enc.attention_mask, primer_enc.attention_mask.expand(B, -1), answer_enc.attention_mask], dim=1)

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels_tensor)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # Evaluate
        soft_prompt.eval()
        correct = 0
        total = 0

        for item in tqdm(eval_ds, desc="Evaluating"):
            label = config["label_map"][item[config["label_field"]]]
            # Get input text
            input_text = item[config["text_field"]][:200]  # Truncate to 200 chars

            with torch.no_grad():
                prompts = soft_prompt(1)

                # Tokenize input text and get embeddings
                input_enc = tokenizer(input_text, return_tensors="pt", truncation=True,
                                     max_length=128, add_special_tokens=False).to(device)
                input_embeds = model.get_input_embeddings()(input_enc.input_ids)

                # Concatenate: [soft_prompts] + [input_text] + [primer]
                combined_embeds = torch.cat([prompts, input_embeds, primer_embeds], dim=1)

                soft_mask = torch.ones(1, prompts.shape[1], device=device)
                attn_mask = torch.cat([soft_mask, input_enc.attention_mask, primer_enc.attention_mask], dim=1)

                out_ids = model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                output = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip().lower()

            if label.lower() in output:
                correct += 1
            total += 1

        accuracy = 100 * correct / total
        all_results[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "soft_tokens": args.soft_tokens,
            "trainable_params": args.soft_tokens * embed_dim + 1,
        }
        print(f"Seed {seed}: {accuracy:.1f}%")

        del model, soft_prompt
        gc.collect()
        torch.cuda.empty_cache()

    # Compute statistics
    accuracies = [r["accuracy"] for r in all_results.values() if "accuracy" in r]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    all_results["summary"] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }
    print(f"\nPrompt Tuning ({args.soft_tokens} tokens): {mean_acc:.1f}% +/- {std_acc:.1f}%")

    return all_results


# =============================================================================
# ENSEMBLE BASELINE
# =============================================================================

def run_ensemble_baseline(args, config, device):
    """Run ensemble baseline combining Llama and Mistral predictions.

    This baseline tests whether simply averaging the predictions from both
    models achieves similar results to the trained bridge. It computes:

        ensemble_logits = alpha * llama_logits + (1-alpha) * mistral_logits

    and takes argmax as the prediction. Tests multiple alpha values to find
    the best mixing coefficient.

    This is a critical experiment to verify that the bridge provides
    "super-additive" performance beyond simple ensembling.
    """
    print("\n" + "=" * 70)
    print(f"ENSEMBLE BASELINE: {args.dataset.upper()}")
    print("=" * 70)
    print("Testing whether ensembling Llama + Mistral achieves bridge performance.")

    # Load evaluation data
    if len(config["load_args"]) == 2:
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    # Alpha values to test
    alphas = args.alphas if hasattr(args, 'alphas') and args.alphas else [0.3, 0.5, 0.7]
    print(f"Testing alpha values: {alphas}")

    all_results = {}

    # Store per-sample logits from each model for ensemble computation
    llama_all_logits = []
    mistral_all_logits = []
    true_labels = []

    # Get unique canonical labels (used for all models and ensemble computation)
    canonical_labels = sorted(set(config["label_map"].values()))

    models_info = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral"),
    ]

    for model_id, model_name in models_info:
        print(f"\nCollecting logits from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        # Build label tokens for this model's tokenizer
        label_tokens = {}
        for label_text in canonical_labels:
            tokens = tokenizer.encode(label_text, add_special_tokens=False)
            label_tokens[label_text] = tokens[0]

        model_logits = []
        correct = 0
        total = 0

        for item in tqdm(eval_ds, desc=f"Evaluating {model_name}"):
            text = item[config["text_field"]]
            raw_label = item[config["label_field"]]
            # Normalize raw label to canonical form (e.g., "1" -> "A" for ARC)
            true_label = config["label_map"].get(raw_label, raw_label)

            prompt = config["prompt_template"].format(
                text=text[:256],
                task_prompt=config["task_prompt"]
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]

            # Get logits for each canonical label token
            label_logits = torch.stack([logits[label_tokens[lbl]] for lbl in canonical_labels])
            # Convert to float32 for numerical stability in softmax/averaging
            label_logits = label_logits.float().cpu()
            model_logits.append(label_logits)

            # Also track individual model accuracy
            pred = label_logits.argmax().item()
            pred_label = canonical_labels[pred]
            if pred_label == true_label:
                correct += 1
            total += 1

            # Store normalized true labels only once (from first model)
            if model_name == "Llama":
                true_labels.append(true_label)

        accuracy = 100 * correct / total
        all_results[f"{model_name}_individual"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        print(f"{model_name} individual: {accuracy:.1f}% ({correct}/{total})")

        if model_name == "Llama":
            llama_all_logits = model_logits
        else:
            mistral_all_logits = model_logits

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Now compute ensemble accuracy for each alpha
    print("\n" + "-" * 50)
    print("Computing ensemble accuracies...")
    print("-" * 50)

    ensemble_results = {}
    best_alpha = None
    best_accuracy = 0

    for alpha in alphas:
        correct = 0
        total = len(true_labels)

        for i in range(total):
            # Ensemble: alpha * llama + (1-alpha) * mistral
            ensemble_logits = alpha * llama_all_logits[i] + (1 - alpha) * mistral_all_logits[i]
            pred = ensemble_logits.argmax().item()
            pred_label = canonical_labels[pred]

            if pred_label == true_labels[i]:
                correct += 1

        accuracy = 100 * correct / total
        ensemble_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        print(f"Ensemble (alpha={alpha}): {accuracy:.1f}% ({correct}/{total})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    # Also try softmax-based averaging (probability space instead of logit space)
    print("\n--- Probability-space ensemble (softmax then average) ---")
    prob_ensemble_results = {}

    for alpha in alphas:
        correct = 0
        total = len(true_labels)

        for i in range(total):
            # Convert to probabilities, then ensemble
            llama_probs = torch.softmax(llama_all_logits[i], dim=-1)
            mistral_probs = torch.softmax(mistral_all_logits[i], dim=-1)
            ensemble_probs = alpha * llama_probs + (1 - alpha) * mistral_probs
            pred = ensemble_probs.argmax().item()
            pred_label = canonical_labels[pred]

            if pred_label == true_labels[i]:
                correct += 1

        accuracy = 100 * correct / total
        prob_ensemble_results[f"prob_alpha_{alpha}"] = {
            "alpha": alpha,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        print(f"Prob Ensemble (alpha={alpha}): {accuracy:.1f}% ({correct}/{total})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = f"prob_{alpha}"

    all_results["ensemble_logit_space"] = ensemble_results
    all_results["ensemble_prob_space"] = prob_ensemble_results
    all_results["best_ensemble"] = {
        "best_alpha": best_alpha,
        "best_accuracy": best_accuracy,
    }

    print("\n" + "=" * 50)
    print(f"BEST ENSEMBLE: alpha={best_alpha}, accuracy={best_accuracy:.1f}%")
    print("=" * 50)

    return all_results


# =============================================================================
# RIDGE REGRESSION BASELINE (LatentMAS-style)
# =============================================================================

def run_ridge_regression_baseline(args, config, device):
    """Run ridge regression baseline for cross-model alignment.

    This implements the LatentMAS approach: learn a linear mapping from source
    (Llama) hidden states to target (Mistral) embedding space using ridge regression.

    W_align = (X'X + lambda*I)^{-1} X'Y

    This is training-free in the neural network sense - just fits a closed-form
    linear transformation.
    """
    if not SKLEARN_AVAILABLE:
        print("ERROR: sklearn not installed. Run: pip install scikit-learn")
        return {}

    print("\n" + "=" * 70)
    print(f"RIDGE REGRESSION BASELINE: {args.dataset.upper()}")
    print("=" * 70)
    print("Testing linear alignment from Llama hidden states to Mistral embeddings")
    print(f"Lambda values to test: {args.lambda_values}")

    # Load data
    if len(config["load_args"]) == 2:
        train_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0], config["load_args"][1],
                              split=config["eval_split"], trust_remote_code=True)
    else:
        train_ds = load_dataset(config["load_args"][0],
                               split=config["train_split"], trust_remote_code=True)
        eval_ds = load_dataset(config["load_args"][0],
                              split=config["eval_split"], trust_remote_code=True)

    # Limit samples for fitting
    max_fit_samples = args.max_fit_samples
    if max_fit_samples and len(train_ds) > max_fit_samples:
        train_ds = train_ds.shuffle(seed=args.seeds[0]).select(range(max_fit_samples))

    if args.max_samples:
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    # Load models
    print("\nLoading Llama (source) model...")
    src_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    src_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    src_tok.pad_token = src_tok.eos_token

    print("Loading Mistral (target) model...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    tgt_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tgt_tok.pad_token = tgt_tok.eos_token

    # Get target embedding RMS for normalization
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
    print(f"Target embedding RMS: {target_rms:.4f}")

    # ==========================================================================
    # Step 1: Collect alignment data
    # ==========================================================================
    print(f"\n--- Collecting alignment data from {len(train_ds)} samples ---")

    source_hiddens = []  # X: Llama hidden states at layer 31
    target_embeds_list = []   # Y: Mistral embeddings of label text

    label_map = config["label_map"]
    source_layer = args.source_layer

    for item in tqdm(train_ds, desc="Collecting alignment pairs"):
        text = item[config["text_field"]]
        label_idx = item[config["label_field"]]

        # Handle different label formats
        if config.get("label_from_key"):
            label_text = label_map.get(label_idx, str(label_idx))
        else:
            label_text = label_map[label_idx]

        # Build prompt for source model
        prompt = config["prompt_template"].format(
            text=text[:256],
            task_prompt=config["task_prompt"]
        )

        # Get Llama hidden state at layer 31 (last token)
        with torch.no_grad():
            src_enc = src_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            # Get hidden state at specified layer, last token position
            src_h = src_out.hidden_states[source_layer][0, -1, :].float().cpu().numpy()
            source_hiddens.append(src_h)

            # Get Mistral embedding of the label text
            label_enc = tgt_tok(label_text, return_tensors="pt", add_special_tokens=False).to(device)
            label_emb = tgt_model.get_input_embeddings()(label_enc.input_ids)
            # Average over tokens if label has multiple tokens
            label_emb = label_emb[0].mean(dim=0).float().cpu().numpy()
            target_embeds_list.append(label_emb)

    X = np.stack(source_hiddens)  # (N, D_src)
    Y = np.stack(target_embeds_list)   # (N, D_tgt)

    print(f"Source matrix X: {X.shape}")
    print(f"Target matrix Y: {Y.shape}")

    # ==========================================================================
    # Step 2: Test multiple lambda values
    # ==========================================================================
    all_results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        seed_results = {}

        for lambda_reg in args.lambda_values:
            print(f"\n--- Lambda = {lambda_reg} ---")

            # Fit ridge regression
            ridge = Ridge(alpha=lambda_reg, fit_intercept=True)
            ridge.fit(X, Y)

            # Get weight matrix stats
            W = ridge.coef_  # (D_tgt, D_src)
            print(f"Weight matrix W: {W.shape}")
            print(f"  W norm: {np.linalg.norm(W):.4f}")
            print(f"  W mean: {W.mean():.6f}, std: {W.std():.6f}")

            # ==========================================================================
            # Step 3: Evaluate
            # ==========================================================================
            correct = 0
            total = 0

            label_names = list(label_map.values())
            primer_text = config.get("task_prompt", "Answer:")

            for item in tqdm(eval_ds, desc=f"Evaluating lambda={lambda_reg}"):
                text = item[config["text_field"]]
                label_idx = item[config["label_field"]]

                # Handle different label formats
                if config.get("label_from_key"):
                    true_label = label_map.get(label_idx, str(label_idx))
                else:
                    true_label = label_map[label_idx]

                # Build prompt and get Llama hidden state
                prompt = config["prompt_template"].format(
                    text=text[:256],
                    task_prompt=config["task_prompt"]
                )

                with torch.no_grad():
                    src_enc = src_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                    src_out = src_model(**src_enc, output_hidden_states=True)
                    src_h = src_out.hidden_states[source_layer][0, -1, :].float().cpu().numpy()

                    # Transform via ridge regression
                    aligned = ridge.predict(src_h.reshape(1, -1))[0]  # (D_tgt,)

                    # Normalize to match target embedding RMS
                    aligned_tensor = torch.from_numpy(aligned).float().to(device)
                    rms = torch.sqrt((aligned_tensor ** 2).mean() + 1e-8)
                    aligned_tensor = (aligned_tensor / rms) * target_rms
                    aligned_tensor = aligned_tensor.bfloat16()

                    # Use as soft token (single token)
                    soft_token = aligned_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D_tgt)

                    # Get primer embeddings
                    primer_enc = tgt_tok(primer_text, return_tensors="pt", add_special_tokens=False).to(device)
                    primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids).bfloat16()

                    # Concatenate: [Primer] + [Soft Token] -> Generate
                    combined_embeds = torch.cat([primer_embeds, soft_token], dim=1)
                    attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

                    out_ids = tgt_model.generate(
                        inputs_embeds=combined_embeds,
                        attention_mask=attn_mask,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tgt_tok.eos_token_id,
                    )
                    output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()

                # Check if correct
                pred_label = None
                for label in label_names:
                    if label.lower() in output:
                        pred_label = label
                        break

                if pred_label and pred_label.lower() == true_label.lower():
                    correct += 1
                total += 1

            accuracy = 100 * correct / total
            seed_results[f"lambda_{lambda_reg}"] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
            print(f"Lambda {lambda_reg}: {accuracy:.1f}% ({correct}/{total})")

        all_results[f"seed_{seed}"] = seed_results

    # Compute summary statistics across seeds
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 70)

    summary = {}
    for lambda_reg in args.lambda_values:
        accuracies = [all_results[f"seed_{s}"][f"lambda_{lambda_reg}"]["accuracy"]
                     for s in args.seeds]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        summary[f"lambda_{lambda_reg}"] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "all_accuracies": accuracies,
        }
        print(f"Lambda {lambda_reg}: {mean_acc:.1f}% +/- {std_acc:.1f}%")

    # Find best lambda
    best_lambda = max(summary.keys(), key=lambda k: summary[k]["mean_accuracy"])
    print(f"\nBest lambda: {best_lambda} ({summary[best_lambda]['mean_accuracy']:.1f}%)")

    all_results["summary"] = summary
    all_results["best_lambda"] = best_lambda

    # Clean up
    del src_model, tgt_model
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Baselines")

    # Required arguments
    parser.add_argument("--baseline", type=str, required=True,
                       choices=["zeroshot", "fewshot", "lora", "dora", "prompt_tuning",
                                "text_relay", "ensemble", "ridge_regression"],
                       help="Baseline type to run")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["sst2", "agnews", "trec", "arc_easy", "winogrande", "hellaswag", "boolq"],
                       help="Dataset to evaluate on")

    # General settings
    parser.add_argument("--output_dir", default="runs/baselines")
    parser.add_argument("--max_samples", type=int, default=200,
                       help="Max eval samples")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--gpu", type=int, default=0)

    # Few-shot settings
    parser.add_argument("--shots", type=int, default=5,
                       help="Number of few-shot examples")

    # LoRA/DoRA settings
    parser.add_argument("--rank", type=int, default=8, help="LoRA/DoRA rank")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=2000)

    # Prompt tuning settings
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)

    # Training settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Ensemble settings
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                       help="Alpha values to test for ensemble (alpha * llama + (1-alpha) * mistral)")

    # Ridge regression settings
    parser.add_argument("--lambda_values", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0],
                       help="Lambda (regularization) values to test for ridge regression")
    parser.add_argument("--max_fit_samples", type=int, default=1000,
                       help="Max training samples to use for fitting ridge regression")
    parser.add_argument("--source_layer", type=int, default=31,
                       help="Which Llama layer to extract hidden states from (31=final)")

    return parser.parse_args()


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run baseline
    if args.baseline == "zeroshot":
        results = run_zeroshot_baseline(args, config, device)
    elif args.baseline == "fewshot":
        results = run_fewshot_baseline(args, config, device)
    elif args.baseline == "lora":
        results = run_lora_baseline(args, config, device)
    elif args.baseline == "dora":
        results = run_dora_baseline(args, config, device)
    elif args.baseline == "prompt_tuning":
        results = run_prompt_tuning_baseline(args, config, device)
    elif args.baseline == "text_relay":
        results = run_text_relay_baseline(args, config, device)
    elif args.baseline == "ensemble":
        results = run_ensemble_baseline(args, config, device)
    elif args.baseline == "ridge_regression":
        results = run_ridge_regression_baseline(args, config, device)
    else:
        print(f"Unknown baseline: {args.baseline}")
        return

    # Save results
    output = {
        "experiment": f"{args.baseline}_baseline",
        "timestamp": timestamp,
        "config": {
            "baseline": args.baseline,
            "dataset": args.dataset,
            "seeds": args.seeds,
            "max_samples": args.max_samples,
        },
        "results": results,
    }

    # Add baseline-specific config
    if args.baseline == "fewshot":
        output["config"]["shots"] = args.shots
    elif args.baseline == "lora":
        output["config"]["rank"] = args.rank
        output["config"]["epochs"] = args.epochs
    elif args.baseline == "dora":
        output["config"]["rank"] = args.rank
        output["config"]["epochs"] = args.epochs
    elif args.baseline == "prompt_tuning":
        output["config"]["soft_tokens"] = args.soft_tokens
        output["config"]["steps"] = args.steps
    elif args.baseline == "ensemble":
        output["config"]["alphas"] = args.alphas
    elif args.baseline == "ridge_regression":
        output["config"]["lambda_values"] = args.lambda_values
        output["config"]["max_fit_samples"] = args.max_fit_samples
        output["config"]["source_layer"] = args.source_layer

    output_file = f"{args.output_dir}/{args.baseline}_{args.dataset}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
