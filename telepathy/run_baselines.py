#!/usr/bin/env python
# telepathy/run_baselines.py
"""
Unified Baselines Script

Runs various baseline methods for comparison with the telepathy bridge.
Supports: zero-shot, few-shot, LoRA fine-tuning, DoRA fine-tuning, and prompt tuning.

Usage:
    python telepathy/run_baselines.py --baseline zeroshot --dataset sst2
    python telepathy/run_baselines.py --baseline fewshot --dataset agnews --shots 5
    python telepathy/run_baselines.py --baseline lora --dataset trec --rank 8
    python telepathy/run_baselines.py --baseline dora --dataset trec --rank 8
    python telepathy/run_baselines.py --baseline prompt_tuning --dataset sst2 --soft_tokens 8

Baseline Types:
- zeroshot: Direct prompting without examples (Llama and Mistral)
- fewshot: In-context learning with k examples
- lora: LoRA fine-tuning on Mistral
- dora: DoRA fine-tuning on Mistral (Weight-Decomposed Low-Rank Adaptation)
- prompt_tuning: Learnable soft prompts on Mistral (no sender model)
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
        "label_map": {"A": "A", "B": "B", "C": "C", "D": "D"},
        "label_from_key": True,  # Labels are string keys, not int indices
        "num_classes": 4,
        "train_split": "train",
        "eval_split": "test",
        "task_prompt": "Answer A, B, C, or D.",
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

        # Get label tokens
        label_tokens = {}
        for idx, label in config["label_map"].items():
            tokens = tokenizer.encode(label, add_special_tokens=False)
            label_tokens[idx] = tokens[0]

        correct = 0
        total = 0

        for item in tqdm(eval_ds, desc=f"Zero-shot {model_name}"):
            text = item[config["text_field"]]
            label = item[config["label_field"]]

            prompt = config["prompt_template"].format(
                text=text[:256],
                task_prompt=config["task_prompt"]
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]

            # Get logits for each label token
            label_logits = torch.stack([logits[label_tokens[i]] for i in range(len(label_tokens))])
            pred = label_logits.argmax().item()

            if pred == label:
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

            label_texts = [config["label_map"][l] for l in batch[config["label_field"]].tolist()]
            B = len(label_texts)

            # Get soft prompts
            prompts = soft_prompt(B)

            # Get primer and answer embeddings
            with torch.no_grad():
                primer_batch = primer_embeds.expand(B, -1, -1)
                answer_texts = [f" {l}{tokenizer.eos_token}" for l in label_texts]
                answer_enc = tokenizer(answer_texts, return_tensors="pt", padding=True,
                                      truncation=True, max_length=16, add_special_tokens=False).to(device)
                answer_embeds = model.get_input_embeddings()(answer_enc.input_ids)

            # Concatenate
            inputs_embeds = torch.cat([primer_batch, prompts, answer_embeds], dim=1)

            K = prompts.shape[1]
            P_len = primer_batch.shape[1]

            ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
            answer_labels = answer_enc.input_ids.clone()
            answer_labels[answer_enc.attention_mask == 0] = -100
            labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

            soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
            full_mask = torch.cat([primer_enc.attention_mask.expand(B, -1), soft_mask, answer_enc.attention_mask], dim=1)

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

            with torch.no_grad():
                prompts = soft_prompt(1)
                combined_embeds = torch.cat([primer_embeds, prompts], dim=1)
                attn_mask = torch.ones(1, combined_embeds.shape[1], device=device)

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
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Baselines")

    # Required arguments
    parser.add_argument("--baseline", type=str, required=True,
                       choices=["zeroshot", "fewshot", "lora", "dora", "prompt_tuning"],
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

    output_file = f"{args.output_dir}/{args.baseline}_{args.dataset}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
