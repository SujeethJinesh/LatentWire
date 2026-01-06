#!/usr/bin/env python3
"""
LoRA baseline training for Mistral.

Addresses reviewer concern: "What if you just fine-tuned Mistral on the same data?"

This trains a LoRA adapter (~200K params to match bridge) on Mistral for classification.

Usage:
    python train_lora_baseline.py --dataset sst2 --rank 8
"""

import argparse
import json
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm
import time


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
            "task_prompt": "Classify the sentiment as positive or negative.\n\nText: {text}\nSentiment:",
        },
        "agnews": {
            "hf_name": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            "train_split": "train",
            "eval_split": "test",
            "task_prompt": "Classify the topic as World, Sports, Business, or Sci/Tech.\n\nText: {text}\nTopic:",
        },
        "trec": {
            "hf_name": ("trec",),
            "text_field": "text",
            "label_field": "coarse_label",
            "label_map": {0: "ABBR", 1: "ENTY", 2: "DESC", 3: "HUM", 4: "LOC", 5: "NUM"},
            "train_split": "train",
            "eval_split": "test",
            "task_prompt": "Classify the question type as ABBR, ENTY, DESC, HUM, LOC, or NUM.\n\nQuestion: {text}\nType:",
        },
    }
    return configs[dataset_name]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_dataset(ds, config, tokenizer, max_samples=None):
    """Prepare dataset for training."""
    texts = []
    labels = []

    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        text = item[config["text_field"]]
        label_id = item[config["label_field"]]
        label_text = config["label_map"][label_id]

        # Create prompt + completion
        prompt = config["task_prompt"].format(text=text)
        full_text = f"{prompt} {label_text}"

        texts.append(full_text)
        labels.append(label_text)

    return texts, labels


def train_lora(model, tokenizer, train_texts, device, config, args):
    """Simple training loop for LoRA."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    total_loss = 0
    num_batches = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        epoch_loss = 0

        # Simple batch iteration
        indices = list(range(len(train_texts)))
        import random
        random.shuffle(indices)

        pbar = tqdm(range(0, len(indices), args.batch_size), desc="Training")
        for batch_start in pbar:
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            batch_texts = [train_texts[i] for i in batch_indices]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # Forward pass with labels = input_ids for causal LM
            labels = inputs.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / (len(indices) // args.batch_size + 1)
        print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")
        total_loss += epoch_loss

    return total_loss / num_batches


def evaluate_lora(model, tokenizer, eval_ds, config, device, max_samples=200):
    """Evaluate LoRA model on classification."""
    model.eval()
    label_map = config["label_map"]
    label_names = list(label_map.values())

    correct = 0
    total = 0

    for i, item in enumerate(tqdm(eval_ds, desc="Evaluating", total=min(max_samples, len(eval_ds)))):
        if i >= max_samples:
            break

        text = item[config["text_field"]]
        true_label = label_map[item[config["label_field"]]]

        prompt = config["task_prompt"].format(text=text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

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
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="runs/lora_baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_dataset_config(args.dataset)

    # Load dataset
    print(f"Loading {args.dataset}...")
    train_ds = load_dataset(*config["hf_name"], split=config["train_split"])
    eval_ds = load_dataset(*config["hf_name"], split=config["eval_split"])

    all_results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Training LoRA on Mistral (seed {seed})")
        print(f"{'='*60}")

        torch.manual_seed(seed)

        # Load model
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        # Configure LoRA
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=["q_proj", "v_proj"],  # Standard targets
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        trainable_params = count_parameters(model)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"(Bridge has ~188K parameters)")

        # Prepare training data
        train_texts, _ = prepare_dataset(train_ds, config, tokenizer, args.max_train_samples)
        print(f"Training on {len(train_texts)} samples")

        # Train
        start_time = time.time()
        avg_loss = train_lora(model, tokenizer, train_texts, device, config, args)
        train_time = time.time() - start_time

        # Evaluate
        print("\nEvaluating...")
        accuracy, correct, total = evaluate_lora(model, tokenizer, eval_ds, config, device, args.max_eval_samples)

        print(f"\nSeed {seed} Results:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"  Train time: {train_time:.1f}s")
        print(f"  Trainable params: {trainable_params:,}")

        all_results[f"seed_{seed}"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "train_time": train_time,
            "trainable_params": trainable_params,
            "avg_loss": avg_loss,
        }

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Compute statistics
    accuracies = [r["accuracy"] for r in all_results.values()]
    mean_acc = sum(accuracies) / len(accuracies)
    std_acc = (sum((x - mean_acc) ** 2 for x in accuracies) / len(accuracies)) ** 0.5

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/lora_{args.dataset}_r{args.rank}.json"

    results = {
        "experiment": f"lora_baseline_{args.dataset}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "rank": args.rank,
            "alpha": args.alpha,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "seeds": args.seeds,
        },
        "results": all_results,
        "summary": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "trainable_params": all_results[f"seed_{args.seeds[0]}"]["trainable_params"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary comparison
    bridge_results = {
        "sst2": 96.7,
        "agnews": 90.7,
        "trec": 95.3,
    }

    print("\n" + "=" * 60)
    print("SUMMARY: LoRA vs Bridge")
    print("=" * 60)
    print(f"LoRA (Mistral, rank={args.rank}): {mean_acc:.1f}% +/- {std_acc:.1f}%")
    print(f"Bridge (Llama->Mistral): {bridge_results.get(args.dataset, 'N/A')}%")
    gap = bridge_results.get(args.dataset, 0) - mean_acc
    print(f"Gap: {'+' if gap > 0 else ''}{gap:.1f}pp")


if __name__ == "__main__":
    main()
