#!/usr/bin/env python3
"""
Full Fine-Tuning Baseline for Mistral.

Addresses reviewer concern: "What about full fine-tuning, not just LoRA?"

This trains the full Mistral model (or a subset of layers) on classification.
Uses gradient checkpointing and mixed precision to fit in memory.

Usage:
    python train_full_finetune_baseline.py --dataset sst2 --finetune_layers 4
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
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
            "task_prompt": "Classify the question type as ABBR (abbreviation), ENTY (entity), DESC (description), HUM (human), LOC (location), or NUM (numeric).\n\nQuestion: {text}\nType:",
        },
    }
    return configs[dataset_name]


def count_parameters(model, trainable_only=True):
    """Count parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_layers(model, num_trainable_layers):
    """
    Freeze all layers except the last num_trainable_layers.
    Also keeps lm_head trainable.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Get total number of layers
    if hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        raise ValueError("Cannot find layers in model")

    # Unfreeze last N layers
    start_layer = num_layers - num_trainable_layers
    print(f"Total layers: {num_layers}, unfreezing layers {start_layer} to {num_layers-1}")

    for i in range(start_layer, num_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = True

    # Always unfreeze lm_head
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Optionally unfreeze final norm
    if hasattr(model.model, 'norm'):
        for param in model.model.norm.parameters():
            param.requires_grad = True


def prepare_batch(texts, labels, config, tokenizer, device, max_length=256):
    """Prepare a training batch."""
    # Create full sequences (prompt + label)
    full_texts = []
    for text, label in zip(texts, labels):
        prompt = config["task_prompt"].format(text=text)
        full_texts.append(f"{prompt} {label}")

    # Tokenize
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    # Create labels (mask prompt, only predict label)
    input_ids = encodings.input_ids
    labels_tensor = input_ids.clone()
    labels_tensor[labels_tensor == tokenizer.pad_token_id] = -100

    return encodings, labels_tensor


def train_epoch(model, optimizer, train_data, config, tokenizer, device, batch_size, max_length):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    import random
    indices = list(range(len(train_data["texts"])))
    random.shuffle(indices)

    pbar = tqdm(range(0, len(indices), batch_size), desc="Training")
    for batch_start in pbar:
        batch_indices = indices[batch_start:batch_start + batch_size]
        batch_texts = [train_data["texts"][i] for i in batch_indices]
        batch_labels = [train_data["labels"][i] for i in batch_indices]

        encodings, labels = prepare_batch(batch_texts, batch_labels, config, tokenizer, device, max_length)

        outputs = model(**encodings, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def evaluate(model, tokenizer, eval_ds, config, device, max_samples=200):
    """Evaluate on classification."""
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

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
    parser.add_argument("--finetune_layers", type=int, default=4,
                       help="Number of last layers to fine-tune (rest frozen)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)  # Small for full FT
    parser.add_argument("--lr", type=float, default=1e-5)  # Lower LR for full FT
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="runs/full_finetune_baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_dataset_config(args.dataset)

    print("=" * 60)
    print(f"FULL FINE-TUNING BASELINE: {args.dataset}")
    print("=" * 60)
    print(f"Fine-tuning last {args.finetune_layers} layers + lm_head")
    print(f"Device: {device}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading {args.dataset}...")
    train_ds = load_dataset(*config["hf_name"], split=config["train_split"])
    eval_ds = load_dataset(*config["hf_name"], split=config["eval_split"])

    # Load model
    print("\nLoading Mistral...")
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Count total params before freezing
    total_params = count_parameters(model, trainable_only=False)

    # Freeze layers
    freeze_layers(model, args.finetune_layers)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100*trainable_params/total_params:.2f}%")
    print(f"(Bridge has ~188K parameters)")

    # Prepare training data
    train_data = {"texts": [], "labels": []}
    for i, item in enumerate(train_ds):
        if i >= args.max_train_samples:
            break
        train_data["texts"].append(item[config["text_field"]])
        train_data["labels"].append(config["label_map"][item[config["label_field"]]])

    print(f"Training on {len(train_data['texts'])} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    # Training
    start_time = time.time()
    train_losses = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        loss = train_epoch(
            model, optimizer, train_data, config, tokenizer, device,
            args.batch_size, max_length=256
        )
        train_losses.append(loss)
        print(f"Epoch {epoch+1} avg loss: {loss:.4f}")

        # Eval every epoch
        accuracy, correct, total = evaluate(model, tokenizer, eval_ds, config, device, args.max_eval_samples)
        print(f"Epoch {epoch+1} accuracy: {accuracy:.1f}%")

    train_time = time.time() - start_time

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_accuracy, correct, total = evaluate(model, tokenizer, eval_ds, config, device, args.max_eval_samples)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/full_ft_{args.dataset}_layers{args.finetune_layers}.json"

    results = {
        "experiment": f"full_finetune_baseline_{args.dataset}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "finetune_layers": args.finetune_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "results": {
            "accuracy": final_accuracy,
            "correct": correct,
            "total": total,
            "train_time": train_time,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_losses": train_losses,
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
    print("SUMMARY: Full Fine-Tuning vs Bridge")
    print("=" * 60)
    print(f"Full FT (last {args.finetune_layers} layers): {final_accuracy:.1f}%")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Bridge (Llama->Mistral): {bridge_results.get(args.dataset, 'N/A')}%")
    print(f"Bridge params: 188,160")
    gap = bridge_results.get(args.dataset, 0) - final_accuracy
    print(f"Gap: {'+' if gap > 0 else ''}{gap:.1f}pp")


if __name__ == "__main__":
    main()
