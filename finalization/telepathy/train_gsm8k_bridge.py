#!/usr/bin/env python3
"""
GSM8K math reasoning experiment for Bridge.

Addresses reviewer concern: "Only classification, no generation/reasoning"

This trains a bridge to transfer math problem understanding from Llama to Mistral,
where Mistral then generates the solution.

Usage:
    python train_gsm8k_bridge.py --num_samples 2000 --epochs 5
"""

import argparse
import json
import os
import re
import time
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random


class TelepathyBridge(nn.Module):
    """Bridge module for reasoning transfer."""

    def __init__(
        self,
        sender_dim: int = 4096,
        receiver_dim: int = 4096,
        num_soft_tokens: int = 32,  # More tokens for reasoning
        internal_dim: int = 512,
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        self.query_tokens = nn.Parameter(torch.randn(num_soft_tokens, internal_dim) * 0.02)
        self.sender_proj = nn.Linear(sender_dim, internal_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=internal_dim, num_heads=8, batch_first=True
        )
        self.output_proj = nn.Linear(internal_dim, receiver_dim)

    def forward(self, sender_hidden_states):
        batch_size = sender_hidden_states.shape[0]
        sender_proj = self.sender_proj(sender_hidden_states)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attn(query=queries, key=sender_proj, value=sender_proj)
        return self.output_proj(attended)


def extract_answer(text):
    """Extract numerical answer from GSM8K format."""
    # GSM8K answers are marked with #### NUMBER
    match = re.search(r'####\s*(\-?[\d,]+)', text)
    if match:
        return match.group(1).replace(',', '')

    # Fallback: look for "answer is X" pattern
    match = re.search(r'answer is[:\s]*(\-?[\d,]+)', text.lower())
    if match:
        return match.group(1).replace(',', '')

    # Fallback: last number in text
    numbers = re.findall(r'\-?[\d,]+', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def prepare_gsm8k_data(dataset, max_samples=None):
    """Prepare GSM8K data for training."""
    data = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        question = item["question"]
        answer_text = item["answer"]

        # Extract the final numerical answer
        final_answer = extract_answer(answer_text)
        if final_answer is None:
            continue

        data.append({
            "question": question,
            "answer_text": answer_text,
            "final_answer": final_answer,
        })

    return data


def train_step(
    bridge, sender, receiver, sender_tok, receiver_tok,
    questions, answer_texts, device, optimizer
):
    """Single training step for reasoning transfer."""
    bridge.train()

    # Encode questions with sender
    sender_inputs = sender_tok(
        questions, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        sender_outputs = sender(**sender_inputs, output_hidden_states=True)
        sender_hidden = sender_outputs.hidden_states[-1]

    # Transform through bridge
    soft_tokens = bridge(sender_hidden)

    # Prepare target sequences for receiver
    # We want receiver to generate the answer given the soft tokens
    target_texts = [f"Solution: {ans}" for ans in answer_texts]
    target_inputs = receiver_tok(
        target_texts, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)

    # Get target embeddings
    target_embeds = receiver.get_input_embeddings()(target_inputs.input_ids)

    # Concatenate soft tokens with target embeddings
    combined_embeds = torch.cat([soft_tokens, target_embeds], dim=1)

    # Create labels: -100 for soft tokens, actual labels for target
    soft_token_labels = torch.full(
        (soft_tokens.shape[0], soft_tokens.shape[1]), -100, dtype=torch.long, device=device
    )
    target_labels = target_inputs.input_ids.clone()
    target_labels[target_inputs.attention_mask == 0] = -100
    labels = torch.cat([soft_token_labels, target_labels], dim=1)

    # Create attention mask
    soft_attn = torch.ones(soft_tokens.shape[0], soft_tokens.shape[1], device=device)
    combined_attn = torch.cat([soft_attn, target_inputs.attention_mask], dim=1)

    # Forward pass
    outputs = receiver(
        inputs_embeds=combined_embeds,
        attention_mask=combined_attn,
        labels=labels,
    )

    loss = outputs.loss

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_gsm8k(bridge, sender, receiver, sender_tok, receiver_tok, eval_data, device, max_samples=200):
    """Evaluate on GSM8K with exact match on final answer."""
    bridge.eval()

    correct = 0
    total = 0
    results = []

    for item in tqdm(eval_data[:max_samples], desc="Evaluating"):
        question = item["question"]
        true_answer = item["final_answer"]

        # Encode question with sender
        sender_inputs = sender_tok(
            question, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            sender_outputs = sender(**sender_inputs, output_hidden_states=True)
            sender_hidden = sender_outputs.hidden_states[-1]

            # Transform through bridge
            soft_tokens = bridge(sender_hidden)

            # Generate solution with receiver
            outputs = receiver.generate(
                inputs_embeds=soft_tokens,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=receiver_tok.eos_token_id,
            )

        response = receiver_tok.decode(outputs[0], skip_special_tokens=True)
        pred_answer = extract_answer(response)

        is_correct = pred_answer and pred_answer == true_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100],
            "true_answer": true_answer,
            "pred_answer": pred_answer,
            "response": response[:150],
            "correct": is_correct,
        })

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, correct, total, results


def evaluate_baseline(model, tokenizer, eval_data, device, max_samples=200, model_name="model"):
    """Evaluate direct prompting baseline."""
    correct = 0
    total = 0

    prompt_template = """Solve this math problem step by step:

{question}

Solution:"""

    for item in tqdm(eval_data[:max_samples], desc=f"Evaluating {model_name}"):
        question = item["question"]
        true_answer = item["final_answer"]

        prompt = prompt_template.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(response)

        is_correct = pred_answer and pred_answer == true_answer
        if is_correct:
            correct += 1
        total += 1

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--num_eval_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_soft_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs/gsm8k_bridge")
    parser.add_argument("--skip_baselines", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GSM8K
    print("Loading GSM8K dataset...")
    train_ds = load_dataset("gsm8k", "main", split="train")
    test_ds = load_dataset("gsm8k", "main", split="test")

    train_data = prepare_gsm8k_data(train_ds, args.num_train_samples)
    eval_data = prepare_gsm8k_data(test_ds, args.num_eval_samples)

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    # Load models
    print("\nLoading Llama (sender)...")
    llama_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama = AutoModelForCausalLM.from_pretrained(
        llama_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    llama_tok = AutoTokenizer.from_pretrained(llama_id)
    llama_tok.pad_token = llama_tok.eos_token

    print("Loading Mistral (receiver)...")
    mistral_id = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral = AutoModelForCausalLM.from_pretrained(
        mistral_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    mistral_tok = AutoTokenizer.from_pretrained(mistral_id)
    mistral_tok.pad_token = mistral_tok.eos_token

    # Freeze base models
    for param in llama.parameters():
        param.requires_grad = False
    for param in mistral.parameters():
        param.requires_grad = False

    # Create bridge
    print(f"\nCreating bridge with {args.num_soft_tokens} soft tokens...")
    bridge = TelepathyBridge(
        sender_dim=4096,
        receiver_dim=4096,
        num_soft_tokens=args.num_soft_tokens,
        internal_dim=512,
    ).to(device).to(torch.bfloat16)

    trainable_params = sum(p.numel() for p in bridge.parameters())
    print(f"Bridge parameters: {trainable_params:,}")

    # Evaluate baselines first
    baseline_results = {}
    if not args.skip_baselines:
        print("\n" + "=" * 60)
        print("BASELINE EVALUATION")
        print("=" * 60)

        print("\nEvaluating Llama direct...")
        llama_acc, llama_corr, llama_tot = evaluate_baseline(
            llama, llama_tok, eval_data, device, args.num_eval_samples, "Llama"
        )
        baseline_results["llama_direct"] = {"accuracy": llama_acc, "correct": llama_corr, "total": llama_tot}
        print(f"Llama: {llama_acc:.1f}% ({llama_corr}/{llama_tot})")

        print("\nEvaluating Mistral direct...")
        mistral_acc, mistral_corr, mistral_tot = evaluate_baseline(
            mistral, mistral_tok, eval_data, device, args.num_eval_samples, "Mistral"
        )
        baseline_results["mistral_direct"] = {"accuracy": mistral_acc, "correct": mistral_corr, "total": mistral_tot}
        print(f"Mistral: {mistral_acc:.1f}% ({mistral_corr}/{mistral_tot})")

    # Training
    print("\n" + "=" * 60)
    print("TRAINING BRIDGE")
    print("=" * 60)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr)

    train_losses = []
    eval_accuracies = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Shuffle training data
        random.shuffle(train_data)

        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(range(0, len(train_data), args.batch_size), desc="Training")
        for batch_start in pbar:
            batch = train_data[batch_start : batch_start + args.batch_size]
            if not batch:
                continue

            questions = [item["question"] for item in batch]
            answers = [item["answer_text"] for item in batch]

            loss = train_step(
                bridge, llama, mistral, llama_tok, mistral_tok,
                questions, answers, device, optimizer
            )

            epoch_loss += loss
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        # Evaluate every epoch
        print("\nEvaluating...")
        accuracy, correct, total, _ = evaluate_gsm8k(
            bridge, llama, mistral, llama_tok, mistral_tok,
            eval_data, device, min(50, args.num_eval_samples)  # Quick eval
        )
        eval_accuracies.append(accuracy)
        print(f"Bridge accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_acc, final_corr, final_tot, detailed_results = evaluate_gsm8k(
        bridge, llama, mistral, llama_tok, mistral_tok,
        eval_data, device, args.num_eval_samples
    )
    print(f"\nBridge final: {final_acc:.1f}% ({final_corr}/{final_tot})")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "experiment": "gsm8k_bridge",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_train_samples": args.num_train_samples,
            "num_eval_samples": args.num_eval_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_soft_tokens": args.num_soft_tokens,
            "seed": args.seed,
            "bridge_params": trainable_params,
        },
        "baseline_results": baseline_results,
        "bridge_results": {
            "accuracy": final_acc,
            "correct": final_corr,
            "total": final_tot,
        },
        "training_losses": train_losses,
        "eval_accuracies": eval_accuracies,
        "sample_predictions": detailed_results[:10],
    }

    output_file = f"{args.output_dir}/gsm8k_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Save checkpoint
    checkpoint = {
        "bridge_state_dict": bridge.state_dict(),
        "config": {
            "sender_dim": 4096,
            "receiver_dim": 4096,
            "num_soft_tokens": args.num_soft_tokens,
            "internal_dim": 512,
        },
    }
    torch.save(checkpoint, f"{args.output_dir}/bridge.pt")
    print(f"Checkpoint saved to: {args.output_dir}/bridge.pt")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if baseline_results:
        print(f"\nBaselines:")
        print(f"  Llama direct:   {baseline_results['llama_direct']['accuracy']:.1f}%")
        print(f"  Mistral direct: {baseline_results['mistral_direct']['accuracy']:.1f}%")

    print(f"\nBridge (Llama→Mistral):")
    print(f"  Accuracy: {final_acc:.1f}%")
    print(f"  Parameters: {trainable_params:,}")

    if baseline_results:
        best_baseline = max(
            baseline_results["llama_direct"]["accuracy"],
            baseline_results["mistral_direct"]["accuracy"]
        )
        gap = final_acc - best_baseline
        print(f"\nGap vs best baseline: {'+' if gap > 0 else ''}{gap:.1f}pp")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION FOR PAPER")
    print("=" * 60)

    if final_acc > 5:  # Any signal above noise
        print("The bridge shows some ability to transfer reasoning signals!")
        print("This extends our method beyond classification.")
    elif final_acc > 0:
        print("Weak signal detected. Reasoning transfer is harder than classification.")
        print("This is an honest limitation to acknowledge.")
    else:
        print("Reasoning transfer failed. The method is currently limited to classification.")
        print("Document this as a limitation and future work direction.")

    # Clean up
    del llama, mistral, bridge
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
