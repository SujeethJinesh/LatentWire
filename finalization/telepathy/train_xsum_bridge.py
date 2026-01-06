#!/usr/bin/env python3
"""
Train Bridge on XSUM Summarization Task

This script trains the Bridge architecture on the XSUM summarization dataset,
demonstrating that Bridge works for generation tasks beyond classification.

Key differences from classification:
- Uses ROUGE scores instead of accuracy
- Generates full text outputs instead of single tokens
- Tests cross-model transfer for abstractive summarization

Usage:
    python telepathy/train_xsum_bridge.py \
        --sender meta-llama/Meta-Llama-3.1-8B-Instruct \
        --receiver mistralai/Mistral-7B-Instruct-v0.3 \
        --train_samples 1000 \
        --eval_samples 100 \
        --output_dir runs/xsum_bridge
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
# Import Bridge architecture from unified comparison
import sys
sys.path.append('.')
from telepathy.run_unified_comparison import PerceiverResampler, UnifiedBridge
from telepathy.rouge_metrics import compute_rouge, RougeResults


def load_xsum_data(split="train", max_samples=None):
    """Load XSUM dataset."""
    ds = load_dataset("xsum", split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


def compute_rouge_scores(predictions, references):
    """
    Compute ROUGE scores for evaluation using the validated rouge_metrics module.

    Returns a dictionary with percentage scores for compatibility with existing code.
    """
    # Use the validated rouge_metrics implementation
    results = compute_rouge(
        predictions,
        references,
        use_stemmer=True,  # XSUM standard
        compute_confidence_intervals=False,  # Skip for speed during training
        show_progress=False
    )

    # Convert to percentage format for backward compatibility
    avg_scores = {
        'rouge1': results.rouge1_f1_mean * 100,
        'rouge2': results.rouge2_f1_mean * 100,
        'rougeL': results.rougeL_f1_mean * 100
    }

    return avg_scores


def train_xsum_bridge(
    bridge, sender, sender_tok, receiver, receiver_tok,
    train_ds, device, output_dir,
    steps=3000, batch_size=2, lr=2e-4, source_layer=24,
    max_input_length=512, max_output_length=128
):
    """
    Train Bridge on XSUM summarization.

    Uses teacher forcing with next-token prediction loss.
    """
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    bridge.train()

    # Create DataLoader
    def collate_fn(batch):
        documents = [item['document'][:max_input_length] for item in batch]
        summaries = [item['summary'][:max_output_length] for item in batch]
        return documents, summaries

    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, drop_last=True)
    data_iter = iter(dataloader)

    losses = []
    step = 0
    pbar = tqdm(total=steps, desc="Training XSUM Bridge")

    while step < steps:
        try:
            documents, summaries = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            documents, summaries = next(data_iter)

        B = len(documents)

        # Format sender prompts
        src_texts = [f"Summarize this article:\n\n{doc}\n\nSummary:" for doc in documents]

        # Encode with sender
        sender_inputs = sender_tok(src_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=max_input_length)
        sender_inputs = {k: v.to(device) for k, v in sender_inputs.items()}

        with torch.no_grad():
            sender_out = sender(
                input_ids=sender_inputs["input_ids"],
                attention_mask=sender_inputs["attention_mask"],
                output_hidden_states=True
            )
            sender_hidden = sender_out.hidden_states[source_layer]

        # Get soft tokens from bridge
        soft_tokens = bridge(sender_hidden, sender_inputs["attention_mask"])

        # Prepare receiver inputs with teacher forcing
        # Add task prompt and target summary for training
        prompts = [f"\nGenerate a concise summary:\n{summary}" for summary in summaries]

        # Tokenize prompts and create embeddings
        prompt_inputs = receiver_tok(prompts, return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_output_length)
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

        # Get embeddings for prompts
        prompt_embeds = receiver.get_input_embeddings()(prompt_inputs["input_ids"])

        # Concatenate soft tokens + prompts
        inputs_embeds = torch.cat([soft_tokens, prompt_embeds], dim=1)

        # Create attention mask
        soft_mask = torch.ones(B, soft_tokens.shape[1], device=device)
        full_mask = torch.cat([soft_mask, prompt_inputs["attention_mask"]], dim=1)

        # Create labels (shift by 1 for next-token prediction)
        labels = prompt_inputs["input_ids"].clone()
        labels[labels == receiver_tok.pad_token_id] = -100  # Ignore padding

        # Pad labels to match inputs_embeds length
        label_padding = torch.full((B, soft_tokens.shape[1]), -100, device=device)
        labels = torch.cat([label_padding, labels], dim=1)

        # Shift labels for next-token prediction
        labels = labels[:, 1:]  # Remove first token
        labels = torch.cat([labels, torch.full((B, 1), -100, device=device)], dim=1)  # Add padding at end

        # Forward through receiver
        outputs = receiver(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        step += 1
        pbar.update(1)

        if step % 100 == 0:
            pbar.set_postfix({"loss": np.mean(losses[-100:])})

        # Save checkpoint periodically
        if step % 500 == 0:
            checkpoint_path = f"{output_dir}/checkpoint_step{step}.pt"
            torch.save({
                'bridge_state_dict': bridge.state_dict(),
                'step': step,
                'loss': np.mean(losses[-100:])
            }, checkpoint_path)

    pbar.close()

    # Save final checkpoint
    final_checkpoint = f"{output_dir}/final_checkpoint.pt"
    torch.save({
        'bridge_state_dict': bridge.state_dict(),
        'step': step,
        'final_loss': np.mean(losses[-100:])
    }, final_checkpoint)

    return {"final_loss": np.mean(losses[-100:])}


def eval_xsum_bridge(
    bridge, sender, sender_tok, receiver, receiver_tok,
    eval_ds, device, max_input_length=512, max_output_length=128,
    source_layer=24, num_beams=4
):
    """Evaluate Bridge on XSUM with ROUGE scores."""
    bridge.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for item in tqdm(eval_ds, desc="Evaluating XSUM"):
            document = item['document'][:max_input_length]
            reference = item['summary']

            # Format sender prompt
            src_text = f"Summarize this article:\n\n{document}\n\nSummary:"

            # Encode with sender
            sender_inputs = sender_tok(src_text, return_tensors="pt", truncation=True,
                                       max_length=max_input_length)
            sender_inputs = {k: v.to(device) for k, v in sender_inputs.items()}

            sender_out = sender(
                input_ids=sender_inputs["input_ids"],
                attention_mask=sender_inputs["attention_mask"],
                output_hidden_states=True
            )
            sender_hidden = sender_out.hidden_states[source_layer]

            # Get soft tokens from bridge
            soft_tokens = bridge(sender_hidden, sender_inputs["attention_mask"])

            # Prepare generation prompt
            gen_prompt = "\nGenerate a concise summary:"
            gen_inputs = receiver_tok(gen_prompt, return_tensors="pt", add_special_tokens=False)
            gen_embeds = receiver.get_input_embeddings()(gen_inputs["input_ids"].to(device))

            # Concatenate soft tokens + generation prompt
            inputs_embeds = torch.cat([soft_tokens, gen_embeds], dim=1)

            # Generate summary
            outputs = receiver.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_output_length,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=receiver_tok.eos_token_id,
                eos_token_id=receiver_tok.eos_token_id
            )

            # Decode prediction (skip the prompt tokens)
            pred_tokens = outputs[0][soft_tokens.shape[1] + gen_embeds.shape[1]:]
            prediction = receiver_tok.decode(pred_tokens, skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)

    # Compute ROUGE scores
    rouge_scores = compute_rouge_scores(predictions, references)

    return {
        "rouge_scores": rouge_scores,
        "num_samples": len(predictions),
        "sample_predictions": list(zip(predictions[:5], references[:5]))  # First 5 for inspection
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--receiver", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output_dir", default="runs/xsum_bridge")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("XSUM BRIDGE TRAINING")
    print("="*70)
    print(f"Sender: {args.sender}")
    print(f"Receiver: {args.receiver}")
    print(f"Train samples: {args.train_samples}")
    print(f"Eval samples: {args.eval_samples}")
    print(f"Soft tokens: {args.num_tokens}")
    print("="*70)

    # Load models
    print("\nLoading models...")
    sender = AutoModelForCausalLM.from_pretrained(
        args.sender, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sender_tok = AutoTokenizer.from_pretrained(args.sender)
    sender_tok.pad_token = sender_tok.eos_token

    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver, torch_dtype=torch.bfloat16, device_map="auto"
    )
    receiver_tok = AutoTokenizer.from_pretrained(args.receiver)
    receiver_tok.pad_token = receiver_tok.eos_token

    sender.eval()
    receiver.eval()

    # Load data
    print("\nLoading XSUM dataset...")
    train_ds = load_xsum_data("train", args.train_samples)
    eval_ds = load_xsum_data("validation", args.eval_samples)

    all_results = []

    for seed in args.seeds:
        print(f"\nSeed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create bridge
        sender_dim = sender.config.hidden_size
        receiver_dim = receiver.config.hidden_size
        bridge = UnifiedBridge(sender_dim, receiver_dim, args.num_tokens).to(device=device, dtype=torch.bfloat16)

        # Train
        train_info = train_xsum_bridge(
            bridge, sender, sender_tok, receiver, receiver_tok,
            train_ds, device, args.output_dir,
            steps=args.train_steps, batch_size=args.batch_size, lr=args.lr
        )

        # Evaluate
        eval_results = eval_xsum_bridge(
            bridge, sender, sender_tok, receiver, receiver_tok,
            eval_ds, device,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length
        )

        seed_results = {
            "seed": seed,
            "train_info": train_info,
            "eval_results": eval_results
        }
        all_results.append(seed_results)

        print(f"ROUGE Scores:")
        for metric, score in eval_results["rouge_scores"].items():
            print(f"  {metric}: {score:.2f}")

    # Save results
    results_file = f"{args.output_dir}/xsum_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    if len(all_results) > 1:
        print("\nAggregated ROUGE scores across seeds:")
        rouge_keys = ["rouge1", "rouge2", "rougeL"]
        for key in rouge_keys:
            scores = [r["eval_results"]["rouge_scores"][key] for r in all_results]
            print(f"  {key}: {np.mean(scores):.2f} ± {np.std(scores):.2f}")


if __name__ == "__main__":
    main()