#!/usr/bin/env python3
"""
Evaluate XSUM Bridge Model

This script evaluates a trained Bridge model on the XSUM summarization task,
computing ROUGE scores and generating sample outputs for qualitative analysis.

Usage:
    python telepathy/eval_xsum_bridge.py \
        --checkpoint runs/xsum_bridge/final_checkpoint.pt \
        --output_dir runs/xsum_eval \
        --num_samples 100
"""

import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import necessary components
import sys
sys.path.append('.')
from telepathy.run_unified_comparison import UnifiedBridge
from telepathy.train_xsum_bridge import load_xsum_data
from telepathy.rouge_metrics import compute_rouge, RougeResults, save_rouge_results


def evaluate_checkpoint(
    checkpoint_path,
    sender_model_id,
    receiver_model_id,
    eval_ds,
    device,
    num_tokens=32,
    source_layer=24,
    max_input_length=512,
    max_output_length=128,
    num_beams=4
):
    """Evaluate a single checkpoint."""

    # Load models
    print("Loading models...")
    sender = AutoModelForCausalLM.from_pretrained(
        sender_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sender_tok = AutoTokenizer.from_pretrained(sender_model_id)
    sender_tok.pad_token = sender_tok.eos_token

    receiver = AutoModelForCausalLM.from_pretrained(
        receiver_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    receiver_tok = AutoTokenizer.from_pretrained(receiver_model_id)
    receiver_tok.pad_token = receiver_tok.eos_token

    sender.eval()
    receiver.eval()

    # Load bridge checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    sender_dim = sender.config.hidden_size
    receiver_dim = receiver.config.hidden_size
    bridge = UnifiedBridge(sender_dim, receiver_dim, num_tokens).to(device=device, dtype=torch.bfloat16)
    bridge.load_state_dict(checkpoint['bridge_state_dict'])
    bridge.eval()

    # Evaluate
    predictions = []
    references = []
    detailed_results = []

    with torch.no_grad():
        for idx, item in enumerate(tqdm(eval_ds, desc="Evaluating")):
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

            # Decode prediction
            pred_tokens = outputs[0][soft_tokens.shape[1] + gen_embeds.shape[1]:]
            prediction = receiver_tok.decode(pred_tokens, skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)

            # Store detailed result for first few samples
            if idx < 10:
                detailed_results.append({
                    "document": document[:500] + "..." if len(document) > 500 else document,
                    "reference": reference,
                    "prediction": prediction
                })

    # Compute ROUGE scores with confidence intervals
    print("\nComputing ROUGE scores with confidence intervals...")
    rouge_results = compute_rouge(
        predictions,
        references,
        compute_confidence_intervals=True,
        n_bootstrap=1000,
        return_per_sample=False,  # Set to True if you want per-sample scores
        show_progress=True
    )

    # Also compute baseline scores (zero-shot)
    print("\nComputing zero-shot baseline...")
    baseline_predictions = []

    with torch.no_grad():
        for item in tqdm(eval_ds, desc="Zero-shot baseline"):
            document = item['document'][:max_input_length]

            # Direct generation with receiver model
            prompt = f"Summarize this article:\n\n{document}\n\nSummary:"
            inputs = receiver_tok(prompt, return_tensors="pt", truncation=True,
                                 max_length=max_input_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = receiver.generate(
                **inputs,
                max_new_tokens=max_output_length,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=receiver_tok.eos_token_id
            )

            prediction = receiver_tok.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                            skip_special_tokens=True)
            baseline_predictions.append(prediction)

    # Compute baseline ROUGE scores with confidence intervals
    print("\nComputing baseline ROUGE scores with confidence intervals...")
    baseline_rouge_results = compute_rouge(
        baseline_predictions,
        references,
        compute_confidence_intervals=True,
        n_bootstrap=1000,
        return_per_sample=False,
        show_progress=True
    )

    return {
        "rouge_results": rouge_results,
        "baseline_rouge_results": baseline_rouge_results,
        "num_samples": len(predictions),
        "detailed_results": detailed_results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to bridge checkpoint")
    parser.add_argument("--sender", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--receiver", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--output_dir", default="runs/xsum_eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("XSUM BRIDGE EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Output dir: {args.output_dir}")
    print("="*70)

    # Load evaluation data
    eval_ds = load_xsum_data("validation", args.num_samples)

    # Evaluate
    results = evaluate_checkpoint(
        args.checkpoint,
        args.sender,
        args.receiver,
        eval_ds,
        device,
        num_tokens=args.num_tokens
    )

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Print Bridge ROUGE scores with confidence intervals
    print("\n" + results["rouge_results"].summary_string())

    # Print baseline ROUGE scores with confidence intervals
    print("\nZero-shot Baseline:")
    print(results["baseline_rouge_results"].summary_string())

    # Calculate and print improvements
    print("\nImprovement over baseline:")
    rouge_results = results["rouge_results"]
    baseline_results = results["baseline_rouge_results"]

    improvements = {
        "ROUGE-1 F1": rouge_results.rouge1_f1_mean - baseline_results.rouge1_f1_mean,
        "ROUGE-2 F1": rouge_results.rouge2_f1_mean - baseline_results.rouge2_f1_mean,
        "ROUGE-L F1": rouge_results.rougeL_f1_mean - baseline_results.rougeL_f1_mean
    }

    for metric, improvement in improvements.items():
        print(f"  {metric}: {improvement:+.4f}")

    # Save comprehensive results
    results_file = f"{args.output_dir}/evaluation_results.json"

    # Convert RougeResults objects to dictionaries for JSON serialization
    comprehensive_results = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "num_samples": results["num_samples"],
            "sender_model": args.sender,
            "receiver_model": args.receiver,
            "num_tokens": args.num_tokens,
            "timestamp": datetime.now().isoformat()
        },
        "bridge_rouge": results["rouge_results"].to_dict(),
        "baseline_rouge": results["baseline_rouge_results"].to_dict(),
        "improvements": improvements,
        "detailed_results": results["detailed_results"]
    }

    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Also save ROUGE results using the dedicated function
    save_rouge_results(
        results["rouge_results"],
        f"{args.output_dir}/bridge_rouge_detailed.json",
        metadata={
            "model": "bridge",
            "checkpoint": args.checkpoint,
            "dataset": "xsum",
            "num_samples": results["num_samples"]
        }
    )

    save_rouge_results(
        results["baseline_rouge_results"],
        f"{args.output_dir}/baseline_rouge_detailed.json",
        metadata={
            "model": "zero-shot baseline",
            "dataset": "xsum",
            "num_samples": results["num_samples"]
        }
    )

    # Save sample predictions
    samples_file = f"{args.output_dir}/sample_predictions.txt"
    with open(samples_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("SAMPLE PREDICTIONS\n")
        f.write("="*70 + "\n\n")

        for i, sample in enumerate(results["detailed_results"]):
            f.write(f"SAMPLE {i+1}\n")
            f.write("-"*50 + "\n")
            f.write(f"DOCUMENT (truncated):\n{sample['document']}\n\n")
            f.write(f"REFERENCE:\n{sample['reference']}\n\n")
            f.write(f"PREDICTION:\n{sample['prediction']}\n\n")
            f.write("="*70 + "\n\n")

    print(f"Sample predictions saved to: {samples_file}")


if __name__ == "__main__":
    main()