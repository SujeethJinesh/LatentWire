#!/usr/bin/env python3
"""
Chain-of-Thought text-relay baseline.

Addresses reviewer concern: "Text-relay baseline is artificially weak (just summarization)"

This implements CoT where Llama explains its reasoning, then Mistral uses that explanation.

Usage:
    python eval_cot_relay.py --dataset sst2
"""

import argparse
import json
import os
import time
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


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
            "cot_prompt": """Analyze this text step by step, then provide your classification.

Text: {text}

Think through: What words indicate sentiment? What is the overall tone?
Provide your reasoning, then state your final classification as either "positive" or "negative".""",
            "receiver_prompt": """Based on the following analysis, classify the sentiment as positive or negative.

Analysis: {analysis}

Classification:""",
        },
        "agnews": {
            "hf_name": ("ag_news",),
            "text_field": "text",
            "label_field": "label",
            "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            "train_split": "train",
            "eval_split": "test",
            "cot_prompt": """Analyze this news article step by step, then classify its topic.

Text: {text}

Think through: What is the main subject? Who are the key entities mentioned?
Provide your reasoning, then state your final classification as one of: World, Sports, Business, or Sci/Tech.""",
            "receiver_prompt": """Based on the following analysis, classify the topic as World, Sports, Business, or Sci/Tech.

Analysis: {analysis}

Topic:""",
        },
        "trec": {
            "hf_name": ("trec",),
            "text_field": "text",
            "label_field": "coarse_label",
            "label_map": {0: "ABBR", 1: "ENTY", 2: "DESC", 3: "HUM", 4: "LOC", 5: "NUM"},
            "train_split": "train",
            "eval_split": "test",
            "cot_prompt": """Analyze this question step by step, then classify what type of answer it expects.

Question: {text}

Think through: What is the question asking for? Is it asking about a person, place, number, description, entity, or abbreviation?
Provide your reasoning, then state your final classification as one of: ABBR, ENTY, DESC, HUM, LOC, or NUM.""",
            "receiver_prompt": """Based on the following analysis, classify the question type as ABBR, ENTY, DESC, HUM, LOC, or NUM.

Analysis: {analysis}

Type:""",
        },
    }
    return configs[dataset_name]


def generate_cot(model, tokenizer, text, config, device, max_tokens=150):
    """Generate chain-of-thought reasoning from Llama."""
    prompt = config["cot_prompt"].format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def classify_with_cot(model, tokenizer, analysis, config, device):
    """Use Mistral to classify based on CoT analysis."""
    prompt = config["receiver_prompt"].format(analysis=analysis)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_label(response, label_names):
    """Parse label from response."""
    response_lower = response.lower()
    for label in label_names:
        if label.lower() in response_lower:
            return label
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sst2", "agnews", "trec"], required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_cot_tokens", type=int, default=150)
    parser.add_argument("--output_dir", default="runs/cot_relay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_dataset_config(args.dataset)

    # Load dataset
    print(f"Loading {args.dataset}...")
    eval_ds = load_dataset(*config["hf_name"], split=config["eval_split"])

    # Load sender (Llama)
    print("Loading Llama (sender)...")
    llama_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama = AutoModelForCausalLM.from_pretrained(
        llama_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    llama_tok = AutoTokenizer.from_pretrained(llama_id)
    llama_tok.pad_token = llama_tok.eos_token

    # Load receiver (Mistral)
    print("Loading Mistral (receiver)...")
    mistral_id = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral = AutoModelForCausalLM.from_pretrained(
        mistral_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    mistral_tok = AutoTokenizer.from_pretrained(mistral_id)
    mistral_tok.pad_token = mistral_tok.eos_token

    label_map = config["label_map"]
    label_names = list(label_map.values())

    correct = 0
    total = 0
    total_cot_tokens = 0
    total_latency_ms = 0
    results_detailed = []

    print(f"\nEvaluating CoT text-relay on {args.dataset}...")
    for i, item in enumerate(tqdm(eval_ds, total=min(args.max_samples, len(eval_ds)))):
        if i >= args.max_samples:
            break

        text = item[config["text_field"]]
        true_label = label_map[item[config["label_field"]]]

        # Synchronize before starting timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # Step 1: Llama generates CoT reasoning
        cot_analysis = generate_cot(llama, llama_tok, text, config, device, args.max_cot_tokens)
        cot_tokens = len(llama_tok.encode(cot_analysis))
        total_cot_tokens += cot_tokens

        # Step 2: Mistral classifies based on CoT
        response = classify_with_cot(mistral, mistral_tok, cot_analysis, config, device)

        # Synchronize before ending timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.time() - start_time) * 1000
        total_latency_ms += latency_ms

        # Parse prediction
        pred_label = parse_label(response, label_names)
        is_correct = pred_label and pred_label.lower() == true_label.lower()

        if is_correct:
            correct += 1
        total += 1

        results_detailed.append({
            "text": text[:100],
            "true_label": true_label,
            "pred_label": pred_label,
            "cot_analysis": cot_analysis[:200],
            "response": response[:50],
            "cot_tokens": cot_tokens,
            "latency_ms": latency_ms,
            "correct": is_correct,
        })

        if i < 5:
            print(f"\n[{i}] True: {true_label}")
            print(f"    CoT ({cot_tokens} tokens): {cot_analysis[:100]}...")
            print(f"    Pred: {response[:30]}, Correct: {is_correct}")

    accuracy = 100 * correct / total
    avg_cot_tokens = total_cot_tokens / total
    avg_latency_ms = total_latency_ms / total

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/cot_relay_{args.dataset}.json"

    results = {
        "experiment": f"cot_relay_{args.dataset}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "max_samples": args.max_samples,
            "max_cot_tokens": args.max_cot_tokens,
            "sender_model": llama_id,
            "receiver_model": mistral_id,
        },
        "summary": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_cot_tokens": avg_cot_tokens,
            "avg_latency_ms": avg_latency_ms,
        },
        "detailed_results": results_detailed[:20],  # First 20 for inspection
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary comparison
    bridge_results = {"sst2": 96.7, "agnews": 90.7, "trec": 95.3}
    bridge_latency = 37.3  # ms
    text_relay_results = {"sst2": 71.0, "agnews": 64.5, "trec": 58.0}
    text_relay_latency = 834.5  # ms

    print("\n" + "=" * 60)
    print("SUMMARY: CoT Text-Relay vs Bridge vs Simple Text-Relay")
    print("=" * 60)
    print(f"\nAccuracy:")
    print(f"  CoT Text-Relay:    {accuracy:.1f}%")
    print(f"  Bridge:            {bridge_results.get(args.dataset, 'N/A')}%")
    print(f"  Simple Text-Relay: {text_relay_results.get(args.dataset, 'N/A')}%")

    print(f"\nLatency:")
    print(f"  CoT Text-Relay:    {avg_latency_ms:.1f}ms ({avg_cot_tokens:.0f} CoT tokens)")
    print(f"  Bridge:            {bridge_latency}ms (16 soft tokens)")
    print(f"  Simple Text-Relay: {text_relay_latency}ms")

    print(f"\nSpeedup vs CoT: {avg_latency_ms / bridge_latency:.1f}x slower")

    # Clean up
    del llama, mistral
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
