#!/usr/bin/env python3
"""
Batched latency and throughput benchmark.

Addresses reviewer concern: "Latency comparison is single-sample. What about batched inference?"

Measures latency and throughput at various batch sizes.

Usage:
    python benchmark_batched_latency.py --checkpoint runs/sst2/bridge.pt
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np


class TelepathyBridge(nn.Module):
    """Bridge module for benchmarking."""

    def __init__(
        self,
        sender_dim: int = 4096,
        receiver_dim: int = 4096,
        num_soft_tokens: int = 16,
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


def benchmark_bridge(
    bridge, sender, receiver, sender_tok, receiver_tok, texts, device, batch_size, num_warmup=3, num_runs=10
):
    """Benchmark bridge inference at given batch size."""
    # Prepare batches
    num_samples = len(texts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    latencies = []

    for run in range(num_warmup + num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        for batch_idx in range(num_batches):
            batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            if not batch_texts:
                continue

            # Encode with sender
            inputs = sender_tok(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)

            with torch.no_grad():
                sender_outputs = sender(**inputs, output_hidden_states=True)
                sender_hidden = sender_outputs.hidden_states[-1]

                # Transform through bridge
                soft_tokens = bridge(sender_hidden)

                # Generate with receiver (greedy, short output)
                _ = receiver.generate(
                    inputs_embeds=soft_tokens,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=receiver_tok.eos_token_id,
                )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        if run >= num_warmup:
            latencies.append(elapsed)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = num_samples / avg_latency

    return {
        "batch_size": batch_size,
        "avg_latency_s": avg_latency,
        "std_latency_s": std_latency,
        "throughput_samples_per_s": throughput,
        "latency_per_sample_ms": (avg_latency / num_samples) * 1000,
    }


def benchmark_text_relay(
    sender, receiver, sender_tok, receiver_tok, texts, device, batch_size, num_warmup=3, num_runs=5
):
    """Benchmark text-relay inference (generate text then process)."""
    num_samples = len(texts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    latencies = []

    prompt_template = "Summarize the key points: {text}\n\nSummary:"
    receiver_template = "Based on: {summary}\n\nClassify as positive or negative:"

    for run in range(num_warmup + num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        for batch_idx in range(num_batches):
            batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            if not batch_texts:
                continue

            # Step 1: Sender generates summary (one at a time for text generation)
            summaries = []
            for text in batch_texts:
                prompt = prompt_template.format(text=text[:200])
                inputs = sender_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

                with torch.no_grad():
                    outputs = sender.generate(
                        **inputs, max_new_tokens=50, do_sample=False, pad_token_id=sender_tok.eos_token_id
                    )
                summary = sender_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                summaries.append(summary)

            # Step 2: Receiver classifies (can batch)
            receiver_prompts = [receiver_template.format(summary=s[:100]) for s in summaries]
            inputs = receiver_tok(
                receiver_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                _ = receiver.generate(
                    **inputs, max_new_tokens=5, do_sample=False, pad_token_id=receiver_tok.eos_token_id
                )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        if run >= num_warmup:
            latencies.append(elapsed)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = num_samples / avg_latency

    return {
        "batch_size": batch_size,
        "avg_latency_s": avg_latency,
        "std_latency_s": std_latency,
        "throughput_samples_per_s": throughput,
        "latency_per_sample_ms": (avg_latency / num_samples) * 1000,
    }


def benchmark_direct(model, tokenizer, texts, device, batch_size, num_warmup=3, num_runs=10):
    """Benchmark direct inference with single model."""
    num_samples = len(texts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    prompt_template = "Text: {text}\n\nIs this positive or negative?"
    latencies = []

    for run in range(num_warmup + num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        for batch_idx in range(num_batches):
            batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            if not batch_texts:
                continue

            prompts = [prompt_template.format(text=t[:200]) for t in batch_texts]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)

            with torch.no_grad():
                _ = model.generate(
                    **inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        if run >= num_warmup:
            latencies.append(elapsed)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = num_samples / avg_latency

    return {
        "batch_size": batch_size,
        "avg_latency_s": avg_latency,
        "std_latency_s": std_latency,
        "throughput_samples_per_s": throughput,
        "latency_per_sample_ms": (avg_latency / num_samples) * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--num_samples", type=int, default=64, help="Total samples to process")
    parser.add_argument("--output_dir", default="runs/batched_latency")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load sample texts from SST-2
    print("Loading test data...")
    ds = load_dataset("glue", "sst2", split="validation")
    texts = [item["sentence"] for item in ds][:args.num_samples]
    print(f"Using {len(texts)} samples")

    # Load models
    print("\nLoading Llama...")
    llama_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama = AutoModelForCausalLM.from_pretrained(llama_id, torch_dtype=torch.bfloat16, device_map="auto")
    llama_tok = AutoTokenizer.from_pretrained(llama_id)
    llama_tok.pad_token = llama_tok.eos_token

    print("Loading Mistral...")
    mistral_id = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral = AutoModelForCausalLM.from_pretrained(mistral_id, torch_dtype=torch.bfloat16, device_map="auto")
    mistral_tok = AutoTokenizer.from_pretrained(mistral_id)
    mistral_tok.pad_token = mistral_tok.eos_token

    # Create bridge (use random weights for latency benchmarking)
    print("Creating bridge...")
    bridge = TelepathyBridge(
        sender_dim=4096,
        receiver_dim=4096,
        num_soft_tokens=16,
        internal_dim=512,
    ).to(device).to(torch.bfloat16)
    bridge.eval()

    all_results = {
        "bridge": [],
        "direct_mistral": [],
        "text_relay": [],
    }

    print("\n" + "=" * 70)
    print("BENCHMARKING LATENCY AT VARIOUS BATCH SIZES")
    print("=" * 70)

    for batch_size in args.batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        # Bridge
        print("  Bridge...")
        bridge_result = benchmark_bridge(
            bridge, llama, mistral, llama_tok, mistral_tok, texts, device, batch_size
        )
        all_results["bridge"].append(bridge_result)
        print(f"    Throughput: {bridge_result['throughput_samples_per_s']:.1f} samples/s")
        print(f"    Latency/sample: {bridge_result['latency_per_sample_ms']:.1f} ms")

        # Direct Mistral
        print("  Direct Mistral...")
        direct_result = benchmark_direct(mistral, mistral_tok, texts, device, batch_size)
        all_results["direct_mistral"].append(direct_result)
        print(f"    Throughput: {direct_result['throughput_samples_per_s']:.1f} samples/s")
        print(f"    Latency/sample: {direct_result['latency_per_sample_ms']:.1f} ms")

        # Text-relay (only for small batch sizes, too slow otherwise)
        if batch_size <= 4:
            print("  Text-Relay...")
            relay_result = benchmark_text_relay(
                llama, mistral, llama_tok, mistral_tok, texts[:16], device, batch_size
            )
            all_results["text_relay"].append(relay_result)
            print(f"    Throughput: {relay_result['throughput_samples_per_s']:.1f} samples/s")
            print(f"    Latency/sample: {relay_result['latency_per_sample_ms']:.1f} ms")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/batched_latency.json"

    results = {
        "experiment": "batched_latency_benchmark",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "batch_sizes": args.batch_sizes,
            "num_samples": args.num_samples,
            "device": str(device),
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Throughput (samples/sec) by Batch Size")
    print("=" * 70)
    print(f"\n{'Batch':<8} {'Bridge':<15} {'Direct':<15} {'Text-Relay':<15} {'Speedup':<10}")
    print("-" * 70)

    for i, bs in enumerate(args.batch_sizes):
        bridge_tp = all_results["bridge"][i]["throughput_samples_per_s"]
        direct_tp = all_results["direct_mistral"][i]["throughput_samples_per_s"]

        relay_tp = "N/A"
        speedup = "N/A"
        if i < len(all_results["text_relay"]):
            relay_tp = f"{all_results['text_relay'][i]['throughput_samples_per_s']:.1f}"
            speedup = f"{bridge_tp / all_results['text_relay'][i]['throughput_samples_per_s']:.1f}x"

        print(f"{bs:<8} {bridge_tp:<15.1f} {direct_tp:<15.1f} {relay_tp:<15} {speedup:<10}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check if speedup is maintained
    bridge_tp_1 = all_results["bridge"][0]["throughput_samples_per_s"]
    bridge_tp_max = all_results["bridge"][-1]["throughput_samples_per_s"]

    print(f"\nBridge throughput scales from {bridge_tp_1:.1f} to {bridge_tp_max:.1f} samples/s")
    print(f"Scaling factor: {bridge_tp_max / bridge_tp_1:.1f}x")

    if len(all_results["text_relay"]) > 0:
        relay_tp_1 = all_results["text_relay"][0]["throughput_samples_per_s"]
        print(f"\nSpeedup vs text-relay maintained at batch_size=1: {bridge_tp_1 / relay_tp_1:.1f}x")

    # Clean up
    del llama, mistral, bridge
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
