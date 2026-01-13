#!/usr/bin/env python
# telepathy/run_benchmarks.py
"""
Unified Benchmarking Script

Benchmarks latency, throughput, and memory usage of the telepathy bridge
compared to alternative approaches.

Usage:
    python telepathy/run_benchmarks.py --benchmark latency --checkpoint runs/sst2/bridge.pt
    python telepathy/run_benchmarks.py --benchmark batched --batch_sizes 1 4 8 16
    python telepathy/run_benchmarks.py --benchmark memory
    python telepathy/run_benchmarks.py --benchmark throughput --num_samples 100

Benchmark Types:
- latency: Single-sample latency comparison (Bridge vs Text-Relay vs Direct)
- batched: Latency at various batch sizes
- memory: Peak GPU memory usage comparison
- throughput: End-to-end throughput measurement
"""
import os
import gc
import json
import time
import argparse
import torch
import torch.nn as nn
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from latent_bridge import LatentBridge

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class BridgeArgs:
    """Args object for LatentBridge interface."""
    def __init__(self, soft_tokens=8, heads=8, depth=2, use_fsq=False, stats_path=None):
        self.soft_tokens = soft_tokens
        self.heads = heads
        self.depth = depth
        self.use_fsq = use_fsq
        self.stats_path = stats_path


def load_models(device):
    """Load Llama and Mistral models."""
    print("Loading Llama...")
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16, device_map=device
    )
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    llama_tok.pad_token = llama_tok.eos_token

    print("Loading Mistral...")
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16, device_map=device
    )
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    mistral_tok.pad_token = mistral_tok.eos_token

    return llama, llama_tok, mistral, mistral_tok


def load_bridge(checkpoint_path, device, soft_tokens=8):
    """Load a trained bridge checkpoint."""
    args = BridgeArgs(soft_tokens=soft_tokens)
    bridge = LatentBridge(args, src_dim=4096, tgt_dim=4096, target_rms=0.03)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'bridge_state_dict' in ckpt:
            bridge.load_state_dict(ckpt['bridge_state_dict'])
        else:
            bridge.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {checkpoint_path}")

    bridge.eval()
    bridge.to(device)
    bridge.to(torch.bfloat16)
    return bridge


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_peak_gpu_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# LATENCY BENCHMARK
# =============================================================================

def run_latency_benchmark(args, device):
    """Measure single-sample latency for different methods."""
    print("\n" + "=" * 70)
    print("LATENCY BENCHMARK")
    print("=" * 70)

    # Load models
    llama, llama_tok, mistral, mistral_tok = load_models(device)

    # Load test data
    print("Loading SST-2 dataset...")
    ds = load_dataset("glue", "sst2", split="validation")
    texts = [item['sentence'] for item in ds][:args.num_samples]

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "num_trials": args.num_trials,
    }

    # 1. Direct Text (Mistral only)
    print("\n--- Direct Text (Mistral) ---")
    direct_times = []

    for _ in range(args.warmup):
        prompt = f"Classify: {texts[0][:200]}\nAnswer:"
        inputs = mistral_tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            mistral.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=mistral_tok.eos_token_id)

    torch.cuda.synchronize() if device.type == 'cuda' else None

    for text in tqdm(texts[:args.num_trials], desc="Direct"):
        prompt = f"Classify: {text[:200]}\nAnswer:"
        inputs = mistral_tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        with torch.no_grad():
            mistral.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=mistral_tok.eos_token_id)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        direct_times.append(time.perf_counter() - t0)

    results["direct_text"] = {
        "method": "direct_text",
        "avg_ms": np.mean(direct_times) * 1000,
        "std_ms": np.std(direct_times) * 1000,
    }
    print(f"Direct Text: {results['direct_text']['avg_ms']:.1f} +/- {results['direct_text']['std_ms']:.1f} ms")

    # 2. Text-Relay (Llama summarize -> Mistral classify)
    print("\n--- Text-Relay (Llama -> text -> Mistral) ---")
    relay_times = []

    for text in tqdm(texts[:args.num_trials], desc="Text-Relay"):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()

        # Llama summarize
        prompt = f"Summarize briefly: {text[:200]}\nSummary:"
        inputs = llama_tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = llama.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=llama_tok.eos_token_id)
            summary = llama_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Mistral classify
        classify_prompt = f"Classify: {summary[:100]}\nAnswer:"
        classify_inputs = mistral_tok(classify_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            mistral.generate(**classify_inputs, max_new_tokens=5, do_sample=False, pad_token_id=mistral_tok.eos_token_id)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        relay_times.append(time.perf_counter() - t0)

    results["text_relay"] = {
        "method": "text_relay",
        "avg_ms": np.mean(relay_times) * 1000,
        "std_ms": np.std(relay_times) * 1000,
    }
    print(f"Text-Relay: {results['text_relay']['avg_ms']:.1f} +/- {results['text_relay']['std_ms']:.1f} ms")

    # 3. Bridge (if checkpoint provided)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print("\n--- Bridge (Llama -> soft tokens -> Mistral) ---")
        bridge = load_bridge(args.checkpoint, device, args.soft_tokens)
        bridge_times = []

        for text in tqdm(texts[:args.num_trials], desc="Bridge"):
            src_inputs = llama_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()

            with torch.no_grad():
                src_out = llama(**src_inputs, output_hidden_states=True)
                src_hidden = src_out.hidden_states[31]
                latents, _, _, _ = bridge(src_hidden, src_inputs.attention_mask)

                primer = mistral_tok.encode("Classify:", add_special_tokens=False)
                prefix_embeds = mistral.get_input_embeddings()(torch.tensor([primer], device=device))
                combined = torch.cat([latents, prefix_embeds], dim=1)
                mistral(inputs_embeds=combined, use_cache=False)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            bridge_times.append(time.perf_counter() - t0)

        results["bridge"] = {
            "method": "bridge",
            "soft_tokens": args.soft_tokens,
            "avg_ms": np.mean(bridge_times) * 1000,
            "std_ms": np.std(bridge_times) * 1000,
        }
        print(f"Bridge: {results['bridge']['avg_ms']:.1f} +/- {results['bridge']['std_ms']:.1f} ms")

        # Speedup
        speedup = results["text_relay"]["avg_ms"] / results["bridge"]["avg_ms"]
        print(f"\nBridge is {speedup:.1f}x faster than Text-Relay")

    return results


# =============================================================================
# BATCHED LATENCY BENCHMARK
# =============================================================================

def run_batched_benchmark(args, device):
    """Measure latency at various batch sizes."""
    print("\n" + "=" * 70)
    print("BATCHED LATENCY BENCHMARK")
    print("=" * 70)

    llama, llama_tok, mistral, mistral_tok = load_models(device)

    # Load test data
    ds = load_dataset("glue", "sst2", split="validation")
    texts = [item['sentence'] for item in ds][:args.num_samples]

    results = {
        "bridge": [],
        "direct_mistral": [],
    }

    # Create bridge (random weights for latency benchmarking)
    bridge_args = BridgeArgs(soft_tokens=args.soft_tokens)
    bridge = LatentBridge(bridge_args, src_dim=4096, tgt_dim=4096, target_rms=0.03)
    bridge = bridge.to(device).to(torch.bfloat16).eval()

    for batch_size in args.batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        num_batches = (len(texts) + batch_size - 1) // batch_size

        # Bridge benchmark
        bridge_latencies = []
        for run in range(args.warmup + args.num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.perf_counter()

            for batch_idx in range(num_batches):
                batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                if not batch_texts:
                    continue

                inputs = llama_tok(batch_texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=256).to(device)
                with torch.no_grad():
                    sender_out = llama(**inputs, output_hidden_states=True)
                    soft_tokens, _, _, _ = bridge(sender_out.hidden_states[-1], inputs.attention_mask)
                    mistral.generate(inputs_embeds=soft_tokens, max_new_tokens=5,
                                   do_sample=False, pad_token_id=mistral_tok.eos_token_id)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            if run >= args.warmup:
                bridge_latencies.append(time.perf_counter() - start)

        bridge_throughput = len(texts) / np.mean(bridge_latencies)
        results["bridge"].append({
            "batch_size": batch_size,
            "throughput_samples_per_s": bridge_throughput,
            "latency_per_sample_ms": (np.mean(bridge_latencies) / len(texts)) * 1000,
        })
        print(f"  Bridge: {bridge_throughput:.1f} samples/s")

        # Direct Mistral benchmark
        direct_latencies = []
        for run in range(args.warmup + args.num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.perf_counter()

            for batch_idx in range(num_batches):
                batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                if not batch_texts:
                    continue

                prompts = [f"Classify: {t[:200]}\nAnswer:" for t in batch_texts]
                inputs = mistral_tok(prompts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=256).to(device)
                with torch.no_grad():
                    mistral.generate(**inputs, max_new_tokens=5, do_sample=False,
                                   pad_token_id=mistral_tok.eos_token_id)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            if run >= args.warmup:
                direct_latencies.append(time.perf_counter() - start)

        direct_throughput = len(texts) / np.mean(direct_latencies)
        results["direct_mistral"].append({
            "batch_size": batch_size,
            "throughput_samples_per_s": direct_throughput,
            "latency_per_sample_ms": (np.mean(direct_latencies) / len(texts)) * 1000,
        })
        print(f"  Direct Mistral: {direct_throughput:.1f} samples/s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Throughput (samples/sec) by Batch Size")
    print("=" * 70)
    print(f"{'Batch':<8} {'Bridge':<15} {'Direct':<15}")
    print("-" * 40)

    for i, bs in enumerate(args.batch_sizes):
        bridge_tp = results["bridge"][i]["throughput_samples_per_s"]
        direct_tp = results["direct_mistral"][i]["throughput_samples_per_s"]
        print(f"{bs:<8} {bridge_tp:<15.1f} {direct_tp:<15.1f}")

    return results


# =============================================================================
# MEMORY BENCHMARK
# =============================================================================

def run_memory_benchmark(args, device):
    """Measure peak GPU memory usage for different methods."""
    print("\n" + "=" * 70)
    print("MEMORY BENCHMARK")
    print("=" * 70)

    results = []

    # 1. Direct inference (Mistral only)
    print("\n--- Direct Mistral Inference ---")
    reset_memory_stats()
    mem_before = get_gpu_memory_mb()

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token

    mem_after_load = get_gpu_memory_mb()

    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)

    mem_peak = get_peak_gpu_memory_mb()

    results.append({
        "method": "direct_inference",
        "model": "Mistral-7B",
        "model_memory_mb": mem_after_load - mem_before,
        "peak_memory_mb": mem_peak,
        "trainable_params": 0,
    })
    print(f"  Peak memory: {mem_peak:.0f} MB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. LoRA (if available)
    if PEFT_AVAILABLE:
        print("\n--- LoRA Training (rank=8) ---")
        reset_memory_stats()
        mem_before = get_gpu_memory_mb()

        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.bfloat16, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        mem_after_load = get_gpu_memory_mb()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        inputs = tokenizer("Hello", return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs.input_ids.clone())
        outputs.loss.backward()

        mem_peak = get_peak_gpu_memory_mb()

        results.append({
            "method": "lora_r8",
            "model": "Mistral-7B + LoRA",
            "model_memory_mb": mem_after_load - mem_before,
            "peak_memory_mb": mem_peak,
            "trainable_params": trainable,
        })
        print(f"  Peak memory: {mem_peak:.0f} MB, Trainable: {trainable:,}")

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Bridge training
    for soft_tokens in [8, 16, 32]:
        print(f"\n--- Bridge Training ({soft_tokens} tokens) ---")
        reset_memory_stats()
        mem_before = get_gpu_memory_mb()

        llama = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16, device_map=device
        )
        llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        llama_tok.pad_token = llama_tok.eos_token

        mem_after_llama = get_gpu_memory_mb()

        mistral = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.bfloat16, device_map=device
        )
        mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        mistral_tok.pad_token = mistral_tok.eos_token

        mem_after_both = get_gpu_memory_mb()

        # Freeze LLMs
        for p in llama.parameters():
            p.requires_grad = False
        for p in mistral.parameters():
            p.requires_grad = False

        bridge_args = BridgeArgs(soft_tokens=soft_tokens)
        bridge = LatentBridge(bridge_args, src_dim=4096, tgt_dim=4096, target_rms=0.03)
        bridge = bridge.to(device).to(torch.bfloat16)
        bridge_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)

        mem_after_bridge = get_gpu_memory_mb()

        # Simulate training
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
        inputs = llama_tok("Hello", return_tensors="pt").to(device)

        with torch.no_grad():
            llama_out = llama(**inputs, output_hidden_states=True)
            llama_hidden = llama_out.hidden_states[31]

        latents, aux_loss, _, _ = bridge(llama_hidden, inputs.attention_mask)
        outputs = mistral(inputs_embeds=latents, labels=inputs.input_ids)
        (outputs.loss + aux_loss).backward()

        mem_peak = get_peak_gpu_memory_mb()

        results.append({
            "method": f"bridge_{soft_tokens}tok",
            "model": "Llama + Bridge + Mistral",
            "soft_tokens": soft_tokens,
            "llama_memory_mb": mem_after_llama - mem_before,
            "mistral_memory_mb": mem_after_both - mem_after_llama,
            "bridge_memory_mb": mem_after_bridge - mem_after_both,
            "peak_memory_mb": mem_peak,
            "trainable_params": bridge_params,
        })
        print(f"  Peak memory: {mem_peak:.0f} MB, Trainable: {bridge_params:,}")

        del llama, mistral, bridge, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("MEMORY BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Peak Memory (MB)':<18} {'Trainable Params':<18}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<25} {r['peak_memory_mb']:>12,.0f} MB    {r['trainable_params']:>15,}")

    return results


# =============================================================================
# THROUGHPUT BENCHMARK
# =============================================================================

def run_throughput_benchmark(args, device):
    """Measure end-to-end throughput."""
    print("\n" + "=" * 70)
    print("THROUGHPUT BENCHMARK")
    print("=" * 70)

    llama, llama_tok, mistral, mistral_tok = load_models(device)

    ds = load_dataset("glue", "sst2", split="validation")
    texts = [item['sentence'] for item in ds][:args.num_samples]

    results = {}

    # Direct Mistral
    print("\n--- Direct Mistral ---")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()

    for text in tqdm(texts, desc="Direct"):
        prompt = f"Classify: {text[:200]}\nAnswer:"
        inputs = mistral_tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            mistral.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=mistral_tok.eos_token_id)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.perf_counter() - start

    results["direct_mistral"] = {
        "total_time_s": elapsed,
        "samples_per_s": len(texts) / elapsed,
        "ms_per_sample": (elapsed / len(texts)) * 1000,
    }
    print(f"  {results['direct_mistral']['samples_per_s']:.1f} samples/s")

    # Bridge (if checkpoint)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print("\n--- Bridge ---")
        bridge = load_bridge(args.checkpoint, device, args.soft_tokens)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()

        for text in tqdm(texts, desc="Bridge"):
            src_inputs = llama_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                src_out = llama(**src_inputs, output_hidden_states=True)
                latents, _, _, _ = bridge(src_out.hidden_states[31], src_inputs.attention_mask)

                primer = mistral_tok.encode("Classify:", add_special_tokens=False)
                prefix_embeds = mistral.get_input_embeddings()(torch.tensor([primer], device=device))
                combined = torch.cat([latents, prefix_embeds], dim=1)
                mistral.generate(inputs_embeds=combined, max_new_tokens=5,
                               do_sample=False, pad_token_id=mistral_tok.eos_token_id)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.perf_counter() - start

        results["bridge"] = {
            "total_time_s": elapsed,
            "samples_per_s": len(texts) / elapsed,
            "ms_per_sample": (elapsed / len(texts)) * 1000,
        }
        print(f"  {results['bridge']['samples_per_s']:.1f} samples/s")

    return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Benchmarks")

    parser.add_argument("--benchmark", type=str, required=True,
                       choices=["latency", "batched", "memory", "throughput"],
                       help="Benchmark type to run")

    # General settings
    parser.add_argument("--output_dir", default="runs/benchmarks")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to bridge checkpoint (for latency/throughput)")
    parser.add_argument("--soft_tokens", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)

    # Latency settings
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)

    # Batched settings
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--num_runs", type=int, default=5)

    # Throughput settings
    parser.add_argument("--num_samples", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run benchmark
    if args.benchmark == "latency":
        results = run_latency_benchmark(args, device)
    elif args.benchmark == "batched":
        results = run_batched_benchmark(args, device)
    elif args.benchmark == "memory":
        results = run_memory_benchmark(args, device)
    elif args.benchmark == "throughput":
        results = run_throughput_benchmark(args, device)
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        return

    # Save results
    output = {
        "experiment": f"{args.benchmark}_benchmark",
        "timestamp": timestamp,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "N/A",
        "config": {
            "benchmark": args.benchmark,
            "checkpoint": args.checkpoint,
            "soft_tokens": args.soft_tokens,
        },
        "results": results,
    }

    output_file = f"{args.output_dir}/{args.benchmark}_benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
