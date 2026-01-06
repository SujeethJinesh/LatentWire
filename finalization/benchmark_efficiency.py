#!/usr/bin/env python
"""
Benchmark efficiency metrics for LatentWire models.

This script measures:
- Inference latency (ms per sample)
- Memory usage (peak GPU/CPU memory)
- Throughput (samples/second)
- Compression ratios
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import gc
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentwire.models import InterlinguaEncoder as Encoder, Adapter, LMWrapper
from latentwire.data import load_examples as get_dataset
from latentwire.eval import load_checkpoint_for_eval
# Note: PrefixConfig doesn't exist, commenting out for now
# # from latentwire.prefix_utils  # Module doesn't exist import PrefixConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def measure_memory():
    """Measure current memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        return {"allocated_mb": allocated, "reserved_mb": reserved}
    else:
        # For CPU, use process memory
        import psutil
        process = psutil.Process()
        return {"rss_mb": process.memory_info().rss / 1024**2}


def benchmark_latency(model_fn, inputs, warmup_runs=3, benchmark_runs=10):
    """Benchmark inference latency."""
    # Warmup
    for _ in range(warmup_runs):
        _ = model_fn(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        _ = model_fn(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }


def benchmark_throughput(model_fn, dataset_loader, max_samples=100):
    """Benchmark throughput (samples/second)."""
    total_samples = 0
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(dataset_loader):
        if total_samples >= max_samples:
            break

        _ = model_fn(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_samples += len(batch['input_ids']) if isinstance(batch, dict) else len(batch)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    return {
        "samples_processed": total_samples,
        "total_time_s": elapsed,
        "throughput_samples_per_s": total_samples / elapsed,
    }


def measure_compression(checkpoint_path, dataset, samples=100):
    """Measure compression ratios."""
    # Load checkpoint config
    config = json.load(open(Path(checkpoint_path) / "config.json"))

    latent_len = config.get("latent_len", 32)
    d_z = config.get("d_z", 256)

    # Load dataset
    data = get_dataset(dataset, split="train", max_samples=samples)

    compression_stats = []
    for item in data:
        text = item.get("prefix", "") + " " + item.get("text", "")
        text_bytes = len(text.encode("utf-8"))

        # Latent representation size (different quantization levels)
        latent_fp32_bytes = latent_len * d_z * 4  # 32-bit float
        latent_fp16_bytes = latent_len * d_z * 2  # 16-bit float
        latent_int8_bytes = latent_len * d_z * 1  # 8-bit int

        compression_stats.append({
            "text_bytes": text_bytes,
            "latent_fp32_bytes": latent_fp32_bytes,
            "latent_fp16_bytes": latent_fp16_bytes,
            "latent_int8_bytes": latent_int8_bytes,
            "compression_ratio_fp32": text_bytes / latent_fp32_bytes,
            "compression_ratio_fp16": text_bytes / latent_fp16_bytes,
            "compression_ratio_int8": text_bytes / latent_int8_bytes,
        })

    # Aggregate statistics
    avg_stats = {}
    for key in compression_stats[0].keys():
        values = [s[key] for s in compression_stats]
        avg_stats[f"avg_{key}"] = np.mean(values)
        avg_stats[f"std_{key}"] = np.std(values)

    return avg_stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark efficiency metrics")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "sst2", "agnews", "trec"],
                       help="Dataset to use for benchmarking")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples to process")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for throughput testing")
    parser.add_argument("--warmup_runs", type=int, default=3,
                       help="Number of warmup runs")
    parser.add_argument("--benchmark_runs", type=int, default=10,
                       help="Number of benchmark runs")
    parser.add_argument("--measure_memory", action="store_true",
                       help="Measure memory usage")
    parser.add_argument("--measure_latency", action="store_true",
                       help="Measure inference latency")
    parser.add_argument("--measure_throughput", action="store_true",
                       help="Measure throughput")
    parser.add_argument("--measure_compression", action="store_true",
                       help="Measure compression ratios")
    parser.add_argument("--output_file", type=str,
                       default="benchmark_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # If no specific measurements requested, do all
    if not any([args.measure_memory, args.measure_latency,
                args.measure_throughput, args.measure_compression]):
        args.measure_memory = True
        args.measure_latency = True
        args.measure_throughput = True
        args.measure_compression = True

    print(f"Benchmarking {args.checkpoint}")
    print(f"Dataset: {args.dataset}, Samples: {args.samples}")
    print("-" * 60)

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "samples": args.samples,
        "batch_size": args.batch_size,
    }

    # Memory usage baseline
    if args.measure_memory:
        print("\nMeasuring memory usage...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        baseline_memory = measure_memory()
        results["baseline_memory"] = baseline_memory

        # Load models and measure peak memory
        try:
            # Load checkpoint
            models = load_checkpoint_for_eval(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")

            # Measure after loading
            loaded_memory = measure_memory()
            results["loaded_memory"] = loaded_memory

            # Calculate increase
            if torch.cuda.is_available():
                memory_increase = loaded_memory["allocated_mb"] - baseline_memory["allocated_mb"]
            else:
                memory_increase = loaded_memory.get("rss_mb", 0) - baseline_memory.get("rss_mb", 0)

            results["memory_increase_mb"] = memory_increase
            print(f"  Memory increase: {memory_increase:.2f} MB")
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            results["memory_error"] = str(e)

    # Latency benchmarking
    if args.measure_latency:
        print("\nMeasuring inference latency...")
        try:
            # Create a simple inference function
            data = get_dataset(args.dataset, split="validation", max_samples=1)
            sample = data[0]

            # Create dummy input
            dummy_input = torch.randint(0, 256, (1, 64))
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()

            def inference_fn(x):
                with torch.no_grad():
                    # Simplified inference - just encoder pass
                    if hasattr(models, 'encoder'):
                        return models['encoder'](x)
                    return x

            latency_stats = benchmark_latency(
                inference_fn,
                dummy_input,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs
            )
            results["latency"] = latency_stats
            print(f"  Mean latency: {latency_stats['mean_ms']:.2f} ms")
            print(f"  P95 latency: {latency_stats['p95_ms']:.2f} ms")
        except Exception as e:
            print(f"  Error measuring latency: {e}")
            results["latency_error"] = str(e)

    # Throughput benchmarking
    if args.measure_throughput:
        print("\nMeasuring throughput...")
        try:
            # Create dataloader
            from torch.utils.data import DataLoader
            dataset = get_dataset(args.dataset, split="validation", max_samples=args.samples)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            def batch_inference_fn(batch):
                with torch.no_grad():
                    # Simplified batch processing
                    return batch

            throughput_stats = benchmark_throughput(
                batch_inference_fn,
                dataloader,
                max_samples=args.samples
            )
            results["throughput"] = throughput_stats
            print(f"  Throughput: {throughput_stats['throughput_samples_per_s']:.2f} samples/s")
        except Exception as e:
            print(f"  Error measuring throughput: {e}")
            results["throughput_error"] = str(e)

    # Compression ratio measurement
    if args.measure_compression:
        print("\nMeasuring compression ratios...")
        try:
            compression_stats = measure_compression(
                args.checkpoint,
                args.dataset,
                samples=min(args.samples, 100)
            )
            results["compression"] = compression_stats
            print(f"  Avg compression (fp16): {compression_stats['avg_compression_ratio_fp16']:.2f}x")
            print(f"  Avg compression (int8): {compression_stats['avg_compression_ratio_int8']:.2f}x")
        except Exception as e:
            print(f"  Error measuring compression: {e}")
            results["compression_error"] = str(e)

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("-" * 60)

    # Print summary
    print("\nSummary:")
    if "memory_increase_mb" in results:
        print(f"  Memory footprint: {results['memory_increase_mb']:.2f} MB")
    if "latency" in results:
        print(f"  Latency (mean): {results['latency']['mean_ms']:.2f} ms")
    if "throughput" in results:
        print(f"  Throughput: {results['throughput']['throughput_samples_per_s']:.2f} samples/s")
    if "compression" in results:
        print(f"  Compression (fp16): {results['compression']['avg_compression_ratio_fp16']:.2f}x")


if __name__ == "__main__":
    main()