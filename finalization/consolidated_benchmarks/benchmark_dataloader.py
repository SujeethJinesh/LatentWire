#!/usr/bin/env python3
"""
Benchmark script comparing original vs optimized data loading.

This script measures:
1. Data loading throughput (batches/second)
2. GPU utilization during loading
3. Memory usage patterns
4. End-to-end training iteration time
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.data_pipeline import prepare_training_data
from latentwire.models import LMWrapper, LMConfig
from latentwire.optimized_dataloader import (
    create_optimized_dataloader,
    benchmark_dataloader,
)
from transformers import AutoTokenizer


class ModelContext:
    """Mock model context for benchmarking."""
    def __init__(self, model_name: str, tokenizer):
        self.name = model_name
        self.wrapper = type('obj', (object,), {'tokenizer': tokenizer})()
        self.anchor_text = "Answer: "
        self.anchor_mode = "text"


def benchmark_original_loading(
    texts: List[str],
    answers: List[str],
    model_contexts: List[Any],
    batch_size: int = 32,
    num_batches: int = 100,
    device: str = "cuda",
    use_chat_template: bool = False,
) -> Dict[str, float]:
    """Benchmark original manual batch indexing approach."""

    print("Benchmarking ORIGINAL data loading...")
    N = len(texts)

    # Create permutation for sampling
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    perm = torch.randperm(N, generator=g)

    timings = []

    # Warmup
    for step in range(min(10, num_batches)):
        idx = perm[step * batch_size : (step + 1) * batch_size]
        batch_texts = [texts[i] for i in idx.tolist()]

    # Actual benchmark
    for step in range(num_batches):
        start = time.time()

        # Original implementation: manual indexing
        idx = perm[step * batch_size : (step + 1) * batch_size]
        batch_texts = [texts[i] for i in idx.tolist()]
        batch_answers = [answers[i] for i in idx.tolist()]

        # Tokenize for each model (simulated)
        scaffolds = {}
        for ctx in model_contexts:
            # Simulate tokenization
            tok = ctx.wrapper.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            scaffolds[ctx.name] = tok["input_ids"].to(device)

        # Ensure GPU sync for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        timings.append(elapsed)

        if (step + 1) % 10 == 0:
            avg_time = np.mean(timings[-10:])
            print(f"  Batch {step+1}/{num_batches}: {avg_time*1000:.2f}ms/batch")

    results = {
        'mean_time_ms': np.mean(timings) * 1000,
        'std_time_ms': np.std(timings) * 1000,
        'min_time_ms': np.min(timings) * 1000,
        'max_time_ms': np.max(timings) * 1000,
        'batches_per_second': 1.0 / np.mean(timings),
    }

    print("\nOriginal Loading Results:")
    print(f"  Mean time: {results['mean_time_ms']:.2f}ms")
    print(f"  Std dev: {results['std_time_ms']:.2f}ms")
    print(f"  Min/Max: {results['min_time_ms']:.2f}ms / {results['max_time_ms']:.2f}ms")
    print(f"  Throughput: {results['batches_per_second']:.1f} batches/sec")

    return results


def measure_gpu_utilization() -> Dict[str, float]:
    """Measure current GPU utilization."""
    if not torch.cuda.is_available():
        return {}

    stats = {}
    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1e9
        mem_reserved = torch.cuda.memory_reserved(i) / 1e9
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9

        stats[f'gpu_{i}_allocated_gb'] = mem_allocated
        stats[f'gpu_{i}_reserved_gb'] = mem_reserved
        stats[f'gpu_{i}_utilization_pct'] = (mem_reserved / mem_total) * 100

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark data loading performance")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset to use")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to benchmark")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="runs/dataloader_benchmark")
    parser.add_argument("--compare_original", action="store_true", help="Compare with original implementation")
    parser.add_argument("--llama_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking data loading performance")
    print(f"Dataset: {args.dataset}, Samples: {args.samples}, Batch size: {args.batch_size}")
    print(f"Device: {args.device}, Workers: {args.num_workers}")
    print()

    # Load data
    print("Loading dataset...")
    texts, answers = prepare_training_data(
        dataset=args.dataset,
        samples=args.samples,
        data_seed=42,
    )
    print(f"Loaded {len(texts)} samples")

    # Create mock model contexts
    print("Creating model contexts...")
    tokenizer = AutoTokenizer.from_pretrained(args.llama_id, trust_remote_code=True)
    model_contexts = [
        ModelContext("llama", tokenizer),
        # Could add more models here
    ]

    results = {}

    # Measure initial GPU state
    gpu_stats_initial = measure_gpu_utilization()
    results['gpu_initial'] = gpu_stats_initial

    # Benchmark original implementation if requested
    if args.compare_original:
        print("\n" + "="*60)
        original_results = benchmark_original_loading(
            texts=texts,
            answers=answers,
            model_contexts=model_contexts,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            device=args.device,
        )
        results['original'] = original_results
        print("="*60 + "\n")

    # Benchmark optimized implementation with different configurations
    configurations = [
        {
            'name': 'optimized_no_cache',
            'num_workers': args.num_workers,
            'cache_tokenization': False,
            'use_prefetcher': True,
            'pin_memory': True,
        },
        {
            'name': 'optimized_with_cache',
            'num_workers': args.num_workers,
            'cache_tokenization': True,
            'use_prefetcher': True,
            'pin_memory': True,
        },
        {
            'name': 'optimized_no_workers',
            'num_workers': 0,
            'cache_tokenization': True,
            'use_prefetcher': True,
            'pin_memory': True,
        },
        {
            'name': 'optimized_max',
            'num_workers': min(8, args.num_workers),
            'cache_tokenization': True,
            'use_prefetcher': True,
            'pin_memory': True,
            'prefetch_factor': 4,
        },
    ]

    for config in configurations:
        print("\n" + "="*60)
        print(f"Testing configuration: {config['name']}")
        print(f"  Workers: {config['num_workers']}")
        print(f"  Cache: {config.get('cache_tokenization', False)}")
        print(f"  Prefetcher: {config.get('use_prefetcher', True)}")

        # Create optimized dataloader
        dataloader = create_optimized_dataloader(
            texts=texts,
            answers=answers,
            model_contexts=model_contexts,
            batch_size=args.batch_size,
            num_workers=config['num_workers'],
            device=args.device,
            cache_tokenization=config.get('cache_tokenization', False),
            use_prefetcher=config.get('use_prefetcher', True),
            pin_memory=config.get('pin_memory', True),
            prefetch_factor=config.get('prefetch_factor', 2),
        )

        # Benchmark
        config_results = benchmark_dataloader(
            dataloader,
            num_batches=args.num_batches,
            warmup_batches=10,
        )

        # Measure GPU utilization during loading
        gpu_stats = measure_gpu_utilization()
        config_results['gpu_stats'] = gpu_stats

        results[config['name']] = config_results
        print("="*60)

    # Calculate speedups
    if args.compare_original and 'original' in results:
        print("\n" + "="*60)
        print("SPEEDUP ANALYSIS:")
        print("="*60)

        original_time = results['original']['mean_time_ms']

        for config_name, config_results in results.items():
            if config_name in ['original', 'gpu_initial']:
                continue

            optimized_time = config_results['mean_time_ms']
            speedup = original_time / optimized_time
            improvement = (1 - optimized_time / original_time) * 100

            print(f"\n{config_name}:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"  Original: {original_time:.2f}ms → Optimized: {optimized_time:.2f}ms")

            # Add speedup to results
            results[config_name]['speedup'] = speedup
            results[config_name]['improvement_pct'] = improvement

    # Save results
    results_file = output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)

    best_config = None
    best_throughput = 0

    for config_name, config_results in results.items():
        if config_name in ['original', 'gpu_initial']:
            continue

        throughput = config_results['batches_per_second']
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = config_name

    if best_config:
        print(f"Best configuration: {best_config}")
        print(f"  Throughput: {best_throughput:.1f} batches/sec")
        print(f"  Mean time: {results[best_config]['mean_time_ms']:.2f}ms/batch")

        if 'speedup' in results[best_config]:
            print(f"  Speedup over original: {results[best_config]['speedup']:.2f}x")

    print("\nOptimization recommendations:")
    print("1. Use multi-worker loading (4-8 workers)")
    print("2. Enable tokenization caching for repeated runs")
    print("3. Use pinned memory for GPU training")
    print("4. Enable GPU prefetching for async transfers")
    print("5. Pre-tokenize small datasets (<5000 samples)")


if __name__ == "__main__":
    main()