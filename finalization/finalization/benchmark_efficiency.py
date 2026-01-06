#!/usr/bin/env python3
"""
Unified Efficiency Benchmark for LatentWire.

This script measures comprehensive efficiency metrics for Phase 4:
- Latency (inference time per sample)
- Memory usage (peak GPU/CPU memory)
- Throughput (samples per second)
- Compression ratios (wire bytes saved)

Compares:
1. LatentWire: Compressed soft token approach
2. Text Baseline: Full text prompting
3. Token-Budget: Truncated text baseline
4. Linear Probe: Direct feature baseline

Usage:
    python scripts/benchmark_efficiency.py \
        --checkpoint runs/checkpoint \
        --dataset squad \
        --samples 100 \
        --output_file results/phase4_efficiency_squad.json
"""

import argparse
import json
import os
import sys
import time
import torch
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from latentwire.eval import load_model_and_data, run_eval_batch
from latentwire.models import load_checkpoint
from latentwire.data import load_examples
from latentwire.metrics import compute_wire_bytes, compute_compression_ratio


def get_memory_usage():
    """Get current memory usage in MB."""
    memory_stats = {}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024

    # CPU memory (requires psutil)
    try:
        import psutil
        process = psutil.Process()
        memory_stats['cpu_mb'] = process.memory_info().rss / 1024 / 1024
    except ImportError:
        memory_stats['cpu_mb'] = 0

    return memory_stats


def measure_latency(
    model,
    inputs,
    num_warmup: int = 3,
    num_runs: int = 10
) -> Dict[str, float]:
    """Measure inference latency with warmup runs."""

    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Actual measurement
    latencies = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    return {
        'mean_ms': np.mean(latencies) * 1000,
        'std_ms': np.std(latencies) * 1000,
        'min_ms': np.min(latencies) * 1000,
        'max_ms': np.max(latencies) * 1000,
        'p50_ms': np.percentile(latencies, 50) * 1000,
        'p95_ms': np.percentile(latencies, 95) * 1000,
        'p99_ms': np.percentile(latencies, 99) * 1000,
    }


def benchmark_method(
    method_name: str,
    model,
    dataloader,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Benchmark a specific method."""

    print(f"\nBenchmarking {method_name}...")

    results = {
        'method': method_name,
        'latencies_ms': [],
        'memory_usage': [],
        'throughput_samples_per_sec': 0,
        'compression_ratio': 1.0,
        'wire_bytes_saved': 0,
        'accuracy_metrics': {},
    }

    # Initial memory measurement
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    initial_memory = get_memory_usage()

    # Process samples
    total_time = 0
    num_samples = 0
    all_predictions = []
    all_targets = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=method_name)):
        if args.max_batches and batch_idx >= args.max_batches:
            break

        # Move batch to device
        if torch.cuda.is_available():
            batch = {k: v.cuda() if torch.is_tensor(v) else v
                    for k, v in batch.items()}

        # Measure batch latency
        batch_start = time.perf_counter()

        with torch.no_grad():
            if method_name == "latentwire":
                outputs = model.generate_from_latent(batch['latent'], batch['labels'])
            elif method_name == "text_baseline":
                outputs = model.generate_from_text(batch['input_ids'], batch['labels'])
            elif method_name == "token_budget":
                truncated_ids = batch['input_ids'][:, :args.token_budget]
                outputs = model.generate_from_text(truncated_ids, batch['labels'])
            else:
                outputs = model(batch['input_ids'], labels=batch['labels'])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_end = time.perf_counter()
        batch_time = batch_end - batch_start

        total_time += batch_time
        num_samples += batch['input_ids'].shape[0]

        # Record predictions and targets
        if 'predictions' in outputs:
            all_predictions.extend(outputs['predictions'])
        if 'labels' in batch:
            all_targets.extend(batch['labels'].cpu().numpy())

        # Record latency per sample
        latency_per_sample = (batch_time / batch['input_ids'].shape[0]) * 1000
        results['latencies_ms'].append(latency_per_sample)

        # Memory usage snapshot
        current_memory = get_memory_usage()
        results['memory_usage'].append(current_memory)

    # Calculate throughput
    results['throughput_samples_per_sec'] = num_samples / total_time if total_time > 0 else 0

    # Calculate compression metrics for latentwire
    if method_name == "latentwire" and hasattr(model, 'compute_compression'):
        compression_stats = model.compute_compression(dataloader)
        results['compression_ratio'] = compression_stats.get('compression_ratio', 1.0)
        results['wire_bytes_saved'] = compression_stats.get('bytes_saved', 0)

    # Calculate accuracy metrics
    if all_predictions and all_targets:
        from sklearn.metrics import accuracy_score, f1_score
        results['accuracy_metrics'] = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'f1_macro': f1_score(all_targets, all_predictions, average='macro'),
        }

    # Aggregate latency statistics
    if results['latencies_ms']:
        results['latency_stats'] = {
            'mean_ms': np.mean(results['latencies_ms']),
            'std_ms': np.std(results['latencies_ms']),
            'p50_ms': np.percentile(results['latencies_ms'], 50),
            'p95_ms': np.percentile(results['latencies_ms'], 95),
            'p99_ms': np.percentile(results['latencies_ms'], 99),
        }

    # Peak memory usage
    peak_memory = get_memory_usage()
    results['peak_memory_mb'] = {
        'gpu': peak_memory.get('gpu_max_allocated_mb', 0),
        'cpu': max([m.get('cpu_mb', 0) for m in results['memory_usage']] + [0]),
    }

    # Memory overhead from initial
    results['memory_overhead_mb'] = {
        'gpu': peak_memory.get('gpu_max_allocated_mb', 0) - initial_memory.get('gpu_allocated_mb', 0),
        'cpu': results['peak_memory_mb']['cpu'] - initial_memory.get('cpu_mb', 0),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark efficiency metrics')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default='squad',
                      choices=['squad', 'sst2', 'agnews', 'trec', 'xsum'],
                      help='Dataset to benchmark')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for evaluation')
    parser.add_argument('--max_batches', type=int, default=None,
                      help='Maximum number of batches to process')
    parser.add_argument('--token_budget', type=int, default=32,
                      help='Token budget for truncated baseline')
    parser.add_argument('--warmup_runs', type=int, default=3,
                      help='Number of warmup runs')
    parser.add_argument('--benchmark_runs', type=int, default=10,
                      help='Number of benchmark runs')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Output JSON file for results')
    parser.add_argument('--methods', type=str, nargs='+',
                      default=['latentwire', 'text_baseline', 'token_budget'],
                      help='Methods to benchmark')

    args = parser.parse_args()

    print(f"=== Efficiency Benchmark ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples}")
    print(f"Methods: {args.methods}")
    print(f"Output: {args.output_file}")
    print()

    # Create output directory
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and data
    print("Loading checkpoint...")
    checkpoint_data = load_checkpoint(args.checkpoint)
    model = checkpoint_data['model']

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = get_dataset(args.dataset, split='validation', max_samples=args.samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Benchmark each method
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': args.checkpoint,
            'dataset': args.dataset,
            'num_samples': args.samples,
            'batch_size': args.batch_size,
            'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
        },
        'methods': {}
    }

    for method in args.methods:
        if method not in ['latentwire', 'text_baseline', 'token_budget']:
            print(f"Skipping unknown method: {method}")
            continue

        method_results = benchmark_method(
            method_name=method,
            model=model,
            dataloader=dataloader,
            args=args
        )

        all_results['methods'][method] = method_results

        # Print summary for this method
        print(f"\n{method} Results:")
        print(f"  Throughput: {method_results['throughput_samples_per_sec']:.2f} samples/sec")
        if 'latency_stats' in method_results:
            print(f"  Latency: {method_results['latency_stats']['mean_ms']:.2f} ± "
                  f"{method_results['latency_stats']['std_ms']:.2f} ms")
        print(f"  Peak GPU Memory: {method_results['peak_memory_mb']['gpu']:.2f} MB")
        if method == 'latentwire':
            print(f"  Compression Ratio: {method_results['compression_ratio']:.2f}x")
        if method_results['accuracy_metrics']:
            print(f"  Accuracy: {method_results['accuracy_metrics']['accuracy']:.3f}")

    # Add comparative analysis
    if len(all_results['methods']) > 1:
        all_results['comparison'] = compute_comparisons(all_results['methods'])

    # Save results
    print(f"\nSaving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison summary
    if 'comparison' in all_results:
        print("\n=== Efficiency Comparison ===")
        comp = all_results['comparison']
        if 'speedup_vs_text' in comp:
            print(f"LatentWire speedup vs text: {comp['speedup_vs_text']:.2f}x")
        if 'memory_savings_percent' in comp:
            print(f"Memory savings: {comp['memory_savings_percent']:.1f}%")
        if 'compression_efficiency' in comp:
            print(f"Compression efficiency: {comp['compression_efficiency']:.2f}")

    print("\nBenchmark complete!")


def compute_comparisons(methods_results: Dict[str, Any]) -> Dict[str, float]:
    """Compute comparative metrics between methods."""
    comparison = {}

    # Get baseline references
    if 'text_baseline' in methods_results and 'latentwire' in methods_results:
        text_results = methods_results['text_baseline']
        lw_results = methods_results['latentwire']

        # Speedup
        if text_results['throughput_samples_per_sec'] > 0:
            comparison['speedup_vs_text'] = (
                lw_results['throughput_samples_per_sec'] /
                text_results['throughput_samples_per_sec']
            )

        # Memory savings
        text_mem = text_results['peak_memory_mb']['gpu']
        lw_mem = lw_results['peak_memory_mb']['gpu']
        if text_mem > 0:
            comparison['memory_savings_percent'] = (
                (text_mem - lw_mem) / text_mem * 100
            )

        # Compression efficiency (ratio * speedup)
        comparison['compression_efficiency'] = (
            lw_results.get('compression_ratio', 1.0) *
            comparison.get('speedup_vs_text', 1.0)
        )

        # Accuracy retention
        if (text_results.get('accuracy_metrics', {}).get('accuracy') and
            lw_results.get('accuracy_metrics', {}).get('accuracy')):
            comparison['accuracy_retention'] = (
                lw_results['accuracy_metrics']['accuracy'] /
                text_results['accuracy_metrics']['accuracy']
            )

    # Token budget comparison
    if 'token_budget' in methods_results and 'latentwire' in methods_results:
        tb_results = methods_results['token_budget']
        lw_results = methods_results['latentwire']

        # Relative performance at same token budget
        if tb_results.get('accuracy_metrics', {}).get('accuracy'):
            comparison['accuracy_vs_token_budget'] = (
                lw_results.get('accuracy_metrics', {}).get('accuracy', 0) /
                tb_results['accuracy_metrics']['accuracy']
            )

    return comparison


if __name__ == '__main__':
    main()