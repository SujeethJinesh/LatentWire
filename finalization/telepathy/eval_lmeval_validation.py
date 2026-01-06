#!/usr/bin/env python3
"""
lm-evaluation-harness Validation Script.

This script validates our custom evaluation implementations against the
standard lm-evaluation-harness framework. This addresses audit concerns
about benchmark implementation correctness.

The script:
1. Runs our custom implementation
2. Runs lm-evaluation-harness on the same benchmarks
3. Compares results and reports discrepancies

Usage:
    # First, install lm-evaluation-harness:
    # pip install lm-eval

    python eval_lmeval_validation.py --benchmark boolq --output_dir runs/lmeval_validation
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# Check if lm-eval is available
try:
    import lm_eval
    LMEVAL_AVAILABLE = True
except ImportError:
    LMEVAL_AVAILABLE = False


def run_custom_evaluation(benchmark, model_id, output_dir, max_samples=200):
    """Run our custom evaluation implementation."""
    print(f"\n{'='*60}")
    print(f"Running CUSTOM evaluation for {benchmark}")
    print(f"{'='*60}\n")

    # Import here to avoid circular imports
    from eval_reasoning_benchmarks import BENCHMARK_CONFIGS, load_model, evaluate_benchmark
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if benchmark not in BENCHMARK_CONFIGS:
        print(f"Benchmark {benchmark} not found in custom implementation")
        return None

    config = BENCHMARK_CONFIGS[benchmark]
    model, tokenizer = load_model(model_id, device)

    results = evaluate_benchmark(
        model, tokenizer, benchmark, config, device,
        max_samples=max_samples, zero_shot=True
    )

    return {
        "implementation": "custom",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
    }


def run_lmeval_evaluation(benchmark, model_id, output_dir, max_samples=200):
    """Run lm-evaluation-harness evaluation."""
    print(f"\n{'='*60}")
    print(f"Running lm-evaluation-harness for {benchmark}")
    print(f"{'='*60}\n")

    if not LMEVAL_AVAILABLE:
        print("lm-evaluation-harness not installed!")
        print("Install with: pip install lm-eval")
        return None

    # Map our benchmark names to lm-eval task names
    task_mapping = {
        "boolq": "boolq",
        "piqa": "piqa",
        "winogrande": "winogrande",
        "arc": "arc_challenge",
        "commonsenseqa": "commonsense_qa",
    }

    if benchmark not in task_mapping:
        print(f"No lm-eval mapping for benchmark: {benchmark}")
        return None

    task_name = task_mapping[benchmark]

    # Run lm-eval via Python API
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    # Create model wrapper
    lm = HFLM(pretrained=model_id, device="cuda" if torch.cuda.is_available() else "cpu")

    # Run evaluation
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=0,  # Zero-shot
        limit=max_samples,
    )

    # Extract accuracy
    task_results = results["results"][task_name]
    acc_key = "acc,none" if "acc,none" in task_results else "acc"
    accuracy = task_results.get(acc_key, task_results.get("acc_norm,none", 0)) * 100

    return {
        "implementation": "lm-evaluation-harness",
        "accuracy": accuracy,
        "raw_results": task_results,
    }


def compare_results(custom_results, lmeval_results, benchmark):
    """Compare results from both implementations."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {benchmark}")
    print(f"{'='*60}\n")

    if custom_results is None:
        print("Custom results not available")
        return None

    if lmeval_results is None:
        print("lm-eval results not available")
        return None

    custom_acc = custom_results["accuracy"]
    lmeval_acc = lmeval_results["accuracy"]
    diff = abs(custom_acc - lmeval_acc)

    print(f"Custom implementation: {custom_acc:.2f}%")
    print(f"lm-evaluation-harness: {lmeval_acc:.2f}%")
    print(f"Difference: {diff:.2f} percentage points")

    # Flag if difference is significant (>2pp)
    status = "PASS" if diff <= 2.0 else "WARN"
    if diff > 5.0:
        status = "FAIL"

    print(f"Status: {status}")

    return {
        "benchmark": benchmark,
        "custom_accuracy": custom_acc,
        "lmeval_accuracy": lmeval_acc,
        "difference_pp": diff,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate evaluations against lm-evaluation-harness")
    parser.add_argument("--benchmark", choices=["boolq", "piqa", "winogrande", "arc", "commonsenseqa"],
                        default="boolq", help="Benchmark to validate")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model to evaluate")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to evaluate")
    parser.add_argument("--output_dir", default="runs/lmeval_validation", help="Output directory")
    parser.add_argument("--skip_lmeval", action="store_true", help="Skip lm-eval (just run custom)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Validation started at {datetime.now().isoformat()}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model_id}")
    print(f"Max samples: {args.max_samples}")
    print(f"lm-evaluation-harness available: {LMEVAL_AVAILABLE}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": args.benchmark,
        "model_id": args.model_id,
        "max_samples": args.max_samples,
        "lmeval_available": LMEVAL_AVAILABLE,
    }

    # Run custom evaluation
    custom_results = run_custom_evaluation(
        args.benchmark, args.model_id, args.output_dir, args.max_samples
    )
    results["custom"] = custom_results

    # Run lm-eval if available and not skipped
    lmeval_results = None
    if not args.skip_lmeval and LMEVAL_AVAILABLE:
        try:
            import torch
            lmeval_results = run_lmeval_evaluation(
                args.benchmark, args.model_id, args.output_dir, args.max_samples
            )
            results["lmeval"] = lmeval_results
        except Exception as e:
            print(f"lm-eval failed: {e}")
            results["lmeval_error"] = str(e)

    # Compare if both available
    if custom_results and lmeval_results:
        comparison = compare_results(custom_results, lmeval_results, args.benchmark)
        results["comparison"] = comparison

    # Save results
    output_file = os.path.join(args.output_dir, f"validation_{args.benchmark}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    if custom_results:
        print(f"Custom: {custom_results['accuracy']:.2f}%")

    if lmeval_results:
        print(f"lm-eval: {lmeval_results['accuracy']:.2f}%")
    elif not LMEVAL_AVAILABLE:
        print("lm-eval: NOT INSTALLED")
        print("\nTo install: pip install lm-eval")
        print("Then re-run this script for full validation")

    return results


if __name__ == "__main__":
    main()
