#!/usr/bin/env python
"""
Test script to verify comprehensive metrics collection in eval.py

This script validates that all required metrics for the paper are being collected:
- Compression ratio (token and byte-level)
- Latency (ms per sample)
- Throughput (samples/sec, tokens/sec)
- Memory usage (GB for CPU and GPU)
- F1 score and Exact Match with confidence intervals
- ROUGE scores (for summarization tasks)
- Per-dataset accuracy breakdown
- Bootstrap confidence intervals for all metrics
"""

import json
import sys
from pathlib import Path


def validate_metrics_json(json_path: str) -> bool:
    """
    Validate that a metrics.json file contains all required fields for the paper.

    Args:
        json_path: Path to the metrics.json file

    Returns:
        True if all required metrics are present, False otherwise
    """
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    required_fields = {
        # Basic info
        'samples': 'Number of evaluation samples',
        'max_new_tokens': 'Maximum tokens to generate',
        'latent_len': 'Latent representation length',
        'device': 'Device used (cuda/cpu)',
        'dtype': 'Data type used',

        # Compression metrics
        'compression_ratio': 'Comprehensive compression metrics',
        'byte_compression': 'Byte-level compression ratios',
        'payload_bytes': 'Interlingua payload size',

        # Performance metrics
        'inference_metrics': 'Latency and throughput metrics',
        'memory_usage': 'Memory usage statistics',

        # Quality metrics with confidence intervals
        'text': 'Text baseline results',
        'latent': 'Latent representation results',
        'token_budget': 'Token budget baseline results',

        # Dataset info
        'dataset': 'Dataset name',
    }

    optional_fields = {
        'rouge_scores': 'ROUGE scores for summarization',
        'per_dataset_metrics': 'Per-dataset accuracy breakdown',
    }

    print("="*80)
    print("COMPREHENSIVE METRICS VALIDATION")
    print("="*80)

    missing_required = []
    missing_optional = []

    # Check required fields
    for field, description in required_fields.items():
        if field not in metrics:
            missing_required.append(f"  - {field}: {description}")
        else:
            print(f"✓ {field}: Found")

    # Check optional fields
    for field, description in optional_fields.items():
        if field not in metrics:
            missing_optional.append(f"  - {field}: {description}")
        else:
            print(f"✓ {field}: Found")

    # Validate confidence intervals
    print("\nConfidence Interval Validation:")
    for baseline in ['text', 'latent', 'token_budget']:
        if baseline in metrics:
            for model in ['llama', 'qwen']:
                if model in metrics[baseline]:
                    model_metrics = metrics[baseline][model]
                    has_ci = 'em_with_ci' in model_metrics and 'f1_with_ci' in model_metrics
                    if has_ci:
                        em_ci = model_metrics['em_with_ci']
                        f1_ci = model_metrics['f1_with_ci']
                        print(f"  ✓ {baseline}/{model}: EM CI [{em_ci.get('ci_lower', 0):.3f}, {em_ci.get('ci_upper', 0):.3f}]")
                        print(f"  ✓ {baseline}/{model}: F1 CI [{f1_ci.get('ci_lower', 0):.3f}, {f1_ci.get('ci_upper', 0):.3f}]")
                    else:
                        print(f"  ✗ {baseline}/{model}: Missing confidence intervals")

    # Validate performance metrics
    if 'inference_metrics' in metrics:
        print("\nPerformance Metrics Validation:")
        for baseline in ['text', 'latent', 'token_budget']:
            if baseline in metrics['inference_metrics']:
                perf = metrics['inference_metrics'][baseline]
                print(f"  {baseline}:")
                print(f"    - Throughput: {perf.get('throughput_samples_per_sec', 0):.2f} samples/sec")
                print(f"    - Latency: {perf.get('latency_ms_per_sample', 0):.2f} ms/sample")

    # Validate memory usage
    if 'memory_usage' in metrics:
        print("\nMemory Usage Validation:")
        mem = metrics['memory_usage']
        print(f"  - Process RSS: {mem.get('process_rss_gb', 0):.2f} GB")
        print(f"  - System Available: {mem.get('system_available_gb', 0):.2f} GB")
        for key in mem:
            if 'gpu' in key:
                print(f"  - {key}: {mem[key]:.2f} GB")

    # Validate compression metrics
    if 'compression_ratio' in metrics:
        print("\nCompression Metrics Validation:")
        comp = metrics['compression_ratio']
        if 'token_ratio' in comp:
            for model in ['llama', 'qwen']:
                if model in comp['token_ratio']:
                    print(f"  - {model} token ratio: {comp['token_ratio'][model]:.2f}x")
        if 'byte_ratio' in comp:
            for model in ['llama', 'qwen']:
                if model in comp['byte_ratio']:
                    print(f"  - {model} byte ratio: {comp['byte_ratio'][model]:.2f}x")

    # Report missing fields
    if missing_required:
        print("\n❌ MISSING REQUIRED FIELDS:")
        for field in missing_required:
            print(field)

    if missing_optional:
        print("\n⚠️ MISSING OPTIONAL FIELDS (may not be applicable):")
        for field in missing_optional:
            print(field)

    # Summary
    print("\n" + "="*80)
    if not missing_required:
        print("✅ ALL REQUIRED METRICS PRESENT - Ready for paper!")
        return True
    else:
        print("❌ MISSING CRITICAL METRICS - Need to update evaluation")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_comprehensive_metrics.py <path_to_metrics.json>")
        print("\nThis script validates that all metrics needed for the paper are collected.")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Error: File {json_path} does not exist")
        sys.exit(1)

    if validate_metrics_json(str(json_path)):
        print("\n✓ Metrics collection is comprehensive and ready for paper submission!")
        sys.exit(0)
    else:
        print("\n✗ Metrics collection needs improvement. Update eval.py with missing metrics.")
        sys.exit(1)


if __name__ == "__main__":
    main()