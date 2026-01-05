#!/usr/bin/env python3
"""Analyze actual compression ratios from real experiments."""

import json
import math
import os
from pathlib import Path


def find_eval_results():
    """Find all evaluation result files."""
    results = []

    # Look for eval metrics in runs directory
    base_dirs = [
        Path("/Users/sujeethjinesh/Desktop/LatentWire/runs"),
        Path("/Users/sujeethjinesh/Desktop/LatentWire/telepathy/results"),
        Path("/Users/sujeethjinesh/Desktop/LatentWire/finalization/results"),
    ]

    for base_dir in base_dirs:
        if base_dir.exists():
            # Find all metrics.json files
            for metrics_file in base_dir.rglob("*metrics*.json"):
                try:
                    with open(metrics_file) as f:
                        data = json.load(f)
                        if "wire" in data or "compression" in data:
                            results.append((metrics_file, data))
                except:
                    pass

            # Also check FINAL_RESULTS.json files
            for final_file in base_dir.rglob("*FINAL*.json"):
                try:
                    with open(final_file) as f:
                        data = json.load(f)
                        results.append((final_file, data))
                except:
                    pass

    return results


def analyze_compression(file_path, data):
    """Analyze compression metrics from a result file."""
    print(f"\nAnalyzing: {file_path.name}")
    print("=" * 60)

    # Check for wire metrics
    if "wire" in data:
        wire = data["wire"]
        print(f"Wire metrics found:")

        # Get prompt sizes
        if "prompt_chars" in wire:
            llama_chars = wire["prompt_chars"].get("llama", 0)
            qwen_chars = wire["prompt_chars"].get("qwen", 0)
            avg_chars = (llama_chars + qwen_chars) / 2 if qwen_chars else llama_chars
            print(f"  Average prompt chars: {avg_chars:.0f}")
            print(f"  Average prompt bytes (UTF-8): {avg_chars:.0f}")  # Assuming mostly ASCII

        # Get latent sizes
        if "latent_shape" in wire:
            B, M, D = wire["latent_shape"]
            print(f"  Latent shape: B={B}, M={M}, D={D}")

            # Check different quantization levels
            if "latent_bytes" in wire:
                lb = wire["latent_bytes"]
                print(f"  Latent bytes:")
                print(f"    fp32: {lb.get('fp32', 'N/A')}")
                print(f"    fp16: {lb.get('fp16', 'N/A')}")
                if "quantized_with_scales" in lb:
                    print(f"    quantized (with scales): {lb['quantized_with_scales']}")

                    # Calculate actual compression
                    if "prompt_chars" in wire:
                        compression = avg_chars / lb['quantized_with_scales']
                        print(f"\n  ACTUAL COMPRESSION: {compression:.2f}x")

                        if compression >= 4.0:
                            print(f"  ✓ Achieves ≥4x compression")
                        else:
                            print(f"  ✗ Falls short of 4x ({compression:.2f}x)")

    # Check for compression field
    if "compression" in data:
        comp = data["compression"]
        if isinstance(comp, dict):
            for model, ratio in comp.items():
                print(f"  Reported compression ({model}): {ratio:.2f}x")
        else:
            print(f"  Reported compression: {comp}")

    # Check for payload bytes
    if "payload_bytes" in data:
        print(f"  Payload bytes: {data['payload_bytes']}")

    # Check for average prompt tokens
    if "avg_prompt_tokens" in data:
        apt = data["avg_prompt_tokens"]
        if isinstance(apt, dict):
            for model, tokens in apt.items():
                print(f"  Avg prompt tokens ({model}): {tokens:.1f}")


def check_longform_prompts():
    """Check compression for longer prompts."""
    print("\n" + "=" * 60)
    print("CHECKING LONGER PROMPTS")
    print("=" * 60)

    # Longer prompt example (typical SQuAD context)
    long_prompt = """System: You are a helpful assistant. Answer the question based on the given context.

Context: The University of Chicago (UChicago, Chicago, or UChi) is a private research university in Chicago. The university, established in 1890, consists of The College, various graduate programs, and interdisciplinary committees organized into four academic research divisions and seven professional schools. Beyond the arts and sciences, Chicago is also well known for its professional schools, which include the Pritzker School of Medicine, the University of Chicago Booth School of Business, the Law School, the School of Social Service Administration, the Harris School of Public Policy Studies, the Graham School of Continuing Liberal and Professional Studies and the Divinity School. The university currently enrolls approximately 5,000 students in the College and around 15,000 students overall.

Question: When was the University of Chicago established?