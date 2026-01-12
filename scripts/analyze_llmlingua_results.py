"""
Analyze LLMLingua compression results and compare to LatentWire.

This script loads LLMLingua compression results and computes:
- Compression ratio statistics
- Wire cost comparison (text bytes vs latent bytes)
- Compression speed analysis
- Per-example compression quality
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import statistics


def load_llmlingua_results(results_file: Path) -> Dict[str, Any]:
    """Load LLMLingua results from JSON file."""
    with open(results_file) as f:
        data = json.load(f)
    return data


def compute_wire_bytes(text: str) -> int:
    """Compute UTF-8 byte count for text (honest wire cost)."""
    return len(text.encode("utf-8"))


def compute_latent_wire_bytes(num_tokens: int, d_z: int, bits_per_value: int) -> int:
    """
    Compute wire cost for LatentWire latents.

    Args:
        num_tokens: Number of latent tokens (M)
        d_z: Latent dimension per token
        bits_per_value: Quantization bits (16, 8, 6, 4)

    Returns:
        Total bytes including quantization overhead
    """
    total_values = num_tokens * d_z

    if bits_per_value == 16:
        # fp16: 2 bytes per value
        return total_values * 2
    elif bits_per_value == 8:
        # int8: 1 byte per value + scale overhead
        # Assume 1 scale per token (fp32 = 4 bytes)
        return total_values * 1 + num_tokens * 4
    elif bits_per_value == 6:
        # int6: pack into bytes (6 bits per value)
        packed_bytes = (total_values * 6 + 7) // 8  # Round up
        return packed_bytes + num_tokens * 4  # Add scale overhead
    elif bits_per_value == 4:
        # int4: 0.5 bytes per value
        return total_values // 2 + num_tokens * 4  # Add scale overhead
    else:
        raise ValueError(f"Unsupported bits_per_value: {bits_per_value}")


def analyze_llmlingua_results(results_dir: Path, d_z: int = 256) -> Dict[str, Any]:
    """
    Analyze LLMLingua results and compute wire cost comparisons.

    Args:
        results_dir: Directory containing llmlingua_results.json files
        d_z: LatentWire latent dimension (for comparison)

    Returns:
        Analysis dictionary
    """
    print("=" * 80)
    print("LLMLingua Results Analysis")
    print("=" * 80)
    print()

    # Find all result files
    result_files = list(results_dir.rglob("llmlingua_results.json"))

    if not result_files:
        print(f"No llmlingua_results.json files found in {results_dir}")
        return {}

    print(f"Found {len(result_files)} result file(s)")
    print()

    analyses = []

    for result_file in result_files:
        print(f"Analyzing: {result_file.relative_to(results_dir)}")
        print("-" * 80)

        data = load_llmlingua_results(result_file)
        summary = data["summary"]
        examples = data["examples"]

        # Compute wire costs for all examples
        original_wire_bytes = []
        compressed_wire_bytes = []
        compression_ratios = []

        for ex in examples:
            orig_bytes = compute_wire_bytes(ex["original_prompt"])
            comp_bytes = compute_wire_bytes(ex["compressed_prompt"])

            original_wire_bytes.append(orig_bytes)
            compressed_wire_bytes.append(comp_bytes)
            compression_ratios.append(orig_bytes / comp_bytes if comp_bytes > 0 else 0)

        # Compute LatentWire equivalent wire costs
        target_tokens = summary.get("target_tokens", 32)
        latent_wire_fp16 = compute_latent_wire_bytes(target_tokens, d_z, 16)
        latent_wire_int8 = compute_latent_wire_bytes(target_tokens, d_z, 8)
        latent_wire_int6 = compute_latent_wire_bytes(target_tokens, d_z, 6)
        latent_wire_int4 = compute_latent_wire_bytes(target_tokens, d_z, 4)

        # Statistics
        avg_original_bytes = statistics.mean(original_wire_bytes)
        avg_compressed_bytes = statistics.mean(compressed_wire_bytes)
        avg_wire_compression = statistics.mean(compression_ratios)

        analysis = {
            "config": {
                "compressor_model": summary.get("compressor_model"),
                "use_llmlingua2": summary.get("use_llmlingua2"),
                "question_aware": summary.get("question_aware"),
                "target_tokens": target_tokens,
                "dataset": summary.get("dataset"),
                "samples": summary.get("samples"),
            },
            "compression": {
                "avg_compression_ratio_tokens": summary.get("avg_compression_ratio"),
                "avg_original_tokens": summary.get("avg_origin_tokens"),
                "avg_compressed_tokens": summary.get("avg_compressed_tokens"),
            },
            "wire_cost": {
                "avg_original_bytes": avg_original_bytes,
                "avg_compressed_bytes": avg_compressed_bytes,
                "avg_wire_compression_ratio": avg_wire_compression,
                "latent_wire_fp16": latent_wire_fp16,
                "latent_wire_int8": latent_wire_int8,
                "latent_wire_int6": latent_wire_int6,
                "latent_wire_int4": latent_wire_int4,
            },
            "performance": {
                "avg_compression_time_ms": summary.get("avg_compression_time_ms"),
                "throughput_examples_per_sec": summary.get("throughput_examples_per_sec"),
            },
            "result_file": str(result_file),
        }

        # Print analysis
        print(f"Configuration:")
        print(f"  Compressor: {analysis['config']['compressor_model']}")
        print(f"  LLMLingua-2: {analysis['config']['use_llmlingua2']}")
        print(f"  Question-aware: {analysis['config']['question_aware']}")
        print(f"  Target tokens: {analysis['config']['target_tokens']}")
        print()

        print(f"Token Compression:")
        print(f"  Average ratio: {analysis['compression']['avg_compression_ratio_tokens']:.2f}x")
        print(f"  Original tokens: {analysis['compression']['avg_original_tokens']:.1f}")
        print(f"  Compressed tokens: {analysis['compression']['avg_compressed_tokens']:.1f}")
        print()

        print(f"Wire Cost (bytes):")
        print(f"  Original text: {avg_original_bytes:.0f} bytes")
        print(f"  Compressed text (LLMLingua): {avg_compressed_bytes:.0f} bytes")
        print(f"  Wire compression ratio: {avg_wire_compression:.2f}x")
        print()
        print(f"  LatentWire comparisons (M={target_tokens}, d_z={d_z}):")
        print(f"    fp16: {latent_wire_fp16} bytes ({avg_original_bytes/latent_wire_fp16:.2f}x compression)")
        print(f"    int8: {latent_wire_int8} bytes ({avg_original_bytes/latent_wire_int8:.2f}x compression)")
        print(f"    int6: {latent_wire_int6} bytes ({avg_original_bytes/latent_wire_int6:.2f}x compression)")
        print(f"    int4: {latent_wire_int4} bytes ({avg_original_bytes/latent_wire_int4:.2f}x compression)")
        print()

        print(f"Performance:")
        print(f"  Avg compression time: {analysis['performance']['avg_compression_time_ms']:.1f}ms")
        print(f"  Throughput: {analysis['performance']['throughput_examples_per_sec']:.2f} examples/sec")
        print()

        analyses.append(analysis)

    # Compare across configurations if multiple
    if len(analyses) > 1:
        print("=" * 80)
        print("Cross-Configuration Comparison")
        print("=" * 80)
        print()

        for i, analysis in enumerate(analyses):
            config = analysis["config"]
            print(f"{i+1}. {config['compressor_model']} (M={config['target_tokens']}, "
                  f"LLMLingua2={config['use_llmlingua2']}, QAware={config['question_aware']})")
            print(f"   Token compression: {analysis['compression']['avg_compression_ratio_tokens']:.2f}x")
            print(f"   Wire compression: {analysis['wire_cost']['avg_wire_compression_ratio']:.2f}x")
            print(f"   Compressed bytes: {analysis['wire_cost']['avg_compressed_bytes']:.0f}")
            print(f"   Compression time: {analysis['performance']['avg_compression_time_ms']:.1f}ms")
            print()

    return {
        "analyses": analyses,
        "summary": {
            "num_configurations": len(analyses),
            "results_dir": str(results_dir),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze LLMLingua compression results")
    parser.add_argument("--results_dir", type=str, default="runs/llmlingua_baseline",
                        help="Directory containing LLMLingua results")
    parser.add_argument("--d_z", type=int, default=256,
                        help="LatentWire latent dimension for comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for analysis JSON (optional)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    # Run analysis
    analysis = analyze_llmlingua_results(results_dir, d_z=args.d_z)

    # Save analysis if requested
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)

        print("=" * 80)
        print(f"Analysis saved to: {output_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()
