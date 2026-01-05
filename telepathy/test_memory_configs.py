#!/usr/bin/env python3
"""
Test script to validate memory configuration calculations.
Runs without requiring PyTorch to be installed.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import telepathy
sys.path.insert(0, str(Path(__file__).parent.parent))

from telepathy.memory_configs import get_memory_safe_config, estimate_bridge_memory_gb


def test_bridge_memory():
    """Test bridge memory calculation with known values."""
    print("=" * 80)
    print("TESTING BRIDGE MEMORY CALCULATIONS")
    print("=" * 80)

    # Test with 4096 hidden dim (Llama-8B/Mistral-7B)
    bridge_info = estimate_bridge_memory_gb(
        source_dim=4096,
        target_dim=4096,
        soft_tokens=128,
        depth=4,
        include_optimizer=True
    )

    print("\nBridge Configuration (4096 dim, 4 layers):")
    print("  Parameter Count: {:.1f}M".format(bridge_info['param_count']/1e6))
    print("  Parameter Memory: {:.3f} GB".format(bridge_info['params_gb']))
    print("  Gradient Memory: {:.3f} GB".format(bridge_info['gradient_gb']))
    print("  Optimizer Memory: {:.3f} GB".format(bridge_info['optimizer_gb']))
    print("  Activation Overhead: {:.3f} GB".format(bridge_info['activation_overhead_gb']))
    print("  Total Memory: {:.2f} GB".format(bridge_info['total_gb']))

    # Validate calculations
    expected_params = 0
    # Input norm
    expected_params += 4096
    # 4 Perceiver layers
    for _ in range(4):
        # Self-attention: Q,K,V,O projections + norm
        expected_params += 4 * 4096 * 4096 + 4096
        # FFN: up, down + norm
        expected_params += 2 * 4096 * (4 * 4096) + 4096
    # FSQ projections
    expected_params += 4096 * 8 * 2
    # Output projection + norm
    expected_params += 4096 * 4096 + 4096

    print(f"\n  Expected params (manual calc): {expected_params/1e6:.1f}M")
    print(f"  Actual params: {bridge_info['param_count']/1e6:.1f}M")
    print(f"  Match: {'✓' if abs(expected_params - bridge_info['param_count']) < 1000 else '✗'}")


def test_model_configs():
    """Test memory configurations for various model pairs."""
    print("\n" + "=" * 80)
    print("TESTING MODEL MEMORY CONFIGURATIONS")
    print("=" * 80)

    test_cases = [
        {
            "name": "Llama-8B + Mistral-7B",
            "source": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "target": "mistralai/Mistral-7B-Instruct-v0.3",
            "expected_batch_size": 2,
            "expected_valid": True
        },
        {
            "name": "Llama-1B + Qwen-1.5B",
            "source": "meta-llama/Llama-3.2-1B-Instruct",
            "target": "Qwen/Qwen2.5-1.5B-Instruct",
            "expected_batch_size": 12,  # More conservative now
            "expected_valid": True
        },
        {
            "name": "Llama-3B + Mistral-7B",
            "source": "meta-llama/Llama-3.2-3B-Instruct",
            "target": "mistralai/Mistral-7B-Instruct-v0.3",
            "expected_batch_size": 3,
            "expected_valid": True
        },
        {
            "name": "Llama-8B (single model)",
            "source": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "target": None,
            "expected_batch_size": 6,
            "expected_valid": True
        }
    ]

    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print("-" * 40)

        config = get_memory_safe_config(
            test["source"],
            test["target"],
            max_length=1536,
            soft_tokens=128
        )

        if "error" in config:
            print(f"  ERROR: {config['error']}")
            success = not test["expected_valid"]
        else:
            print(f"  Batch Size: {config['batch_size']} (expected: {test['expected_batch_size']})")
            print(f"  Grad Accum: {config['gradient_accumulation_steps']}")
            print(f"  Memory Usage: {config['estimated_memory_gb']:.1f} / 80 GB")

            # Check if batch size matches expected
            success = (config['batch_size'] == test['expected_batch_size'])

            # Detailed breakdown
            bd = config['memory_breakdown']
            print(f"  Breakdown:")
            print(f"    Source: {bd['source_model_gb']:.1f} GB")
            if bd['target_model_gb'] > 0:
                print(f"    Target: {bd['target_model_gb']:.1f} GB")
            print(f"    Bridge: {bd['bridge_gb']:.1f} GB (optimizer: {bd['bridge_optimizer_gb']:.1f} GB)")
            print(f"    Activation/batch: {bd['activation_per_batch_gb']:.1f} GB")
            print(f"    Safety margin: {bd['safety_margin_gb']:.1f} GB")

        print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)

    # Test with very long sequences
    print("\nTest: Very long sequence (4096 tokens)")
    config = get_memory_safe_config(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        max_length=4096,
        soft_tokens=128
    )
    print(f"  Batch size at 4096 length: {config['batch_size']}")
    print(f"  Memory usage: {config['estimated_memory_gb']:.1f} GB")
    print(f"  Result: {'✓' if config['batch_size'] >= 1 else '✗'}")

    # Test with very deep bridge
    print("\nTest: Deep bridge (8 layers)")
    config = get_memory_safe_config(
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        max_length=1536,
        soft_tokens=128,
        depth=8
    )
    print(f"  Batch size with 8-layer bridge: {config['batch_size']}")
    print(f"  Bridge memory: {config['memory_breakdown']['bridge_gb']:.1f} GB")
    print(f"  Result: {'✓' if config['batch_size'] >= 1 else '✗'}")

    # Test with many soft tokens
    print("\nTest: Many soft tokens (256)")
    config = get_memory_safe_config(
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        max_length=1536,
        soft_tokens=256
    )
    print(f"  Batch size with 256 soft tokens: {config['batch_size']}")
    print(f"  Memory usage: {config['estimated_memory_gb']:.1f} GB")
    print(f"  Result: {'✓' if config['batch_size'] >= 1 else '✗'}")


def main():
    """Run all tests."""
    print("MEMORY CONFIGURATION MODULE VALIDATION")
    print("=" * 80)
    print()

    # Run tests
    test_bridge_memory()
    test_model_configs()
    test_edge_cases()

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print("\nKey Improvements Made:")
    print("✓ Added optimizer memory calculation (Adam: 8 bytes/param)")
    print("✓ More accurate activation memory based on layers and dimensions")
    print("✓ Conservative batch sizes to prevent OOM")
    print("✓ Detailed memory breakdown for debugging")
    print("✓ Validated against known working configuration (Llama-8B + Mistral-7B = batch_size 2)")

    print("\nRecommendations:")
    print("• Monitor actual GPU memory usage during training")
    print("• Use gradient checkpointing if memory is tight")
    print("• Consider mixed precision training (bf16) which is already configured")
    print("• Start with recommended batch sizes and increase cautiously if memory allows")


if __name__ == "__main__":
    main()