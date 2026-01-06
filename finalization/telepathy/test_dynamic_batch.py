#!/usr/bin/env python3
"""Test script for dynamic batch size optimizer with LatentWire models.

This script demonstrates how to integrate the dynamic batch size optimizer
with the LatentWire training pipeline to automatically find optimal batch sizes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Dict, Any
import json
from datetime import datetime

from telepathy.dynamic_batch_size import (
    DynamicBatchSizeOptimizer,
    BatchSizeConfig,
    optimize_batch_size_for_model
)


def test_with_latentwire_model():
    """Test dynamic batch size optimization with actual LatentWire components."""

    print("=" * 60)
    print("Testing Dynamic Batch Size Optimizer with LatentWire")
    print("=" * 60)

    try:
        from latentwire.models import InterlinguaInterlinguaEncoder, Adapter

        # Test configuration
        latent_len = 32
        d_z = 256
        vocab_size = 32000  # Llama vocab size
        d_model = 4096  # Llama hidden size

        # Create encoder and adapter (smaller components for testing)
        encoder = InterlinguaEncoder(
            encoder_type="byte",
            latent_len=latent_len,
            d_z=d_z,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        adapter = Adapter(
            latent_len=latent_len,
            d_z=d_z,
            d_llm=d_model
        )

        if torch.cuda.is_available():
            encoder = encoder.cuda()
            adapter = adapter.cuda()

        print(f"\nCreated LatentWire components:")
        print(f"  Encoder: {encoder.encoder_type}")
        print(f"  Latent dimensions: {latent_len} x {d_z}")
        print(f"  Adapter output: {d_model}")

        # Configure batch size optimizer
        config = BatchSizeConfig(
            min_batch_size=1,
            max_batch_size=256,
            initial_batch_size=8,
            sequence_length=512,
            memory_fraction=0.9,
            gradient_accumulation_steps=1
        )

        optimizer = DynamicBatchSizeOptimizer(config)

        # Define a forward function that mimics training
        def forward_fn(input_ids):
            batch_size, seq_len = input_ids.shape

            # Simulate byte encoding
            dummy_bytes = torch.randint(0, 256, (batch_size, seq_len * 4), device=input_ids.device)

            # Encode to latent
            z = encoder(dummy_bytes)

            # Adapt to model dimension
            adapted = adapter(z)

            # Simulate model output (much smaller than full LLM)
            output = torch.randn(batch_size, seq_len, vocab_size, device=input_ids.device)

            return output

        # Find optimal batch size
        print("\nSearching for optimal batch size...")
        optimal_batch_size = optimizer.find_optimal_batch_size(
            encoder,  # Use encoder as the model
            forward_fn=forward_fn,
            binary_search=True
        )

        print(f"\nResults:")
        print(f"  Optimal batch size: {optimal_batch_size}")
        print(f"  Effective batch size: {optimizer.effective_batch_size}")

        # Test different sequence lengths
        print("\nAdjusting for different sequence lengths:")
        for seq_len in [256, 512, 1024, 2048]:
            adjusted = optimizer.adjust_for_sequence_length(seq_len)
            print(f"  Seq length {seq_len}: batch size {adjusted}")

        # Test gradient accumulation
        print("\nGradient accumulation for target batch sizes:")
        for target in [64, 128, 256, 512]:
            grad_steps = optimizer.get_gradient_accumulation_steps(target)
            actual = optimal_batch_size * grad_steps
            print(f"  Target: {target}, Grad steps: {grad_steps}, Actual: {actual}")

        # Get summary
        summary = optimizer.get_summary()

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "optimal_batch_size": optimal_batch_size,
            "effective_batch_size": optimizer.effective_batch_size,
            "summary": summary,
            "test_type": "latentwire_components"
        }

        output_file = "telepathy/batch_size_optimization_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    except ImportError as e:
        print(f"Could not import LatentWire components: {e}")
        print("Running fallback test with dummy model...")
        test_with_dummy_model()

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_with_dummy_model():
    """Test with a simple dummy model (fallback if LatentWire not available)."""

    print("\n" + "=" * 60)
    print("Testing with Dummy Model")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No GPU available. Skipping test.")
        return

    # Create a simple dummy model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=4096):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = SimpleModel().cuda()

    # Configure optimizer
    config = BatchSizeConfig(
        min_batch_size=1,
        max_batch_size=128,
        sequence_length=512,
        memory_fraction=0.9
    )

    optimizer = DynamicBatchSizeOptimizer(config)

    # Find optimal batch size
    print("\nSearching for optimal batch size...")
    optimal = optimizer.find_optimal_batch_size(model, binary_search=True)

    print(f"\nOptimal batch size: {optimal}")

    # Get summary
    summary = optimizer.get_summary()
    print(f"\nSummary:")
    print(f"  GPUs: {summary['num_gpus']}")
    print(f"  Optimal batch size: {summary['optimal_batch_size']}")
    print(f"  Successful tests: {summary['successful_batch_sizes']}")


def test_model_comparison():
    """Compare optimal batch sizes for different models."""

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    models_to_test = [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", 512),
        ("Qwen/Qwen2.5-7B-Instruct", 512),
    ]

    results = {}

    for model_name, seq_length in models_to_test:
        print(f"\nOptimizing for {model_name}...")
        try:
            result = optimize_batch_size_for_model(
                model_name=model_name,
                sequence_length=seq_length,
                target_memory_fraction=0.9
            )
            results[model_name] = result
            print(f"  Optimal batch size: {result['optimal_batch_size']}")
        except Exception as e:
            print(f"  Error: {e}")

    # Save comparison results
    if results:
        comparison_file = "telepathy/batch_size_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nComparison results saved to: {comparison_file}")


def main():
    """Main test function."""

    import argparse

    parser = argparse.ArgumentParser(description="Test dynamic batch size optimizer")
    parser.add_argument("--test", type=str, default="latentwire",
                       choices=["latentwire", "dummy", "comparison", "all"],
                       help="Which test to run")

    args = parser.parse_args()

    if args.test == "latentwire" or args.test == "all":
        test_with_latentwire_model()

    if args.test == "dummy" or args.test == "all":
        test_with_dummy_model()

    if args.test == "comparison" or args.test == "all":
        test_model_comparison()

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()