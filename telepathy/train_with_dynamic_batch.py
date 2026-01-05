#!/usr/bin/env python3
"""Training script with dynamic batch size optimization.

This script integrates the dynamic batch size optimizer with the LatentWire
training pipeline to automatically find and use optimal batch sizes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse

from telepathy.dynamic_batch_size import (
    DynamicBatchSizeOptimizer,
    BatchSizeConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def integrate_with_latentwire_training(args):
    """Integrate dynamic batch size optimization with LatentWire training.

    Args:
        args: Training arguments
    """
    from latentwire.train import main as train_main
    from latentwire.models import InterlinguaEncoder, Adapter, LMWrapper
    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("LatentWire Training with Dynamic Batch Size Optimization")
    print("=" * 60)

    # Step 1: Load a minimal version of the model to test batch sizes
    print("\nStep 1: Loading model for batch size optimization...")

    if args.optimize_for_model == "llama":
        model_id = args.llama_id
        model_size_gb = 14.0  # Llama-8B
    else:
        model_id = args.qwen_id
        model_size_gb = 13.0  # Qwen-7B

    print(f"Optimizing for: {model_id}")

    # Step 2: Configure batch size optimizer
    config = BatchSizeConfig(
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        initial_batch_size=args.initial_batch_size,
        sequence_length=args.sequence_length,
        model_size_gb=model_size_gb,
        memory_fraction=args.memory_fraction,
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision
    )

    optimizer = DynamicBatchSizeOptimizer(config)

    # Step 3: Create minimal components for testing
    print("\nStep 2: Creating minimal components for testing...")

    encoder = InterlinguaEncoder(
        encoder_type=args.encoder_type,
        latent_len=args.latent_len,
        d_z=args.d_z,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    adapter = Adapter(
        latent_len=args.latent_len,
        d_z=args.d_z,
        d_llm=4096 if "llama" in model_id.lower() else 3584  # Approximate hidden sizes
    )

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        adapter = adapter.cuda()

    # Step 4: Find optimal batch size
    print("\nStep 3: Finding optimal batch size...")

    def forward_fn(input_ids):
        """Simulate training forward pass."""
        batch_size, seq_len = input_ids.shape

        # Simulate byte encoding
        dummy_bytes = torch.randint(0, 256, (batch_size, seq_len * 4), device=input_ids.device)

        # Encode to latent
        z = encoder(dummy_bytes)

        # Adapt to model dimension
        adapted = adapter(z)

        # Simulate loss computation
        loss = adapted.mean()

        return loss

    optimal_batch_size = optimizer.find_optimal_batch_size(
        encoder,
        forward_fn=forward_fn,
        binary_search=True
    )

    print(f"\n✓ Optimal batch size found: {optimal_batch_size}")
    print(f"  Per-GPU batch size: {optimal_batch_size // optimizer.num_gpus if optimizer.num_gpus > 1 else optimal_batch_size}")

    # Step 5: Calculate gradient accumulation if needed
    if args.target_batch_size and args.target_batch_size > optimal_batch_size:
        grad_accum_steps = optimizer.get_gradient_accumulation_steps(args.target_batch_size)
        effective_batch_size = optimal_batch_size * grad_accum_steps
        print(f"\n✓ Gradient accumulation configuration:")
        print(f"  Target batch size: {args.target_batch_size}")
        print(f"  Actual batch size: {optimal_batch_size}")
        print(f"  Gradient accumulation steps: {grad_accum_steps}")
        print(f"  Effective batch size: {effective_batch_size}")
    else:
        grad_accum_steps = 1
        effective_batch_size = optimal_batch_size

    # Step 6: Generate training configuration
    training_config = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "optimal_batch_size": optimal_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "num_gpus": optimizer.num_gpus,
        "sequence_length": args.sequence_length,
        "memory_fraction_target": args.memory_fraction,
        "optimization_summary": optimizer.get_summary()
    }

    # Save configuration
    config_path = Path(args.output_dir) / "batch_size_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"\n✓ Configuration saved to: {config_path}")

    # Step 7: Generate training command
    print("\n" + "=" * 60)
    print("Recommended Training Command:")
    print("=" * 60)

    training_cmd = f"""
python latentwire/train.py \\
    --llama_id {args.llama_id} \\
    --qwen_id {args.qwen_id} \\
    --samples {args.samples} \\
    --epochs {args.epochs} \\
    --batch_size {optimal_batch_size} \\
    --gradient_accumulation_steps {grad_accum_steps} \\
    --latent_len {args.latent_len} \\
    --d_z {args.d_z} \\
    --encoder_type {args.encoder_type} \\
    --dataset {args.dataset} \\
    --sequential_models \\
    --warm_anchor_text "Answer: " \\
    --first_token_ce_weight 0.5 \\
    --output_dir {args.output_dir}
    """

    print(training_cmd)

    # Step 8: Optionally start training immediately
    if args.start_training:
        print("\nStarting training with optimized batch size...")
        # Note: In practice, you would modify the train.py arguments here
        # For now, we just show the command
        print("(Training would start here with the optimized settings)")

    return training_config


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Train with dynamic batch size optimization")

    # Model configuration
    parser.add_argument("--llama_id", type=str,
                       default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Llama model ID")
    parser.add_argument("--qwen_id", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Qwen model ID")
    parser.add_argument("--optimize_for_model", type=str, default="llama",
                       choices=["llama", "qwen"],
                       help="Which model to optimize batch size for")

    # Training configuration
    parser.add_argument("--samples", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--latent_len", type=int, default=32,
                       help="Latent sequence length")
    parser.add_argument("--d_z", type=int, default=256,
                       help="Latent dimension")
    parser.add_argument("--encoder_type", type=str, default="byte",
                       help="Encoder type")
    parser.add_argument("--dataset", type=str, default="squad",
                       help="Dataset to use")

    # Batch size optimization
    parser.add_argument("--min_batch_size", type=int, default=1,
                       help="Minimum batch size to test")
    parser.add_argument("--max_batch_size", type=int, default=256,
                       help="Maximum batch size to test")
    parser.add_argument("--initial_batch_size", type=int, default=8,
                       help="Initial batch size for optimization")
    parser.add_argument("--sequence_length", type=int, default=512,
                       help="Sequence length for optimization")
    parser.add_argument("--memory_fraction", type=float, default=0.9,
                       help="Target GPU memory utilization (0-1)")
    parser.add_argument("--target_batch_size", type=int, default=None,
                       help="Target effective batch size (uses gradient accumulation if needed)")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training")

    # Output and execution
    parser.add_argument("--output_dir", type=str, default="runs/dynamic_batch",
                       help="Output directory for results")
    parser.add_argument("--start_training", action="store_true",
                       help="Start training immediately after optimization")

    args = parser.parse_args()

    # Check for GPU availability
    if not torch.cuda.is_available():
        print("Warning: No GPU available. Dynamic batch size optimization requires GPU.")
        print("Please run this on a machine with GPU support.")
        return

    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")

    # Run optimization and optionally start training
    try:
        config = integrate_with_latentwire_training(args)

        print("\n" + "=" * 60)
        print("Optimization Complete!")
        print("=" * 60)
        print(f"\nOptimal configuration:")
        print(f"  Batch size: {config['optimal_batch_size']}")
        print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(f"  Effective batch size: {config['effective_batch_size']}")
        print(f"\nConfiguration saved to: {args.output_dir}/batch_size_config.json")

    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()