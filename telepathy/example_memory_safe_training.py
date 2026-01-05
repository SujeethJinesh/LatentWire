#!/usr/bin/env python3
# telepathy/example_memory_safe_training.py
"""
Example of using memory_configs.py to set safe training parameters.

This script demonstrates how to integrate the memory configuration generator
into your training workflow to prevent OOM errors on 80GB H100 GPUs.
"""
import argparse
from memory_configs import get_memory_safe_config


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-safe telepathy training example")

    # Model selection
    parser.add_argument("--source_model",
                        default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Source model name/path")
    parser.add_argument("--target_model",
                        default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Target model name/path (None for single-model)")

    # Bridge configuration
    parser.add_argument("--soft_tokens", type=int, default=128,
                        help="Number of soft tokens for bridge")
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of Perceiver layers in bridge")

    # Sequence configuration
    parser.add_argument("--max_length", type=int, default=1536,
                        help="Maximum sequence length")

    # GPU configuration
    parser.add_argument("--gpu_memory_gb", type=float, default=80.0,
                        help="Available GPU memory in GB")
    parser.add_argument("--safety_margin", type=float, default=0.2,
                        help="Safety margin (0.2 = 20% buffer)")

    # Training configuration (will be overridden by memory config)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto-configured if None)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation (auto-configured if None)")

    # Other training params
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Training steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("MEMORY-SAFE TRAINING CONFIGURATION")
    print("="*80)

    # Get memory-safe configuration
    config = get_memory_safe_config(
        source_model=args.source_model,
        target_model=args.target_model,
        max_length=args.max_length,
        soft_tokens=args.soft_tokens,
        depth=args.depth,
        gpu_memory_gb=args.gpu_memory_gb,
        safety_margin=args.safety_margin,
    )

    # Check if configuration is valid
    if "error" in config:
        print(f"ERROR: {config['error']}")
        print(f"The selected models are too large for {args.gpu_memory_gb}GB GPU memory.")
        return

    # Override batch size and gradient accumulation if not manually set
    if args.batch_size is None:
        args.batch_size = config['batch_size']
        print(f"Auto-configured batch_size: {args.batch_size}")
    else:
        print(f"Using manual batch_size: {args.batch_size} (warning: may OOM)")

    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = config['gradient_accumulation_steps']
        print(f"Auto-configured gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    else:
        print(f"Using manual gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    # Display configuration
    print("\nModel Configuration:")
    print(f"  Source: {args.source_model}")
    if args.target_model:
        print(f"  Target: {args.target_model}")
    else:
        print(f"  Target: None (single-model bridge)")

    print("\nBridge Configuration:")
    print(f"  Soft tokens: {args.soft_tokens}")
    print(f"  Perceiver depth: {args.depth}")

    print("\nTraining Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Training steps: {args.steps}")
    print(f"  Warmup steps: {args.warmup_steps}")

    print("\nMemory Estimate:")
    print(f"  Estimated usage: {config['estimated_memory_gb']:.1f} GB / {args.gpu_memory_gb} GB")
    print(f"  Safety margin: {args.safety_margin * 100:.0f}%")

    if "memory_breakdown" in config:
        breakdown = config["memory_breakdown"]
        print("\nDetailed Memory Breakdown:")
        print(f"  Source model: {breakdown['source_model_gb']:.1f} GB")
        if breakdown.get('target_model_gb', 0) > 0:
            print(f"  Target model: {breakdown['target_model_gb']:.1f} GB")
        print(f"  Bridge: {breakdown['bridge_gb']:.1f} GB")
        print(f"  Activations: {breakdown['total_activation_gb']:.1f} GB")
        print(f"  Total: {breakdown['total_estimated_gb']:.1f} GB")

    print("\n" + "="*80)
    print("SLURM SCRIPT RECOMMENDATION")
    print("="*80)

    # Generate SLURM recommendations based on memory requirements
    if config['estimated_memory_gb'] > 40:
        print("Recommended SLURM settings for high memory usage:")
        print("  #SBATCH --mem=256GB")
        print("  #SBATCH --gpus=4  # Use all 4 GPUs for safety")
    else:
        print("Recommended SLURM settings for moderate memory usage:")
        print("  #SBATCH --mem=128GB")
        print("  #SBATCH --gpus=4  # Can potentially use fewer GPUs")

    print("\nExample training command for HPC:")
    print(f"""
python telepathy/train_telepathy_v15.py \\
    --source_model "{args.source_model}" \\
    --target_model "{args.target_model}" \\
    --batch_size {args.batch_size} \\
    --grad_accum {args.gradient_accumulation_steps} \\
    --soft_tokens {args.soft_tokens} \\
    --depth {args.depth} \\
    --lr {args.lr} \\
    --steps {args.steps} \\
    --warmup_steps {args.warmup_steps}
    """)

    print("\n" + "="*80)
    print("Ready for training with memory-safe configuration!")
    print("="*80)


if __name__ == "__main__":
    main()