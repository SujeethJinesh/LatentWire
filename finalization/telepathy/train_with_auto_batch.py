#!/usr/bin/env python3
"""
Example training script that uses automatic batch size optimization.
Demonstrates integration with the batch size optimizer.
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path


def get_optimal_batch_size(source_model, target_model, config):
    """
    Get optimal batch size using the optimizer.

    Args:
        source_model: Source model ID
        target_model: Target model ID
        config: Additional configuration dict

    Returns:
        Dict with batch_size and gradient_accumulation_steps
    """
    cmd = [
        "python", "telepathy/optimize_batch_size.py",
        "--source-model", source_model,
        "--target-model", target_model,
        "--gpu-memory", str(config.get("gpu_memory", 80.0)),
        "--safety-margin", str(config.get("safety_margin", 0.2)),
        "--sequence-length", str(config.get("sequence_length", 512)),
        "--latent-len", str(config.get("latent_len", 128)),
        "--d-z", str(config.get("d_z", 768)),
        "--desired-batch-size", str(config.get("desired_batch_size", 64)),
        "--json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error getting batch size: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    return json.loads(result.stdout)


def build_training_command(source_model, target_model, batch_config, training_config):
    """
    Build the telepathy training command with optimal batch size.

    Args:
        source_model: Source model ID
        target_model: Target model ID
        batch_config: Output from batch size optimizer
        training_config: Additional training configuration

    Returns:
        List of command arguments
    """
    cmd = [
        "python", "telepathy/train_telepathy.py",
        "--source_model", source_model,
        "--target_model", target_model,
        "--batch_size", str(batch_config["max_batch_size"]),
        "--gradient_accumulation_steps", str(batch_config["gradient_accumulation_steps"]),
        "--latent_len", str(training_config.get("latent_len", 128)),
        "--d_z", str(training_config.get("d_z", 768)),
        "--epochs", str(training_config.get("epochs", 10)),
        "--learning_rate", str(training_config.get("learning_rate", 1e-4)),
        "--output_dir", training_config.get("output_dir", "runs/telepathy"),
        "--dataset", training_config.get("dataset", "arxiv"),
        "--samples", str(training_config.get("samples", 10000)),
    ]

    # Add optional parameters
    if training_config.get("resume_from"):
        cmd.extend(["--resume_from", training_config["resume_from"]])

    if training_config.get("eval_every"):
        cmd.extend(["--eval_every", str(training_config["eval_every"])])

    if training_config.get("save_every"):
        cmd.extend(["--save_every", str(training_config["save_every"])])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Train telepathy with automatic batch size optimization"
    )

    # Model selection
    parser.add_argument(
        "--source-model",
        type=str,
        required=True,
        help="Source model ID"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Target model ID"
    )

    # GPU configuration
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=80.0,
        help="GPU memory in GB (default: 80 for H100)"
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.2,
        help="Memory safety margin (default: 0.2 = 20%%)"
    )

    # Training configuration
    parser.add_argument(
        "--latent-len",
        type=int,
        default=128,
        help="Number of latent tokens"
    )
    parser.add_argument(
        "--d-z",
        type=int,
        default=768,
        help="Latent dimension"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--desired-batch-size",
        type=int,
        default=64,
        help="Desired effective batch size"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="arxiv",
        choices=["arxiv", "squad", "hotpotqa"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/telepathy",
        help="Output directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Telepathy Training with Auto Batch Size Optimization")
    print(f"{'='*60}")
    print(f"Source Model: {args.source_model}")
    print(f"Target Model: {args.target_model}")
    print(f"GPU Memory: {args.gpu_memory} GB")
    print(f"Safety Margin: {args.safety_margin*100:.0f}%")
    print(f"{'='*60}\n")

    # Get optimal batch size
    print("Calculating optimal batch size...")
    batch_config = get_optimal_batch_size(
        args.source_model,
        args.target_model,
        {
            "gpu_memory": args.gpu_memory,
            "safety_margin": args.safety_margin,
            "sequence_length": 512,
            "latent_len": args.latent_len,
            "d_z": args.d_z,
            "desired_batch_size": args.desired_batch_size,
        }
    )

    print(f"✓ Max Batch Size: {batch_config['max_batch_size']}")
    print(f"✓ Gradient Accumulation: {batch_config['gradient_accumulation_steps']} steps")
    print(f"✓ Effective Batch Size: {batch_config['effective_batch_size']}")
    print(f"✓ Estimated Memory: {batch_config['estimated_memory_usage_gb']:.1f} GB")
    print(f"✓ Memory Utilization: {batch_config['memory_utilization']*100:.0f}%\n")

    # Build training command
    training_config = {
        "latent_len": args.latent_len,
        "d_z": args.d_z,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "output_dir": args.output_dir,
        "dataset": args.dataset,
        "samples": args.samples,
    }

    cmd = build_training_command(
        args.source_model,
        args.target_model,
        batch_config,
        training_config
    )

    # Print or execute command
    print(f"{'='*60}")
    print("Training Command:")
    print(f"{'='*60}")
    print(" \\\n    ".join(cmd))
    print("")

    if args.dry_run:
        print("(Dry run - command not executed)")
    else:
        print("Starting training...")
        print(f"{'='*60}\n")

        # Execute training
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"\n{'='*60}")
            print("Training completed successfully!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Training failed with exit code {result.returncode}")
            print(f"{'='*60}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()