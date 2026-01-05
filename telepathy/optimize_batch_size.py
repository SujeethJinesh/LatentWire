#!/usr/bin/env python3
"""
Automatic batch size optimizer for cross-model telepathy training.
Calculates maximum safe batch sizes based on model combinations and available GPU memory.
"""

import argparse
import json
import math
# Removed type hints for Python 3 compatibility
import sys

# Model memory estimates (in GB) based on empirical measurements
# These include model weights + typical activation memory per batch item
MODEL_BASE_MEMORY = {
    # Llama family
    "meta-llama/Llama-3.2-1B-Instruct": 2.5,
    "meta-llama/Llama-3.2-3B-Instruct": 7.0,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 16.0,
    "meta-llama/Meta-Llama-3-8B-Instruct": 16.0,

    # Qwen family
    "Qwen/Qwen2.5-0.5B-Instruct": 1.2,
    "Qwen/Qwen2.5-1.5B-Instruct": 3.5,
    "Qwen/Qwen2.5-3B-Instruct": 7.0,
    "Qwen/Qwen2.5-7B-Instruct": 14.0,

    # Mistral family
    "mistralai/Mistral-7B-Instruct-v0.3": 14.0,
    "mistralai/Ministral-8B-Instruct-2410": 16.5,

    # Phi family
    "microsoft/phi-2": 5.5,
    "microsoft/Phi-3.5-mini-instruct": 7.5,
}

# Memory per batch item (in GB) - includes activations, gradients, optimizer states
MEMORY_PER_BATCH_ITEM = {
    # Smaller models (< 2B params)
    "meta-llama/Llama-3.2-1B-Instruct": 0.25,
    "Qwen/Qwen2.5-0.5B-Instruct": 0.15,

    # Medium models (2-5B params)
    "meta-llama/Llama-3.2-3B-Instruct": 0.4,
    "Qwen/Qwen2.5-1.5B-Instruct": 0.3,
    "Qwen/Qwen2.5-3B-Instruct": 0.4,
    "microsoft/phi-2": 0.35,

    # Larger models (7-8B params)
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 0.6,
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.6,
    "Qwen/Qwen2.5-7B-Instruct": 0.55,
    "mistralai/Mistral-7B-Instruct-v0.3": 0.55,
    "mistralai/Ministral-8B-Instruct-2410": 0.65,
    "microsoft/Phi-3.5-mini-instruct": 0.5,
}

# Additional memory for telepathy components
TELEPATHY_OVERHEAD = {
    "encoder": 0.5,  # Encoder network
    "adapters": 0.3,  # Per-model adapters
    "buffers": 1.0,  # Intermediate buffers
    "pytorch": 2.0,  # PyTorch framework overhead
}


def get_model_memory_requirements(
    source_model,
    target_model,
    batch_size,
    sequence_length=512,
    latent_len=128,
    d_z=768,
):
    """
    Calculate memory requirements for a given model pair and batch size.

    Args:
        source_model: Source model name/ID
        target_model: Target model name/ID
        batch_size: Batch size to evaluate
        sequence_length: Maximum sequence length
        latent_len: Number of latent tokens
        d_z: Latent dimension

    Returns:
        Dict with memory breakdown in GB
    """
    # Get base memory for models
    source_base = MODEL_BASE_MEMORY.get(source_model, 16.0)  # Default to 8B size
    target_base = MODEL_BASE_MEMORY.get(target_model, 16.0)

    # Get per-batch memory
    source_per_batch = MEMORY_PER_BATCH_ITEM.get(source_model, 0.6)
    target_per_batch = MEMORY_PER_BATCH_ITEM.get(target_model, 0.6)

    # Calculate batch memory
    source_batch_mem = source_per_batch * batch_size
    target_batch_mem = target_per_batch * batch_size

    # Latent memory (encoder output + gradients)
    latent_memory = (batch_size * latent_len * d_z * 4 * 3) / (1024**3)  # fp32, x3 for grad+optimizer

    # Total memory breakdown
    memory = {
        "source_base": source_base,
        "target_base": target_base,
        "source_batch": source_batch_mem,
        "target_batch": target_batch_mem,
        "latent": latent_memory,
        "encoder": TELEPATHY_OVERHEAD["encoder"],
        "adapters": TELEPATHY_OVERHEAD["adapters"],
        "buffers": TELEPATHY_OVERHEAD["buffers"],
        "pytorch": TELEPATHY_OVERHEAD["pytorch"],
    }

    memory["total"] = sum(memory.values())

    return memory


def calculate_max_batch_size(
    source_model,
    target_model,
    gpu_memory=80.0,
    safety_margin=0.2,
    sequence_length=512,
    latent_len=128,
    d_z=768,
    min_batch_size=1,
    max_batch_size=256,
):
    """
    Calculate maximum safe batch size for a model pair.

    Args:
        source_model: Source model name/ID
        target_model: Target model name/ID
        gpu_memory: Available GPU memory in GB
        safety_margin: Safety margin (0.2 = 20%)
        sequence_length: Maximum sequence length
        latent_len: Number of latent tokens
        d_z: Latent dimension
        min_batch_size: Minimum batch size to consider
        max_batch_size: Maximum batch size to consider

    Returns:
        Tuple of (max_batch_size, memory_breakdown)
    """
    # Available memory after safety margin
    available_memory = gpu_memory * (1 - safety_margin)

    # Binary search for maximum batch size
    left, right = min_batch_size, max_batch_size
    best_batch_size = min_batch_size
    best_memory = {}

    while left <= right:
        mid = (left + right) // 2
        memory = get_model_memory_requirements(
            source_model, target_model, mid,
            sequence_length, latent_len, d_z
        )

        if memory["total"] <= available_memory:
            best_batch_size = mid
            best_memory = memory
            left = mid + 1
        else:
            right = mid - 1

    return best_batch_size, best_memory


def suggest_gradient_accumulation(
    actual_batch_size,
    desired_batch_size=64,
):
    """
    Suggest gradient accumulation steps to reach desired effective batch size.

    Args:
        actual_batch_size: Maximum batch size that fits in memory
        desired_batch_size: Desired effective batch size

    Returns:
        Gradient accumulation steps
    """
    if actual_batch_size >= desired_batch_size:
        return 1

    # Find accumulation steps that give us close to desired batch size
    accumulation = int(math.ceil(float(desired_batch_size) / float(actual_batch_size)))

    # Try to use powers of 2 when possible
    power_of_2 = int(2 ** math.ceil(math.log(accumulation) / math.log(2)))
    if power_of_2 * actual_batch_size <= desired_batch_size * 1.5:
        return power_of_2

    return accumulation


def main():
    parser = argparse.ArgumentParser(
        description="Calculate optimal batch size for telepathy training"
    )
    parser.add_argument(
        "--source-model",
        type=str,
        required=True,
        help="Source model name/ID"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Target model name/ID"
    )
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
        help="Safety margin as fraction (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--latent-len",
        type=int,
        default=128,
        help="Number of latent tokens (default: 128)"
    )
    parser.add_argument(
        "--d-z",
        type=int,
        default=768,
        help="Latent dimension (default: 768)"
    )
    parser.add_argument(
        "--desired-batch-size",
        type=int,
        default=64,
        help="Desired effective batch size for gradient accumulation (default: 64)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed memory breakdown"
    )

    args = parser.parse_args()

    # Calculate maximum batch size
    max_batch_size, memory_breakdown = calculate_max_batch_size(
        args.source_model,
        args.target_model,
        args.gpu_memory,
        args.safety_margin,
        args.sequence_length,
        args.latent_len,
        args.d_z,
    )

    # Calculate gradient accumulation
    grad_accum = suggest_gradient_accumulation(
        max_batch_size,
        args.desired_batch_size
    )

    effective_batch_size = max_batch_size * grad_accum

    # Prepare output
    result = {
        "source_model": args.source_model,
        "target_model": args.target_model,
        "max_batch_size": max_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch_size,
        "gpu_memory_gb": args.gpu_memory,
        "safety_margin": args.safety_margin,
        "estimated_memory_usage_gb": memory_breakdown["total"],
        "memory_utilization": memory_breakdown["total"] / args.gpu_memory,
    }

    if args.verbose:
        result["memory_breakdown"] = memory_breakdown

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*60)
        print("Batch Size Optimization Results")
        print("="*60)
        print("Model Pair: {} -> {}".format(args.source_model, args.target_model))
        print("GPU Memory: {:.1f} GB (safety margin: {:.0f}%)".format(
            args.gpu_memory, args.safety_margin*100))
        print("="*60)
        print("Maximum Batch Size: {}".format(max_batch_size))
        print("Gradient Accumulation: {} steps".format(grad_accum))
        print("Effective Batch Size: {}".format(effective_batch_size))
        print("Estimated Memory Usage: {:.1f} GB ({:.0f}%)".format(
            memory_breakdown['total'], memory_breakdown['total']/args.gpu_memory*100))

        if args.verbose:
            print("\nMemory Breakdown:")
            print("  Source Model Base: {:.1f} GB".format(memory_breakdown['source_base']))
            print("  Target Model Base: {:.1f} GB".format(memory_breakdown['target_base']))
            print("  Source Batch Memory: {:.1f} GB".format(memory_breakdown['source_batch']))
            print("  Target Batch Memory: {:.1f} GB".format(memory_breakdown['target_batch']))
            print("  Latent Memory: {:.1f} GB".format(memory_breakdown['latent']))
            print("  Encoder: {:.1f} GB".format(memory_breakdown['encoder']))
            print("  Adapters: {:.1f} GB".format(memory_breakdown['adapters']))
            print("  Buffers: {:.1f} GB".format(memory_breakdown['buffers']))
            print("  PyTorch Overhead: {:.1f} GB".format(memory_breakdown['pytorch']))

        print("="*60)
        print("\nRecommended training command additions:")
        print("  --batch_size {}".format(max_batch_size))
        if grad_accum > 1:
            print("  --gradient_accumulation_steps {}".format(grad_accum))
        print("")

    # Exit with error code if batch size is too small
    if max_batch_size < 1:
        sys.stderr.write("ERROR: Cannot fit even batch_size=1 in memory!\n")
        sys.stderr.write("Memory required: {:.1f} GB\n".format(memory_breakdown['total']))
        sys.stderr.write("Memory available: {:.1f} GB\n".format(args.gpu_memory * (1-args.safety_margin)))
        sys.exit(1)

    return result


def get_batch_size_for_models(
    source_model,
    target_model,
    gpu_memory=80.0,
    safety_margin=0.2,
    **kwargs
):
    """
    Convenience function for use in other scripts.

    Example:
        from optimize_batch_size import get_batch_size_for_models
        config = get_batch_size_for_models(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct"
        )
        batch_size = config["max_batch_size"]
    """
    max_batch_size, memory_breakdown = calculate_max_batch_size(
        source_model,
        target_model,
        gpu_memory,
        safety_margin,
        **kwargs
    )

    return {
        "max_batch_size": max_batch_size,
        "memory_breakdown": memory_breakdown,
        "gradient_accumulation_steps": suggest_gradient_accumulation(
            max_batch_size,
            kwargs.get("desired_batch_size", 64)
        )
    }


if __name__ == "__main__":
    main()