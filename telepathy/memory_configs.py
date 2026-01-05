#!/usr/bin/env python
# telepathy/memory_configs.py
"""
Memory-safe configuration generator for telepathy bridge training.

Calculates safe batch sizes and gradient accumulation steps for different
model combinations on 80GB H100 GPUs, accounting for:
- Model parameter memory (bfloat16 = 2 bytes/param)
- Activation memory during forward/backward passes
- Gradient memory for trainable parameters
- Bridge architecture overhead
- 20% safety margin to prevent OOM

Known working configs:
- Llama-8B + Mistral-7B: batch_size=2, grad_accum=4
"""
import math
from typing import Dict, Tuple


# Model parameter counts (in billions)
MODEL_PARAMS = {
    # Llama models
    "meta-llama/Llama-3.2-1B": 1.24,
    "meta-llama/Llama-3.2-1B-Instruct": 1.24,
    "meta-llama/Llama-3.2-3B": 3.21,
    "meta-llama/Llama-3.2-3B-Instruct": 3.21,
    "meta-llama/Meta-Llama-3.1-8B": 8.03,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 8.03,

    # Mistral models
    "mistralai/Mistral-7B-v0.3": 7.24,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.24,

    # Qwen models
    "Qwen/Qwen2.5-1.5B": 1.54,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.54,
    "Qwen/Qwen2.5-3B": 3.09,
    "Qwen/Qwen2.5-3B-Instruct": 3.09,
    "Qwen/Qwen2.5-7B": 7.62,
    "Qwen/Qwen2.5-7B-Instruct": 7.62,
}

# Hidden dimensions for models (needed for bridge sizing)
MODEL_HIDDEN_DIM = {
    # Llama models
    "meta-llama/Llama-3.2-1B": 2048,
    "meta-llama/Llama-3.2-1B-Instruct": 2048,
    "meta-llama/Llama-3.2-3B": 3072,
    "meta-llama/Llama-3.2-3B-Instruct": 3072,
    "meta-llama/Meta-Llama-3.1-8B": 4096,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 4096,

    # Mistral models
    "mistralai/Mistral-7B-v0.3": 4096,
    "mistralai/Mistral-7B-Instruct-v0.3": 4096,

    # Qwen models
    "Qwen/Qwen2.5-1.5B": 1536,
    "Qwen/Qwen2.5-1.5B-Instruct": 1536,
    "Qwen/Qwen2.5-3B": 2048,
    "Qwen/Qwen2.5-3B-Instruct": 2048,
    "Qwen/Qwen2.5-7B": 3584,
    "Qwen/Qwen2.5-7B-Instruct": 3584,
}


def estimate_bridge_memory_gb(
    source_dim,
    target_dim,
    soft_tokens=128,
    depth=4,
    heads=8,
    fsq_levels=8,
    fsq_dims=8,
):
    """
    Estimate memory usage for the bridge architecture.

    Args:
        source_dim: Hidden dimension of source model
        target_dim: Hidden dimension of target model
        soft_tokens: Number of soft tokens (K)
        depth: Number of Perceiver layers
        heads: Number of attention heads
        fsq_levels: Quantization levels per FSQ dimension
        fsq_dims: Number of FSQ dimensions

    Returns:
        Estimated memory in GB for bridge components
    """
    params = 0

    # Input normalizer
    params += source_dim  # LayerNorm

    # Perceiver layers
    perceiver_dim = source_dim  # Assuming same dim
    for _ in range(depth):
        # Self-attention
        params += 4 * perceiver_dim * perceiver_dim  # Q, K, V, O projections
        params += perceiver_dim  # LayerNorm

        # FFN
        ffn_dim = 4 * perceiver_dim
        params += perceiver_dim * ffn_dim * 2  # Up and down projections
        params += perceiver_dim  # LayerNorm

    # FSQ projections
    params += source_dim * fsq_dims  # Project down
    params += fsq_dims * source_dim  # Project up

    # Transformer projector to target dimension
    params += source_dim * target_dim  # Linear projection
    params += target_dim  # LayerNorm

    # Convert to GB (bfloat16 = 2 bytes per param)
    bridge_gb = (params * 2) / (1024**3)

    # Add activation memory overhead (roughly 2-3x params during training)
    activation_overhead = 2.5
    total_bridge_gb = bridge_gb * activation_overhead

    return total_bridge_gb


def get_memory_safe_config(
    source_model,
    target_model=None,
    max_length=1536,
    soft_tokens=128,
    depth=4,
    gpu_memory_gb=80.0,
    safety_margin=0.2,
    force_single_model=False,
):
    """
    Calculate memory-safe configuration for training.

    Args:
        source_model: Name/path of source model
        target_model: Name/path of target model (None for single model)
        max_length: Maximum sequence length
        soft_tokens: Number of soft tokens for bridge
        depth: Number of Perceiver layers in bridge
        gpu_memory_gb: Available GPU memory in GB
        safety_margin: Safety margin (0.2 = 20% buffer)
        force_single_model: Force single-model mode even if target provided

    Returns:
        Dictionary with:
            - batch_size: Safe batch size
            - gradient_accumulation_steps: Gradient accumulation steps
            - max_length: Maximum sequence length
            - dual_model_possible: Whether dual-model training is possible
            - estimated_memory_gb: Estimated memory usage
            - memory_breakdown: Detailed memory breakdown
    """
    # Get model sizes
    if source_model not in MODEL_PARAMS:
        raise ValueError(f"Unknown source model: {source_model}")

    source_params_b = MODEL_PARAMS[source_model]
    source_dim = MODEL_HIDDEN_DIM[source_model]

    # Check if dual-model training
    dual_model = target_model is not None and not force_single_model

    if dual_model:
        if target_model not in MODEL_PARAMS:
            raise ValueError(f"Unknown target model: {target_model}")
        target_params_b = MODEL_PARAMS[target_model]
        target_dim = MODEL_HIDDEN_DIM[target_model]
    else:
        target_params_b = 0
        target_dim = source_dim

    # Calculate base memory requirements (GB)
    source_mem_gb = source_params_b * 2  # bfloat16 = 2 bytes/param
    target_mem_gb = target_params_b * 2 if dual_model else 0

    # Estimate bridge memory
    bridge_mem_gb = estimate_bridge_memory_gb(
        source_dim=source_dim,
        target_dim=target_dim,
        soft_tokens=soft_tokens,
        depth=depth,
    )

    # Base memory (models + bridge)
    base_memory_gb = source_mem_gb + target_mem_gb + bridge_mem_gb

    # Activation memory scales with batch size and sequence length
    # Rough estimate: ~4-6 GB per batch for 8B model at 1.5K length
    if source_params_b + target_params_b > 12:
        activation_gb_per_batch = 6.0
    elif source_params_b + target_params_b > 8:
        activation_gb_per_batch = 5.0
    elif source_params_b + target_params_b > 4:
        activation_gb_per_batch = 4.0
    else:
        activation_gb_per_batch = 3.0

    # Scale by sequence length
    activation_gb_per_batch *= (max_length / 1536)

    # Available memory after base requirements and safety margin
    available_memory_gb = gpu_memory_gb * (1 - safety_margin)
    remaining_gb = available_memory_gb - base_memory_gb

    # Calculate safe batch size
    if remaining_gb <= 0:
        # Cannot fit even base models
        return {
            "batch_size": 0,
            "gradient_accumulation_steps": 0,
            "max_length": max_length,
            "dual_model_possible": False,
            "estimated_memory_gb": base_memory_gb,
            "memory_breakdown": {
                "source_model_gb": source_mem_gb,
                "target_model_gb": target_mem_gb,
                "bridge_gb": bridge_mem_gb,
                "total_base_gb": base_memory_gb,
                "available_gb": available_memory_gb,
            },
            "error": "Models too large for available GPU memory"
        }

    # Calculate maximum batch size
    max_batch_size = int(remaining_gb / activation_gb_per_batch)

    # Apply heuristics based on known working configs
    if dual_model:
        # Dual-model configs
        if source_params_b + target_params_b >= 15:  # e.g., 8B + 7B
            batch_size = min(max_batch_size, 2)
            grad_accum = 4
        elif source_params_b + target_params_b >= 10:  # e.g., 7B + 3B
            batch_size = min(max_batch_size, 4)
            grad_accum = 2
        elif source_params_b + target_params_b >= 5:  # e.g., 3B + 3B
            batch_size = min(max_batch_size, 8)
            grad_accum = 1
        else:  # Small models
            batch_size = min(max_batch_size, 16)
            grad_accum = 1
    else:
        # Single-model configs (more memory available)
        if source_params_b >= 7:
            batch_size = min(max_batch_size, 8)
            grad_accum = 2
        elif source_params_b >= 3:
            batch_size = min(max_batch_size, 16)
            grad_accum = 1
        else:
            batch_size = min(max_batch_size, 32)
            grad_accum = 1

    # Ensure at least batch_size=1
    if batch_size < 1:
        batch_size = 1
        grad_accum = 8  # Compensate with more accumulation

    # Estimate final memory usage
    estimated_memory_gb = base_memory_gb + (batch_size * activation_gb_per_batch)

    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_length": max_length,
        "dual_model_possible": dual_model,
        "estimated_memory_gb": estimated_memory_gb,
        "memory_breakdown": {
            "source_model_gb": source_mem_gb,
            "target_model_gb": target_mem_gb,
            "bridge_gb": bridge_mem_gb,
            "activation_per_batch_gb": activation_gb_per_batch,
            "total_activation_gb": batch_size * activation_gb_per_batch,
            "total_base_gb": base_memory_gb,
            "total_estimated_gb": estimated_memory_gb,
            "available_gb": available_memory_gb,
            "safety_margin_gb": gpu_memory_gb * safety_margin,
        }
    }


def print_config_table(configs):
    """Print a formatted table of memory configurations."""
    print("\n" + "="*100)
    print("MEMORY-SAFE CONFIGURATIONS FOR 80GB H100")
    print("="*100)

    for (source, target), config in configs.items():
        model_desc = f"{source.split('/')[-1]} → {target.split('/')[-1]}" if target else source.split('/')[-1]

        if "error" in config:
            print(f"\n{model_desc}:")
            print(f"  ERROR: {config['error']}")
            continue

        print(f"\n{model_desc}:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']}")
        print(f"  Effective Batch Size: {config['batch_size'] * config['gradient_accumulation_steps']}")
        print(f"  Max Sequence Length: {config['max_length']}")
        print(f"  Estimated Memory: {config['estimated_memory_gb']:.1f} GB / 80 GB")

        if "memory_breakdown" in config:
            breakdown = config["memory_breakdown"]
            print(f"  Memory Breakdown:")
            print(f"    - Source Model: {breakdown['source_model_gb']:.1f} GB")
            if breakdown['target_model_gb'] > 0:
                print(f"    - Target Model: {breakdown['target_model_gb']:.1f} GB")
            print(f"    - Bridge: {breakdown['bridge_gb']:.1f} GB")
            print(f"    - Activations: {breakdown['total_activation_gb']:.1f} GB")


def main():
    """Generate and display memory-safe configurations for common model pairs."""

    # Test configurations
    test_configs = [
        # Dual-model configs (cross-model bridge)
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
        ("meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("meta-llama/Llama-3.2-1B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),

        # Single-model configs (same-model bridge)
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", None),
        ("mistralai/Mistral-7B-Instruct-v0.3", None),
        ("meta-llama/Llama-3.2-3B-Instruct", None),
        ("meta-llama/Llama-3.2-1B-Instruct", None),
    ]

    configs = {}
    for source, target in test_configs:
        config = get_memory_safe_config(source, target)
        configs[(source, target)] = config

    print_config_table(configs)

    # Example usage in code
    print("\n" + "="*100)
    print("EXAMPLE USAGE IN TRAINING SCRIPT:")
    print("="*100)
    print("""
from telepathy.memory_configs import get_memory_safe_config

# Get safe config for your model pair
config = get_memory_safe_config(
    source_model="meta-llama/Llama-3.2-3B-Instruct",
    target_model="mistralai/Mistral-7B-Instruct-v0.3",
    max_length=1536,
    soft_tokens=128,
)

# Use in training
args.batch_size = config['batch_size']
args.gradient_accumulation_steps = config['gradient_accumulation_steps']
args.max_length = config['max_length']

print(f"Training with batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
print(f"Estimated memory usage: {config['estimated_memory_gb']:.1f} GB")
    """)


if __name__ == "__main__":
    main()