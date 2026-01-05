#!/usr/bin/env python
# telepathy/memory_configs.py
"""
Memory-safe configuration generator for telepathy bridge training.

Calculates safe batch sizes and gradient accumulation steps for different
model combinations on 80GB H100 GPUs, accounting for:
- Model parameter memory (bfloat16 = 2 bytes/param)
- Optimizer states for trainable parameters (Adam: 8 bytes/param)
- Activation memory during forward/backward passes (including attention matrices)
- Gradient memory for trainable parameters
- Bridge architecture overhead (params + gradients + optimizer)
- 25% safety margin to prevent OOM

Known working configs:
- Llama-8B + Mistral-7B: batch_size=2, grad_accum=4 (VERIFIED)

CRITICAL FIXES (2025-01):
- Added optimizer memory for model adapters (~1% params assumed trainable)
- Fixed activation memory to include attention matrices (seq_len^2 scaling)
- Increased safety margin from 20% to 25% for reliability
- Force batch_size=2 for Llama-8B + Mistral-7B (proven config)
"""
import math
from typing import Dict, Tuple


# Model parameter counts (in billions)
MODEL_PARAMS = {
    # Llama models
    "meta-llama/Llama-3.1-1B": 1.24,
    "meta-llama/Llama-3.1-1B-Instruct": 1.24,
    "meta-llama/Llama-3.1-3B": 3.21,
    "meta-llama/Llama-3.1-3B-Instruct": 3.21,
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
    "meta-llama/Llama-3.1-1B": 2048,
    "meta-llama/Llama-3.1-1B-Instruct": 2048,
    "meta-llama/Llama-3.1-3B": 3072,
    "meta-llama/Llama-3.1-3B-Instruct": 3072,
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
    include_optimizer=True,
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
        include_optimizer: Whether to include Adam optimizer states

    Returns:
        Dictionary with detailed memory breakdown in GB
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

    # Memory calculations
    # 1. Parameter memory (bfloat16 = 2 bytes per param)
    params_gb = (params * 2) / (1024**3)

    # 2. Gradient memory (same as params for trainable parameters)
    gradient_gb = params_gb  # All bridge params are trainable

    # 3. Optimizer states (Adam: momentum + variance in fp32)
    # Each state is 4 bytes per param, 2 states = 8 bytes total
    optimizer_gb = 0
    if include_optimizer:
        optimizer_gb = (params * 8) / (1024**3)

    # 4. Activation memory (scales with batch size, accounted separately)
    # Here we only account for the persistent activation overhead
    activation_overhead_gb = params_gb * 1.5  # Persistent buffers, workspace

    # Total bridge memory (without batch-dependent activations)
    total_bridge_gb = params_gb + gradient_gb + optimizer_gb + activation_overhead_gb

    return {
        "params_gb": params_gb,
        "gradient_gb": gradient_gb,
        "optimizer_gb": optimizer_gb,
        "activation_overhead_gb": activation_overhead_gb,
        "total_gb": total_bridge_gb,
        "param_count": params
    }


def get_memory_safe_config(
    source_model,
    target_model=None,
    max_length=1536,
    soft_tokens=128,
    depth=4,
    gpu_memory_gb=80.0,
    safety_margin=0.25,  # Increased from 0.2 to 0.25 for more safety
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
        safety_margin: Safety margin (0.25 = 25% buffer)
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

    # Add optimizer memory for any trainable parameters in the main models
    # Even if models are mostly frozen, adapter layers may exist
    # Assume ~1% of model params might be trainable (adapters/LoRA)
    source_optimizer_gb = source_params_b * 0.01 * 8  # 8 bytes per trainable param
    target_optimizer_gb = target_params_b * 0.01 * 8 if dual_model else 0

    # Estimate bridge memory
    bridge_info = estimate_bridge_memory_gb(
        source_dim=source_dim,
        target_dim=target_dim,
        soft_tokens=soft_tokens,
        depth=depth,
        include_optimizer=True,
    )
    bridge_mem_gb = bridge_info["total_gb"]

    # Base memory (models + bridge with optimizer + model optimizer states)
    base_memory_gb = (source_mem_gb + target_mem_gb + bridge_mem_gb +
                      source_optimizer_gb + target_optimizer_gb)

    # Activation memory calculation (more comprehensive)
    # Account for:
    # 1. Hidden states at each layer (forward + backward)
    # 2. Attention matrices (Q*K^T) which scale with seq_len^2
    # 3. Gradient accumulation for activations
    # 4. Temporary buffers and workspace

    # Calculate total model layers (approximate)
    # Llama/Mistral/Qwen at different sizes have different layer counts
    if source_params_b >= 7:
        source_layers = 32  # 7-8B models
        source_heads = 32   # Attention heads
    elif source_params_b >= 3:
        source_layers = 28  # 3B models
        source_heads = 24
    else:
        source_layers = 16  # 1B models
        source_heads = 16

    if dual_model:
        if target_params_b >= 7:
            target_layers = 32
            target_heads = 32
        elif target_params_b >= 3:
            target_layers = 28
            target_heads = 28
        else:
            target_layers = 16
            target_heads = 16
    else:
        target_layers = 0
        target_heads = 0

    # Calculate activation memory per batch more accurately

    # 1. Hidden states memory (forward + backward pass)
    # Each layer stores: input, output, and gradients
    # Formula: seq_len * hidden_dim * layers * 3 (input/output/grad) * 2 bytes
    source_hidden_gb = (max_length * source_dim * source_layers * 3 * 2) / (1024**3)

    # 2. Attention memory (Q*K^T matrices are seq_len x seq_len)
    # Formula: layers * seq_len^2 * 2 bytes (aggregated across heads)
    # Note: With flash attention, this is often optimized
    source_attention_gb = (source_layers * max_length * max_length * 2) / (1024**3)

    # 3. FFN intermediate activations (typically 4x hidden_dim)
    # Formula: seq_len * (4 * hidden_dim) * layers * 2 * 2 bytes
    source_ffn_gb = (max_length * (4 * source_dim) * source_layers * 2 * 2) / (1024**3)

    # Total for source model
    source_activation_gb = source_hidden_gb + source_attention_gb + source_ffn_gb

    # Same calculation for target model if dual
    target_activation_gb = 0
    if dual_model:
        target_hidden_gb = (max_length * target_dim * target_layers * 3 * 2) / (1024**3)
        target_attention_gb = (target_layers * max_length * max_length * 2) / (1024**3)
        target_ffn_gb = (max_length * (4 * target_dim) * target_layers * 2 * 2) / (1024**3)
        target_activation_gb = target_hidden_gb + target_attention_gb + target_ffn_gb

    # Bridge activation memory (Perceiver layers with cross-attention)
    # Perceiver has both self-attention and cross-attention
    bridge_hidden_gb = (max_length * source_dim * depth * 3 * 2) / (1024**3)
    bridge_attention_gb = (depth * 8 * soft_tokens * soft_tokens * 2) / (1024**3)  # Self-attention on soft tokens
    bridge_cross_attention_gb = (depth * 8 * soft_tokens * max_length * 2) / (1024**3)  # Cross-attention
    bridge_activation_gb = bridge_hidden_gb + bridge_attention_gb + bridge_cross_attention_gb

    # Total activation memory per batch
    activation_gb_per_batch = source_activation_gb + target_activation_gb + bridge_activation_gb

    # Add 15% overhead for temporary tensors, optimizer workspace, memory fragmentation
    # (Reduced from 30% since we're already being conservative in other calculations)
    activation_gb_per_batch *= 1.15

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

    # Apply conservative heuristics based on known working configs
    # CRITICAL: Be extra conservative to prevent OOM on 80GB H100s
    if dual_model:
        # Dual-model configs (very conservative due to two models + bridge)
        if source_params_b + target_params_b >= 15:  # e.g., 8B + 7B
            # VERIFIED WORKING: batch_size=2 for Llama-8B + Mistral-7B
            # Force batch_size=2 regardless of calculation since it's proven to work
            batch_size = 2
            grad_accum = 4
        elif source_params_b + target_params_b >= 10:  # e.g., 7B + 3B
            batch_size = min(max_batch_size, 2)  # Conservative: use 2 for safety
            grad_accum = 4
        elif source_params_b + target_params_b >= 5:  # e.g., 3B + 3B
            batch_size = min(max_batch_size, 4)  # Reduced further for safety
            grad_accum = 2
        else:  # Small models (<5B total)
            batch_size = min(max_batch_size, 8)  # Reduced further for safety
            grad_accum = 1
    else:
        # Single-model configs (still conservative)
        if source_params_b >= 7:
            batch_size = min(max_batch_size, 4)  # Reduced from 6 to 4 for safety
            grad_accum = 2
        elif source_params_b >= 3:
            batch_size = min(max_batch_size, 8)  # Reduced from 12 to 8 for safety
            grad_accum = 1
        else:
            batch_size = min(max_batch_size, 16)  # Reduced from 24 to 16 for safety
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
            "source_optimizer_gb": source_optimizer_gb,
            "target_model_gb": target_mem_gb,
            "target_optimizer_gb": target_optimizer_gb,
            "bridge_gb": bridge_mem_gb,
            "bridge_params_gb": bridge_info["params_gb"],
            "bridge_optimizer_gb": bridge_info["optimizer_gb"],
            "bridge_gradients_gb": bridge_info["gradient_gb"],
            "bridge_param_count": bridge_info["param_count"],
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
            if breakdown['source_optimizer_gb'] > 0:
                print(f"      • Optimizer (1% trainable): {breakdown['source_optimizer_gb']:.2f} GB")
            if breakdown['target_model_gb'] > 0:
                print(f"    - Target Model: {breakdown['target_model_gb']:.1f} GB")
                if breakdown['target_optimizer_gb'] > 0:
                    print(f"      • Optimizer (1% trainable): {breakdown['target_optimizer_gb']:.2f} GB")
            print(f"    - Bridge Total: {breakdown['bridge_gb']:.1f} GB")
            print(f"      • Params: {breakdown['bridge_params_gb']:.2f} GB ({breakdown['bridge_param_count']/1e6:.1f}M params)")
            print(f"      • Gradients: {breakdown['bridge_gradients_gb']:.2f} GB")
            print(f"      • Optimizer: {breakdown['bridge_optimizer_gb']:.2f} GB")
            print(f"    - Activations/batch: {breakdown['activation_per_batch_gb']:.1f} GB")
            print(f"    - Total Activations: {breakdown['total_activation_gb']:.1f} GB")


def main():
    """Generate and display memory-safe configurations for common model pairs."""

    # Test configurations
    test_configs = [
        # Dual-model configs (cross-model bridge)
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("meta-llama/Llama-3.1-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("meta-llama/Llama-3.1-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
        ("meta-llama/Llama-3.1-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("meta-llama/Llama-3.1-1B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),

        # Single-model configs (same-model bridge)
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", None),
        ("mistralai/Mistral-7B-Instruct-v0.3", None),
        ("meta-llama/Llama-3.1-3B-Instruct", None),
        ("meta-llama/Llama-3.1-1B-Instruct", None),
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
    source_model="meta-llama/Llama-3.1-3B-Instruct",
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