#!/usr/bin/env python3
"""
Calculate optimal batch sizes for H100 GPUs with Llama-8B model.

This script performs detailed memory calculations to determine the maximum
safe batch sizes for different GPU configurations, accounting for:
- Model weights
- Optimizer states (AdamW)
- Activations
- Gradient storage
- System overhead
"""

import math
# Type hints removed for Python 2.7 compatibility


class GPUConfig:
    """GPU configuration and specifications."""
    def __init__(self, name, memory_gb, num_gpus):
        self.name = name
        self.memory_gb = memory_gb
        self.num_gpus = num_gpus


class ModelConfig:
    """Model configuration and memory requirements."""
    def __init__(self, name, param_count_b, hidden_dim, num_layers, vocab_size):
        self.name = name
        self.param_count_b = param_count_b  # in billions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size


class TrainingConfig:
    """Training configuration affecting memory."""
    def __init__(self, seq_length, dtype_bytes, gradient_checkpointing, mixed_precision, optimizer):
        self.seq_length = seq_length
        self.dtype_bytes = dtype_bytes  # 2 for fp16/bf16, 4 for fp32
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer  # "adamw", "sgd", etc.


class BatchSizeCalculator:
    """Calculate optimal batch sizes for different GPU configurations."""

    def __init__(self, gpu, model, training):
        self.gpu = gpu
        self.model = model
        self.training = training

    def calculate_model_memory_gb(self):
        """Calculate memory needed for model weights."""
        param_bytes = self.model.param_count_b * 1e9 * self.training.dtype_bytes
        return param_bytes / 1e9

    def calculate_optimizer_memory_gb(self):
        """Calculate memory needed for optimizer states."""
        if self.training.optimizer == "adamw":
            # AdamW needs 2 momentum buffers (fp32) for each parameter
            # In mixed precision training:
            # - Master weights: fp32 (4 bytes)
            # - Momentum 1: fp32 (4 bytes)
            # - Momentum 2: fp32 (4 bytes)
            # Total: 12 bytes per parameter for optimizer states
            if self.training.mixed_precision:
                bytes_per_param = 12  # 3 * fp32 states
            else:
                # Without mixed precision: weights + mom1 + mom2 all in fp32
                bytes_per_param = 8  # Just mom1 + mom2 (weights counted separately)
        elif self.training.optimizer == "sgd":
            # SGD with momentum needs 1 buffer
            bytes_per_param = 4  # momentum buffer
        else:
            # Basic SGD - no additional state
            bytes_per_param = 0

        optimizer_bytes = self.model.param_count_b * 1e9 * bytes_per_param
        return optimizer_bytes / 1e9

    def calculate_activation_memory_gb(self, batch_size):
        """Calculate memory needed for activations during forward/backward pass."""
        # Approximate activation memory based on empirical measurements
        # For transformer models: ~4 * batch_size * seq_len * hidden_dim * num_layers

        if self.training.gradient_checkpointing:
            # Gradient checkpointing reduces activation memory significantly
            # Only need to store activations for one layer at a time
            memory_multiplier = 1.5
        else:
            # Need to store all intermediate activations
            memory_multiplier = 4.0

        activation_bytes = (
            memory_multiplier *
            batch_size *
            self.training.seq_length *
            self.model.hidden_dim *
            self.model.num_layers *
            self.training.dtype_bytes
        )

        return activation_bytes / 1e9

    def calculate_gradient_memory_gb(self):
        """Calculate memory needed for gradients."""
        # Gradients are same size as model parameters
        grad_bytes = self.model.param_count_b * 1e9 * self.training.dtype_bytes
        return grad_bytes / 1e9

    def calculate_misc_overhead_gb(self):
        """Calculate miscellaneous memory overhead (CUDA kernels, temp buffers, etc.)."""
        # Conservative estimate: 5-10% of total GPU memory
        return self.gpu.memory_gb * 0.08

    def calculate_max_batch_size(self, target_utilization=0.85):
        """Calculate maximum batch size for given configuration.

        Args:
            target_utilization: Target GPU memory utilization (0.85 = 85%)

        Returns:
            Dictionary with batch size and memory breakdown
        """
        # Calculate fixed memory costs
        model_mem = self.calculate_model_memory_gb()
        optimizer_mem = self.calculate_optimizer_memory_gb()
        gradient_mem = self.calculate_gradient_memory_gb()
        overhead_mem = self.calculate_misc_overhead_gb()

        # Total fixed memory
        fixed_memory = model_mem + optimizer_mem + gradient_mem + overhead_mem

        # Available memory for activations
        total_available = self.gpu.memory_gb * self.gpu.num_gpus * target_utilization

        if self.gpu.num_gpus > 1:
            # With DDP, model is replicated on each GPU
            # But batch is split across GPUs
            available_for_activations = (self.gpu.memory_gb * target_utilization - fixed_memory) * self.gpu.num_gpus
        else:
            available_for_activations = total_available - fixed_memory

        # Binary search for maximum batch size
        min_batch = 1
        max_batch = 1024
        optimal_batch = 1

        while min_batch <= max_batch:
            mid_batch = (min_batch + max_batch) // 2
            activation_mem = self.calculate_activation_memory_gb(mid_batch)

            if activation_mem <= available_for_activations:
                optimal_batch = mid_batch
                min_batch = mid_batch + 1
            else:
                max_batch = mid_batch - 1

        # Calculate actual memory usage
        actual_activation_mem = self.calculate_activation_memory_gb(optimal_batch)
        total_memory_used = fixed_memory + actual_activation_mem

        if self.gpu.num_gpus > 1:
            # For DDP, report per-GPU values
            per_gpu_memory = fixed_memory + actual_activation_mem / self.gpu.num_gpus
            utilization = per_gpu_memory / self.gpu.memory_gb
        else:
            utilization = total_memory_used / self.gpu.memory_gb

        return {
            'batch_size': optimal_batch,
            'memory_breakdown': {
                'model_gb': model_mem,
                'optimizer_gb': optimizer_mem,
                'gradients_gb': gradient_mem,
                'activations_gb': actual_activation_mem,
                'overhead_gb': overhead_mem,
                'total_gb': total_memory_used,
            },
            'per_gpu_batch': optimal_batch // self.gpu.num_gpus if self.gpu.num_gpus > 1 else optimal_batch,
            'utilization': utilization,
            'gradient_accumulation_steps': 1,
            'effective_batch_size': optimal_batch,
        }


def main():
    """Calculate and display optimal batch sizes for H100 configurations."""

    # H100 GPU configuration
    h100 = GPUConfig(name="H100 80GB", memory_gb=80.0, num_gpus=1)

    # Llama-3.1-8B model configuration
    llama_8b = ModelConfig(
        name="Llama-3.1-8B",
        param_count_b=8.03,
        hidden_dim=4096,
        num_layers=32,
        vocab_size=128256,
    )

    # Training configuration
    training = TrainingConfig(
        seq_length=1024,  # Typical context length
        dtype_bytes=2,  # bf16
        gradient_checkpointing=True,
        mixed_precision=True,
        optimizer="adamw",
    )

    print("="*80)
    print("OPTIMAL BATCH SIZE CALCULATIONS FOR H100 GPUs WITH LLAMA-8B")
    print("="*80)
    print(f"\nModel: {llama_8b.name} ({llama_8b.param_count_b}B parameters)")
    print(f"Training: seq_len={training.seq_length}, dtype=bf16, optimizer={training.optimizer}")
    print(f"Gradient Checkpointing: {training.gradient_checkpointing}")
    print(f"Mixed Precision: {training.mixed_precision}")
    print("\n" + "="*80)

    # Calculate for different GPU configurations
    configs = [
        (1, 0.85),  # Single GPU, 85% utilization
        (2, 0.85),  # 2 GPUs
        (3, 0.85),  # 3 GPUs
        (4, 0.85),  # 4 GPUs
    ]

    results = {}

    for num_gpus, target_util in configs:
        gpu_config = GPUConfig(name=f"{num_gpus}x H100", memory_gb=80.0, num_gpus=num_gpus)
        calculator = BatchSizeCalculator(gpu_config, llama_8b, training)
        result = calculator.calculate_max_batch_size(target_util)
        results[num_gpus] = result

        print(f"\n{num_gpus}x H100 Configuration (Target: {int(target_util*100)}% utilization):")
        print("-" * 60)
        print(f"  Total Batch Size: {result['batch_size']}")
        print(f"  Per-GPU Batch Size: {result['per_gpu_batch']}")
        print(f"  Effective Batch Size: {result['effective_batch_size']}")
        print(f"  GPU Memory Utilization: {result['utilization']*100:.1f}%")
        print(f"\n  Memory Breakdown (per GPU):")
        breakdown = result['memory_breakdown']
        if num_gpus > 1:
            print(f"    - Model Weights: {breakdown['model_gb']:.1f} GB")
            print(f"    - Optimizer States: {breakdown['optimizer_gb']:.1f} GB")
            print(f"    - Gradients: {breakdown['gradients_gb']:.1f} GB")
            print(f"    - Activations: {breakdown['activations_gb']/num_gpus:.1f} GB")
            print(f"    - Overhead: {breakdown['overhead_gb']:.1f} GB")
            print(f"    - Total per GPU: {breakdown['model_gb'] + breakdown['optimizer_gb'] + breakdown['gradients_gb'] + breakdown['activations_gb']/num_gpus + breakdown['overhead_gb']:.1f} GB")
        else:
            print(f"    - Model Weights: {breakdown['model_gb']:.1f} GB")
            print(f"    - Optimizer States: {breakdown['optimizer_gb']:.1f} GB")
            print(f"    - Gradients: {breakdown['gradients_gb']:.1f} GB")
            print(f"    - Activations: {breakdown['activations_gb']:.1f} GB")
            print(f"    - Overhead: {breakdown['overhead_gb']:.1f} GB")
            print(f"    - Total: {breakdown['total_gb']:.1f} GB")

    # Generate recommendations for config.yaml
    print("\n" + "="*80)
    print("RECOMMENDED CONFIG.YAML SETTINGS")
    print("="*80)

    print("\nSuggested batch_config updates:")
    print("```yaml")
    print("training:")
    print("  batch_config:")
    print("    # H100 (80GB) configurations - OPTIMIZED VALUES")
    print("    h100_single:")
    print(f"      batch_size: {results[1]['batch_size']}")
    print("      gradient_accumulation_steps: 1")
    print(f"      effective_batch_size: {results[1]['effective_batch_size']}")
    print("      elastic_batch_size: true")
    print("      elastic_target_util: 0.85")
    print("      notes: 'Optimized for single H100 with gradient checkpointing'")
    print()
    print("    h100_dual:")
    print(f"      batch_size: {results[2]['per_gpu_batch']}  # per-GPU batch size")
    print("      gradient_accumulation_steps: 1")
    print(f"      effective_batch_size: {results[2]['effective_batch_size']}")
    print("      elastic_batch_size: true")
    print("      elastic_target_util: 0.85")
    print("      notes: 'DDP on 2x H100 GPUs'")
    print()
    print("    h100_triple:")
    print(f"      batch_size: {results[3]['per_gpu_batch']}  # per-GPU batch size")
    print("      gradient_accumulation_steps: 1")
    print(f"      effective_batch_size: {results[3]['effective_batch_size']}")
    print("      elastic_batch_size: true")
    print("      elastic_target_util: 0.85")
    print("      notes: 'DDP on 3x H100 GPUs'")
    print()
    print("    h100_quad:")
    print(f"      batch_size: {results[4]['per_gpu_batch']}  # per-GPU batch size")
    print("      gradient_accumulation_steps: 1")
    print(f"      effective_batch_size: {results[4]['effective_batch_size']}")
    print("      elastic_batch_size: true")
    print("      elastic_target_util: 0.85")
    print("      notes: 'DDP on 4x H100 GPUs (HPC standard)'")
    print("```")

    # Additional recommendations
    print("\n" + "="*80)
    print("ADDITIONAL RECOMMENDATIONS")
    print("="*80)
    print("""
1. Memory Optimization Strategies:
   - Enable gradient checkpointing (--grad_ckpt) to reduce activation memory
   - Use bf16 mixed precision (--mixed_precision bf16) for H100s
   - Consider FSDP for very large models or higher batch sizes

2. Performance Tuning:
   - H100s benefit from larger batch sizes due to high memory bandwidth
   - Use torch.compile() for additional speedup (10-20%)
   - Enable TF32 for matmuls (automatic on H100)

3. Multi-GPU Scaling:
   - 1 GPU: Max single-GPU batch for development/debugging
   - 2 GPUs: Good for medium-scale experiments
   - 4 GPUs: Optimal for production training on HPC

4. Elastic Batch Sizing:
   - The elastic_batch_size feature in train.py can dynamically adjust
   - Set elastic_target_util between 0.80-0.90 for safety margin
   - Monitor GPU memory usage and adjust if OOMs occur

5. Gradient Accumulation:
   - If batch sizes are still too large, use gradient accumulation
   - Effective batch = batch_size * gradient_accumulation_steps * num_gpus
""")


if __name__ == "__main__":
    main()