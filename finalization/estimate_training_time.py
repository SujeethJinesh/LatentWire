#!/usr/bin/env python3
"""
Estimate training time for LatentWire experiment with different GPU configurations.
Based on model sizes and batch processing requirements.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    params_billions: float
    memory_gb_fp16: float  # Approximate memory for model weights in FP16
    memory_gb_activations: float  # Approximate memory for activations per sample

@dataclass
class TrainingConfig:
    """Training configuration."""
    samples: int
    epochs: int
    batch_size: int
    latent_len: int
    d_z: int

@dataclass
class GPUConfig:
    """GPU configuration."""
    name: str
    n_gpus: int
    memory_per_gpu_gb: float
    tflops_fp16: float  # H100 theoretical peak

class TrainingTimeEstimator:
    """Estimate training time for different GPU configurations."""

    def __init__(self):
        # Model configurations
        self.llama = ModelConfig(
            name="Llama-3.1-8B",
            params_billions=8.0,
            memory_gb_fp16=16.0,  # ~2GB per billion params in FP16
            memory_gb_activations=0.5  # Per sample
        )

        self.qwen = ModelConfig(
            name="Qwen-2.5-7B",
            params_billions=7.0,
            memory_gb_fp16=14.0,
            memory_gb_activations=0.5
        )

        # H100 GPU specs
        self.h100 = GPUConfig(
            name="H100",
            n_gpus=1,
            memory_per_gpu_gb=80.0,
            tflops_fp16=989.0  # Theoretical peak
        )

    def estimate_memory_usage(self, config: TrainingConfig, n_gpus: int) -> Dict[str, float]:
        """Estimate memory usage for training."""

        # Base model memory (frozen, but still loaded)
        model_memory = self.llama.memory_gb_fp16 + self.qwen.memory_gb_fp16

        # Encoder and adapter memory (trainable parameters)
        # Encoder: input_dim -> d_z, with latent_len outputs
        encoder_params = 768 * config.d_z  # Approximate
        adapter_params = config.d_z * 4096 * 2  # Two adapters (Llama + Qwen)
        trainable_memory_gb = (encoder_params + adapter_params) * 2 / 1e9  # FP16

        # Optimizer states (Adam: 2x params for momentum + variance)
        optimizer_memory_gb = trainable_memory_gb * 2

        # Activation memory (scales with batch size)
        activation_memory_gb = (self.llama.memory_gb_activations +
                                self.qwen.memory_gb_activations) * config.batch_size

        # Gradient memory
        gradient_memory_gb = trainable_memory_gb

        total_memory_gb = (model_memory + trainable_memory_gb +
                          optimizer_memory_gb + activation_memory_gb +
                          gradient_memory_gb)

        # Memory per GPU
        memory_per_gpu = total_memory_gb / n_gpus if n_gpus > 1 else total_memory_gb

        return {
            'model_memory_gb': model_memory,
            'trainable_memory_gb': trainable_memory_gb,
            'optimizer_memory_gb': optimizer_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'gradient_memory_gb': gradient_memory_gb,
            'total_memory_gb': total_memory_gb,
            'memory_per_gpu_gb': memory_per_gpu
        }

    def estimate_compute_time(self, config: TrainingConfig, n_gpus: int) -> Dict[str, float]:
        """Estimate compute time based on FLOPs."""

        # Forward pass FLOPs (approximate)
        # Each sample goes through encoder + 2 models
        flops_per_sample_forward = (
            # Encoder
            768 * config.d_z * config.latent_len +
            # Llama forward (simplified)
            self.llama.params_billions * 1e9 * 2 +  # 2 FLOPs per param
            # Qwen forward
            self.qwen.params_billions * 1e9 * 2
        )

        # Backward pass is ~2x forward
        flops_per_sample = flops_per_sample_forward * 3

        # Total FLOPs for training
        total_samples = config.samples * config.epochs
        total_flops = flops_per_sample * total_samples

        # Time estimation based on GPU throughput
        # Assume 30% efficiency for real-world training (conservative)
        effective_tflops = self.h100.tflops_fp16 * 0.3 * n_gpus

        compute_time_seconds = total_flops / (effective_tflops * 1e12)

        # Add overhead for data loading, checkpointing, logging
        overhead_factor = 1.3
        total_time_seconds = compute_time_seconds * overhead_factor

        return {
            'flops_per_sample': flops_per_sample,
            'total_flops': total_flops,
            'compute_time_seconds': compute_time_seconds,
            'total_time_seconds': total_time_seconds,
            'total_time_hours': total_time_seconds / 3600
        }

    def estimate_batch_processing_time(self, config: TrainingConfig, n_gpus: int) -> Dict[str, float]:
        """Estimate time based on batch processing rates from empirical data."""

        # Empirical estimates (samples per second per GPU)
        # Based on typical training speeds for 7-8B models
        if n_gpus == 1:
            samples_per_second = 2.0  # Conservative for sequential processing
        elif n_gpus == 2:
            samples_per_second = 3.5  # Some parallelism
        elif n_gpus == 4:
            samples_per_second = 6.0  # Good parallelism
        else:
            samples_per_second = 1.5 * n_gpus  # Scaling efficiency decreases

        # Adjust for batch size
        if config.batch_size > 4:
            # Larger batches are more efficient
            samples_per_second *= (1 + 0.1 * math.log2(config.batch_size / 4))

        # Total training time
        total_samples = config.samples * config.epochs
        time_seconds = total_samples / samples_per_second

        return {
            'samples_per_second': samples_per_second,
            'samples_per_hour': samples_per_second * 3600,
            'time_seconds': time_seconds,
            'time_hours': time_seconds / 3600,
            'time_per_epoch_hours': time_seconds / (3600 * config.epochs)
        }

    def analyze_configurations(self, config: TrainingConfig) -> List[Dict]:
        """Analyze different GPU configurations."""

        results = []

        for n_gpus in [1, 2, 4]:
            memory = self.estimate_memory_usage(config, n_gpus)
            compute = self.estimate_compute_time(config, n_gpus)
            batch = self.estimate_batch_processing_time(config, n_gpus)

            # Check if configuration is feasible
            feasible = memory['memory_per_gpu_gb'] < self.h100.memory_per_gpu_gb

            results.append({
                'n_gpus': n_gpus,
                'feasible': feasible,
                'memory_per_gpu_gb': memory['memory_per_gpu_gb'],
                'total_memory_gb': memory['total_memory_gb'],
                'estimated_hours_compute': compute['total_time_hours'],
                'estimated_hours_empirical': batch['time_hours'],
                'samples_per_second': batch['samples_per_second'],
                'time_per_epoch_hours': batch['time_per_epoch_hours']
            })

        return results


def main():
    """Main analysis."""

    # Current configuration from RUN.sh
    config = TrainingConfig(
        samples=5000,
        epochs=8,
        batch_size=4,
        latent_len=32,
        d_z=256
    )

    estimator = TrainingTimeEstimator()

    print("=" * 80)
    print("LATENTWIRE TRAINING TIME ESTIMATION")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Samples: {config.samples}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Latent: {config.latent_len}x{config.d_z}")
    print(f"  Total samples to process: {config.samples * config.epochs}")
    print()

    print("=" * 80)
    print("GPU CONFIGURATION ANALYSIS")
    print("=" * 80)
    print()

    results = estimator.analyze_configurations(config)

    for result in results:
        print(f"Configuration: {result['n_gpus']} GPU(s)")
        print("-" * 40)
        print(f"  Memory per GPU: {result['memory_per_gpu_gb']:.1f} GB / 80 GB")
        print(f"  Memory feasible: {'✓' if result['feasible'] else '✗'}")
        print(f"  Samples/second: {result['samples_per_second']:.2f}")
        print(f"  Time per epoch: {result['time_per_epoch_hours']:.2f} hours")
        print(f"  Total time (empirical): {result['estimated_hours_empirical']:.2f} hours")
        print(f"  Total time (compute): {result['estimated_hours_compute']:.2f} hours")
        print()

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find optimal configuration
    feasible_configs = [r for r in results if r['feasible']]
    under_12h = [r for r in feasible_configs if r['estimated_hours_empirical'] < 12]

    if under_12h:
        optimal = min(under_12h, key=lambda x: x['n_gpus'])
        print(f"✓ OPTIMAL: {optimal['n_gpus']} GPU(s)")
        print(f"  - Completes in {optimal['estimated_hours_empirical']:.1f} hours")
        print(f"  - Uses {optimal['memory_per_gpu_gb']:.1f} GB per GPU")
        print(f"  - Saves {4 - optimal['n_gpus']} GPU(s) for other jobs")
    else:
        print("✗ No configuration completes within 12 hours")
        print("  Consider:")
        print("  - Reducing epochs from 8 to 6")
        print("  - Reducing samples from 5000 to 3000")
        print("  - Increasing batch size to 8 (if memory allows)")

    print()
    print("Additional considerations:")
    print("  - 1 GPU: Most memory efficient, good for debugging")
    print("  - 2 GPUs: Balanced speed/resource usage")
    print("  - 4 GPUs: Fastest but uses all available resources")
    print()
    print("Note: Estimates are conservative and actual times may be faster.")


if __name__ == "__main__":
    main()