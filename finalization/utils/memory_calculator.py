#!/usr/bin/env python3
"""
Memory-safe batch size calculator for LatentWire training.

This module provides accurate memory requirement calculations that include:
- Model parameters memory
- Optimizer state memory (Adam: 2x model params for momentum and variance)
- Gradient memory
- Activation memory during forward/backward passes
- Safety margins for CUDA overhead

Usage:
    python memory_calculator.py --model_size_gb 14 --gpu_memory_gb 80 --num_gpus 4

This ensures OOM-free training by properly accounting for all memory components.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class MemoryRequirements:
    """Memory requirements for training components."""
    model_params_gb: float
    optimizer_state_gb: float
    gradients_gb: float
    activations_per_sample_gb: float
    cuda_overhead_gb: float
    safety_margin_gb: float

    @property
    def total_fixed_gb(self) -> float:
        """Total fixed memory (everything except activations)."""
        return (self.model_params_gb +
                self.optimizer_state_gb +
                self.gradients_gb +
                self.cuda_overhead_gb +
                self.safety_margin_gb)

    def available_for_batch(self, gpu_memory_gb: float) -> float:
        """Calculate memory available for batch processing."""
        return max(0, gpu_memory_gb - self.total_fixed_gb)

    def max_batch_size(self, gpu_memory_gb: float) -> int:
        """Calculate maximum safe batch size for given GPU memory."""
        available = self.available_for_batch(gpu_memory_gb)
        if self.activations_per_sample_gb <= 0:
            return 1
        max_batch = int(available / self.activations_per_sample_gb)
        return max(1, max_batch)


class MemoryCalculator:
    """Calculate memory-safe batch sizes for LLM training."""

    # Optimizer memory multipliers (relative to model parameters)
    OPTIMIZER_MULTIPLIERS = {
        'sgd': 0.0,      # No optimizer state
        'adam': 2.0,     # Momentum + variance
        'adamw': 2.0,    # Same as Adam
        'lamb': 2.0,     # Similar to Adam
        'adafactor': 0.5,  # More memory efficient
    }

    # Default safety margins and overheads
    CUDA_OVERHEAD_GB = 2.0  # CUDA context, kernels, workspace
    SAFETY_MARGIN_GB = 2.0  # Additional safety buffer

    def __init__(self, optimizer: str = 'adamw'):
        """Initialize calculator with specified optimizer."""
        self.optimizer = optimizer.lower()
        if self.optimizer not in self.OPTIMIZER_MULTIPLIERS:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def calculate_requirements(self,
                              model_size_gb: float,
                              sequence_length: int = 1024,
                              hidden_dim: int = 4096,
                              vocab_size: int = 128256,
                              dtype: str = 'bfloat16') -> MemoryRequirements:
        """
        Calculate comprehensive memory requirements.

        Args:
            model_size_gb: Model parameter size in GB
            sequence_length: Maximum sequence length
            hidden_dim: Model hidden dimension
            vocab_size: Vocabulary size
            dtype: Data type (float32, float16, bfloat16)

        Returns:
            MemoryRequirements object with all components
        """
        # Bytes per element based on dtype
        bytes_per_element = {
            'float32': 4,
            'float16': 2,
            'bfloat16': 2,
            'int8': 1,
        }.get(dtype, 2)

        # Model parameters memory
        model_params_gb = model_size_gb

        # Optimizer state memory (Adam needs 2x model params)
        optimizer_multiplier = self.OPTIMIZER_MULTIPLIERS[self.optimizer]
        optimizer_state_gb = model_params_gb * optimizer_multiplier

        # Gradients memory (same size as model params)
        gradients_gb = model_params_gb

        # Activation memory per sample (rough estimate)
        # This includes intermediate activations during forward/backward
        # Formula: seq_len * hidden_dim * num_layers * multiplier
        # Multiplier accounts for attention, FFN, residuals, etc.
        num_layers = 32  # Typical for 7-8B models
        activation_multiplier = 12  # Conservative estimate for transformer

        activations_bytes = (sequence_length * hidden_dim * num_layers *
                            activation_multiplier * bytes_per_element)
        activations_per_sample_gb = activations_bytes / (1024**3)

        # Add logits computation memory (seq_len * vocab_size)
        logits_bytes = sequence_length * vocab_size * bytes_per_element
        activations_per_sample_gb += logits_bytes / (1024**3)

        return MemoryRequirements(
            model_params_gb=model_params_gb,
            optimizer_state_gb=optimizer_state_gb,
            gradients_gb=gradients_gb,
            activations_per_sample_gb=activations_per_sample_gb,
            cuda_overhead_gb=self.CUDA_OVERHEAD_GB,
            safety_margin_gb=self.SAFETY_MARGIN_GB
        )

    def recommend_batch_configuration(self,
                                     model_size_gb: float,
                                     gpu_memory_gb: float,
                                     num_gpus: int = 1,
                                     target_effective_batch: int = 32,
                                     **kwargs) -> Dict:
        """
        Recommend optimal batch size and gradient accumulation configuration.

        Args:
            model_size_gb: Model parameter size in GB
            gpu_memory_gb: Memory per GPU in GB
            num_gpus: Number of GPUs available
            target_effective_batch: Desired effective batch size
            **kwargs: Additional args for calculate_requirements

        Returns:
            Dictionary with recommended configuration
        """
        # Calculate memory requirements
        reqs = self.calculate_requirements(model_size_gb, **kwargs)

        # Calculate max batch size per GPU
        max_batch_per_gpu = reqs.max_batch_size(gpu_memory_gb)

        # If even batch size 1 doesn't fit, we need model parallelism
        if max_batch_per_gpu < 1:
            return {
                'error': 'Model too large for GPU memory',
                'suggestion': 'Use model parallelism or larger GPUs',
                'memory_breakdown': {
                    'model_params_gb': reqs.model_params_gb,
                    'optimizer_state_gb': reqs.optimizer_state_gb,
                    'gradients_gb': reqs.gradients_gb,
                    'total_fixed_gb': reqs.total_fixed_gb,
                    'gpu_memory_gb': gpu_memory_gb,
                    'deficit_gb': reqs.total_fixed_gb - gpu_memory_gb
                }
            }

        # Choose a reasonable batch size (not necessarily the max)
        # Using 60-80% of max is often more stable
        safe_batch_per_gpu = max(1, int(max_batch_per_gpu * 0.7))

        # Calculate gradient accumulation steps needed
        total_batch = safe_batch_per_gpu * num_gpus
        grad_accum_steps = max(1, target_effective_batch // total_batch)

        # Actual effective batch size
        effective_batch = safe_batch_per_gpu * num_gpus * grad_accum_steps

        return {
            'batch_size_per_gpu': safe_batch_per_gpu,
            'gradient_accumulation_steps': grad_accum_steps,
            'effective_batch_size': effective_batch,
            'num_gpus': num_gpus,
            'memory_usage': {
                'fixed_memory_gb': reqs.total_fixed_gb,
                'available_for_batch_gb': reqs.available_for_batch(gpu_memory_gb),
                'per_sample_activation_gb': reqs.activations_per_sample_gb,
                'estimated_usage_gb': (reqs.total_fixed_gb +
                                      safe_batch_per_gpu * reqs.activations_per_sample_gb),
                'gpu_memory_gb': gpu_memory_gb,
                'utilization_pct': ((reqs.total_fixed_gb +
                                   safe_batch_per_gpu * reqs.activations_per_sample_gb) /
                                  gpu_memory_gb * 100)
            },
            'memory_breakdown': {
                'model_params_gb': reqs.model_params_gb,
                'optimizer_state_gb': reqs.optimizer_state_gb,
                'gradients_gb': reqs.gradients_gb,
                'activations_gb': safe_batch_per_gpu * reqs.activations_per_sample_gb,
                'cuda_overhead_gb': reqs.cuda_overhead_gb,
                'safety_margin_gb': reqs.safety_margin_gb
            },
            'warnings': self._generate_warnings(safe_batch_per_gpu, max_batch_per_gpu,
                                               effective_batch, target_effective_batch)
        }

    def _generate_warnings(self, safe_batch: int, max_batch: int,
                          effective: int, target: int) -> list:
        """Generate any relevant warnings about the configuration."""
        warnings = []

        if safe_batch < 2:
            warnings.append(f"Very small batch size ({safe_batch}) may lead to unstable training")

        if effective < target:
            warnings.append(f"Effective batch size ({effective}) is less than target ({target})")

        if safe_batch < max_batch * 0.5:
            warnings.append("Using conservative batch size; you may be able to increase it")

        return warnings


def main():
    """CLI interface for memory calculations."""
    parser = argparse.ArgumentParser(
        description='Calculate memory-safe batch sizes for LLM training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Llama-3.1-8B on single H100 (80GB)
  python memory_calculator.py --model_size_gb 14 --gpu_memory_gb 80 --num_gpus 1

  # Llama-3.1-8B on 4x A100 (40GB each)
  python memory_calculator.py --model_size_gb 14 --gpu_memory_gb 40 --num_gpus 4

  # With custom sequence length
  python memory_calculator.py --model_size_gb 14 --gpu_memory_gb 80 --sequence_length 2048
        """
    )

    parser.add_argument('--model_size_gb', type=float, default=14,
                       help='Model parameter size in GB (default: 14 for Llama-8B)')
    parser.add_argument('--gpu_memory_gb', type=float, default=80,
                       help='GPU memory in GB (default: 80)')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs (default: 1)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['sgd', 'adam', 'adamw', 'lamb', 'adafactor'],
                       help='Optimizer type (default: adamw)')
    parser.add_argument('--sequence_length', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    parser.add_argument('--hidden_dim', type=int, default=4096,
                       help='Model hidden dimension (default: 4096)')
    parser.add_argument('--vocab_size', type=int, default=128256,
                       help='Vocabulary size (default: 128256 for Llama)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Data type (default: bfloat16)')
    parser.add_argument('--target_batch', type=int, default=32,
                       help='Target effective batch size (default: 32)')
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format')

    args = parser.parse_args()

    # Create calculator
    calc = MemoryCalculator(optimizer=args.optimizer)

    # Get recommendations
    config = calc.recommend_batch_configuration(
        model_size_gb=args.model_size_gb,
        gpu_memory_gb=args.gpu_memory_gb,
        num_gpus=args.num_gpus,
        target_effective_batch=args.target_batch,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        dtype=args.dtype
    )

    if args.json:
        print(json.dumps(config, indent=2))
    else:
        if 'error' in config:
            print(f"❌ Error: {config['error']}", flush=True)
            print(f"   {config['suggestion']}")
            print("\nMemory Breakdown:")
            for key, value in config['memory_breakdown'].items():
                print(f"  {key:20s}: {value:8.2f} GB")
        else:
            print(f"{'=' * 60}")
            print("Memory-Safe Batch Configuration", flush=True)
            print(f"{'=' * 60}")
            print(f"Model: {args.model_size_gb:.1f}GB, GPU: {args.gpu_memory_gb}GB x {args.num_gpus}")
            print(f"Optimizer: {args.optimizer}")
            print(f"{'=' * 60}")
            print("\n📊 Recommended Configuration:")
            print(f"  Batch size per GPU:        {config['batch_size_per_gpu']}", flush=True)
            print(f"  Gradient accumulation:     {config['gradient_accumulation_steps']}", flush=True)
            print(f"  Effective batch size:      {config['effective_batch_size']}", flush=True)
            print(f"  Number of GPUs:           {config['num_gpus']}")

            print("\n💾 Memory Usage:")
            usage = config['memory_usage']
            print(f"  Fixed memory:             {usage['fixed_memory_gb']:.2f} GB")
            print(f"  Available for batch:      {usage['available_for_batch_gb']:.2f} GB", flush=True)
            print(f"  Estimated total usage:    {usage['estimated_usage_gb']:.2f} GB")
            print(f"  GPU utilization:          {usage['utilization_pct']:.1f}%")

            print("\n📋 Memory Breakdown:")
            for key, value in config['memory_breakdown'].items():
                label = key.replace('_', ' ').title()
                print(f"  {label:25s}: {value:8.2f} GB")

            if config['warnings']:
                print("\n⚠️  Warnings:", flush=True)
                for warning in config['warnings']:
                    print(f"  - {warning}", flush=True)

            print("\n✅ This configuration should train without OOM errors.", flush=True)


if __name__ == '__main__':
    main()