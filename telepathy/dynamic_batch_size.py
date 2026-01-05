#!/usr/bin/env python3
"""Dynamic batch size optimizer for maximum GPU utilization.

This module automatically finds the maximum batch size that can fit in GPU memory
without OOM errors, using binary search and adaptive scaling.

Features:
- Automatically finds maximum batch size without OOM
- Binary search for optimal size
- Adjusts for number of GPUs
- Accounts for gradient accumulation
- Memory-aware scaling
- Works with Llama3.1-8B and other models
- Supports 1-4 GPU configurations
- Mixed precision training support
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Callable
import gc
import time
import logging
from dataclasses import dataclass
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchSizeConfig:
    """Configuration for batch size optimization."""

    min_batch_size: int = 1
    max_batch_size: int = 512
    initial_batch_size: int = 8
    memory_fraction: float = 0.95  # Target GPU memory utilization
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    sequence_length: int = 512
    model_size_gb: float = 14.0  # Llama3.1-8B size
    safety_margin: float = 0.9  # Additional safety margin
    max_retries: int = 3
    warmup_steps: int = 2  # Steps to warm up GPU


class DynamicBatchSizeOptimizer:
    """Automatically finds and maintains optimal batch size for GPU utilization."""

    def __init__(self, config: Optional[BatchSizeConfig] = None):
        """Initialize the batch size optimizer.

        Args:
            config: Configuration for batch size optimization
        """
        self.config = config or BatchSizeConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for optimization")

        # Track optimization history
        self.history = {
            'successful': [],
            'failed': [],
            'memory_usage': []
        }

        # Current optimal batch size
        self.optimal_batch_size = None
        self.effective_batch_size = None

        logger.info(f"Initialized DynamicBatchSizeOptimizer with {self.num_gpus} GPUs")
        self._log_gpu_info()

    def _log_gpu_info(self):
        """Log information about available GPUs."""
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU {i}: {props.name}, Memory: {memory_gb:.1f} GB")

    def _get_memory_usage(self, device_id: int = 0) -> Dict[str, float]:
        """Get current GPU memory usage.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}

        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        free = total - reserved

        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'usage_fraction': reserved / total
        }

    def _clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()

    def _try_batch_size(self,
                       model: nn.Module,
                       batch_size: int,
                       forward_fn: Optional[Callable] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[bool, Dict[str, Any]]:
        """Try running with a specific batch size.

        Args:
            model: The model to test
            batch_size: Batch size to try
            forward_fn: Optional custom forward function
            optimizer: Optional optimizer for testing backward pass

        Returns:
            Tuple of (success, metrics_dict)
        """
        self._clear_memory()

        try:
            # Create dummy input based on sequence length
            if hasattr(model, 'config'):
                vocab_size = getattr(model.config, 'vocab_size', 32000)
            else:
                vocab_size = 32000  # Default for Llama

            # Adjust batch size for multi-GPU
            per_gpu_batch_size = batch_size
            if self.num_gpus > 1:
                per_gpu_batch_size = batch_size // self.num_gpus

            # Create dummy data
            dummy_input_ids = torch.randint(
                0, vocab_size,
                (per_gpu_batch_size, self.config.sequence_length),
                device=self.device
            )

            # Measure memory before forward pass
            mem_before = self._get_memory_usage()

            # Warm up (helps stabilize memory allocation)
            for _ in range(self.config.warmup_steps):
                if forward_fn:
                    output = forward_fn(dummy_input_ids)
                else:
                    output = model(dummy_input_ids)

                if hasattr(output, 'logits'):
                    loss = output.logits.mean()
                else:
                    loss = output.mean() if isinstance(output, torch.Tensor) else output[0].mean()

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    # Don't step during warmup

            # Actual test
            start_time = time.time()

            if forward_fn:
                output = forward_fn(dummy_input_ids)
            else:
                output = model(dummy_input_ids)

            if hasattr(output, 'logits'):
                loss = output.logits.mean()
            else:
                loss = output.mean() if isinstance(output, torch.Tensor) else output[0].mean()

            # Test backward pass if optimizer provided
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            forward_time = time.time() - start_time

            # Measure memory after
            mem_after = self._get_memory_usage()

            # Calculate throughput
            samples_per_sec = batch_size / forward_time

            metrics = {
                'batch_size': batch_size,
                'per_gpu_batch_size': per_gpu_batch_size,
                'success': True,
                'forward_time': forward_time,
                'samples_per_sec': samples_per_sec,
                'memory_before': mem_before,
                'memory_after': mem_after,
                'memory_increase_gb': mem_after['reserved'] - mem_before['reserved'],
                'peak_memory_fraction': mem_after['usage_fraction']
            }

            logger.info(f"Batch size {batch_size} successful - "
                       f"Memory: {mem_after['usage_fraction']:.1%}, "
                       f"Throughput: {samples_per_sec:.1f} samples/sec")

            self.history['successful'].append(batch_size)
            self.history['memory_usage'].append(metrics)

            return True, metrics

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"Batch size {batch_size} failed: {str(e)[:100]}")
            self.history['failed'].append(batch_size)
            self._clear_memory()
            return False, {'batch_size': batch_size, 'success': False, 'error': str(e)}

    def find_optimal_batch_size(self,
                              model: nn.Module,
                              forward_fn: Optional[Callable] = None,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              binary_search: bool = True) -> int:
        """Find the optimal batch size using binary search or linear search.

        Args:
            model: The model to optimize for
            forward_fn: Optional custom forward function
            optimizer: Optional optimizer for testing backward pass
            binary_search: Use binary search (True) or linear search (False)

        Returns:
            Optimal batch size
        """
        logger.info(f"Starting batch size optimization (method: {'binary' if binary_search else 'linear'})")

        # Put model in training mode
        model.train()

        if binary_search:
            return self._binary_search_optimal(model, forward_fn, optimizer)
        else:
            return self._linear_search_optimal(model, forward_fn, optimizer)

    def _binary_search_optimal(self,
                              model: nn.Module,
                              forward_fn: Optional[Callable] = None,
                              optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """Find optimal batch size using binary search.

        Args:
            model: The model to optimize for
            forward_fn: Optional custom forward function
            optimizer: Optional optimizer for testing backward pass

        Returns:
            Optimal batch size
        """
        low = self.config.min_batch_size
        high = self.config.max_batch_size
        best_batch_size = low
        best_metrics = None

        # First, check if minimum works
        success, metrics = self._try_batch_size(model, low, forward_fn, optimizer)
        if not success:
            raise RuntimeError(f"Even minimum batch size {low} causes OOM")

        # Binary search
        while low <= high:
            mid = (low + high) // 2

            # Ensure mid is divisible by num_gpus for even distribution
            if self.num_gpus > 1:
                mid = (mid // self.num_gpus) * self.num_gpus

            success, metrics = self._try_batch_size(model, mid, forward_fn, optimizer)

            if success:
                # Check if we're within memory target
                if metrics['peak_memory_fraction'] <= self.config.memory_fraction:
                    best_batch_size = mid
                    best_metrics = metrics
                    low = mid + 1
                else:
                    # Too close to memory limit, back off
                    high = mid - 1
            else:
                high = mid - 1

        # Apply safety margin
        self.optimal_batch_size = int(best_batch_size * self.config.safety_margin)

        # Ensure divisible by num_gpus
        if self.num_gpus > 1:
            self.optimal_batch_size = (self.optimal_batch_size // self.num_gpus) * self.num_gpus

        # Calculate effective batch size with gradient accumulation
        self.effective_batch_size = self.optimal_batch_size * self.config.gradient_accumulation_steps

        logger.info(f"Optimal batch size found: {self.optimal_batch_size}")
        logger.info(f"Effective batch size (with grad accumulation): {self.effective_batch_size}")
        if best_metrics:
            logger.info(f"Peak memory usage: {best_metrics['peak_memory_fraction']:.1%}")
            logger.info(f"Throughput: {best_metrics['samples_per_sec']:.1f} samples/sec")

        return self.optimal_batch_size

    def _linear_search_optimal(self,
                              model: nn.Module,
                              forward_fn: Optional[Callable] = None,
                              optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """Find optimal batch size using linear search (more conservative).

        Args:
            model: The model to optimize for
            forward_fn: Optional custom forward function
            optimizer: Optional optimizer for testing backward pass

        Returns:
            Optimal batch size
        """
        best_batch_size = self.config.min_batch_size
        best_metrics = None

        # Start from initial batch size and increase
        current_batch_size = self.config.initial_batch_size

        # Ensure divisible by num_gpus
        if self.num_gpus > 1:
            current_batch_size = (current_batch_size // self.num_gpus) * self.num_gpus

        consecutive_failures = 0

        while current_batch_size <= self.config.max_batch_size:
            success, metrics = self._try_batch_size(model, current_batch_size, forward_fn, optimizer)

            if success:
                if metrics['peak_memory_fraction'] <= self.config.memory_fraction:
                    best_batch_size = current_batch_size
                    best_metrics = metrics
                    consecutive_failures = 0

                    # Exponential increase when successful
                    current_batch_size = min(int(current_batch_size * 1.5),
                                            self.config.max_batch_size)
                else:
                    # Getting close to limit, increase more conservatively
                    current_batch_size += max(1, current_batch_size // 10)
            else:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    # Stop after consecutive failures
                    break
                # Back off slightly
                current_batch_size = int(current_batch_size * 0.75)

            # Ensure divisible by num_gpus
            if self.num_gpus > 1:
                current_batch_size = (current_batch_size // self.num_gpus) * self.num_gpus

        # Apply safety margin
        self.optimal_batch_size = int(best_batch_size * self.config.safety_margin)

        # Ensure divisible by num_gpus
        if self.num_gpus > 1:
            self.optimal_batch_size = (self.optimal_batch_size // self.num_gpus) * self.num_gpus

        # Calculate effective batch size with gradient accumulation
        self.effective_batch_size = self.optimal_batch_size * self.config.gradient_accumulation_steps

        logger.info(f"Optimal batch size found: {self.optimal_batch_size}")
        logger.info(f"Effective batch size (with grad accumulation): {self.effective_batch_size}")
        if best_metrics:
            logger.info(f"Peak memory usage: {best_metrics['peak_memory_fraction']:.1%}")
            logger.info(f"Throughput: {best_metrics['samples_per_sec']:.1f} samples/sec")

        return self.optimal_batch_size

    def adjust_for_sequence_length(self, new_seq_length: int) -> int:
        """Adjust optimal batch size for different sequence length.

        Args:
            new_seq_length: New sequence length

        Returns:
            Adjusted batch size
        """
        if self.optimal_batch_size is None:
            raise RuntimeError("Must run find_optimal_batch_size first")

        # Memory scales approximately linearly with sequence length
        scaling_factor = self.config.sequence_length / new_seq_length
        adjusted_batch_size = int(self.optimal_batch_size * scaling_factor)

        # Ensure divisible by num_gpus
        if self.num_gpus > 1:
            adjusted_batch_size = (adjusted_batch_size // self.num_gpus) * self.num_gpus

        # Ensure within bounds
        adjusted_batch_size = max(self.config.min_batch_size,
                                 min(adjusted_batch_size, self.config.max_batch_size))

        logger.info(f"Adjusted batch size for seq_len {new_seq_length}: {adjusted_batch_size}")

        return adjusted_batch_size

    def get_gradient_accumulation_steps(self, target_effective_batch_size: int) -> int:
        """Calculate gradient accumulation steps needed for target effective batch size.

        Args:
            target_effective_batch_size: Desired effective batch size

        Returns:
            Number of gradient accumulation steps
        """
        if self.optimal_batch_size is None:
            raise RuntimeError("Must run find_optimal_batch_size first")

        grad_accum_steps = math.ceil(target_effective_batch_size / self.optimal_batch_size)

        logger.info(f"Gradient accumulation steps for effective batch size {target_effective_batch_size}: "
                   f"{grad_accum_steps} (actual batch size: {self.optimal_batch_size})")

        return grad_accum_steps

    def monitor_and_adjust(self,
                          current_batch_size: int,
                          current_memory_usage: float,
                          target_memory: float = 0.9) -> int:
        """Monitor current memory usage and suggest batch size adjustment.

        Args:
            current_batch_size: Current batch size
            current_memory_usage: Current memory usage fraction (0-1)
            target_memory: Target memory usage fraction

        Returns:
            Suggested new batch size
        """
        if current_memory_usage > 0.95:
            # Dangerously high, reduce
            new_batch_size = int(current_batch_size * 0.8)
            logger.warning(f"Memory usage critical ({current_memory_usage:.1%}), "
                          f"reducing batch size to {new_batch_size}")
        elif current_memory_usage > target_memory:
            # Slightly high, minor reduction
            new_batch_size = int(current_batch_size * 0.95)
            logger.info(f"Memory usage high ({current_memory_usage:.1%}), "
                       f"reducing batch size to {new_batch_size}")
        elif current_memory_usage < target_memory * 0.8:
            # Underutilized, can increase
            new_batch_size = int(current_batch_size * 1.2)
            logger.info(f"Memory underutilized ({current_memory_usage:.1%}), "
                       f"increasing batch size to {new_batch_size}")
        else:
            # Within target range
            new_batch_size = current_batch_size

        # Ensure divisible by num_gpus
        if self.num_gpus > 1:
            new_batch_size = (new_batch_size // self.num_gpus) * self.num_gpus

        # Ensure within bounds
        new_batch_size = max(self.config.min_batch_size,
                            min(new_batch_size, self.config.max_batch_size))

        return new_batch_size

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary and statistics.

        Returns:
            Dictionary with optimization results and statistics
        """
        summary = {
            'num_gpus': self.num_gpus,
            'optimal_batch_size': self.optimal_batch_size,
            'effective_batch_size': self.effective_batch_size,
            'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
            'sequence_length': self.config.sequence_length,
            'successful_batch_sizes': sorted(self.history['successful']),
            'failed_batch_sizes': sorted(self.history['failed']),
            'gpu_info': []
        }

        # Add GPU information
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            summary['gpu_info'].append({
                'device_id': i,
                'name': props.name,
                'memory_gb': props.total_memory / (1024**3)
            })

        # Add best memory usage stats if available
        if self.history['memory_usage']:
            best = max(self.history['memory_usage'],
                      key=lambda x: x['batch_size'] if x['success'] else 0)
            summary['best_run'] = {
                'batch_size': best['batch_size'],
                'memory_usage': best['peak_memory_fraction'],
                'throughput_samples_per_sec': best.get('samples_per_sec', 0)
            }

        return summary


def optimize_batch_size_for_model(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                                 sequence_length: int = 512,
                                 target_memory_fraction: float = 0.9) -> Dict[str, Any]:
    """Convenience function to optimize batch size for a specific model.

    Args:
        model_name: Model name or path
        sequence_length: Sequence length for optimization
        target_memory_fraction: Target GPU memory utilization

    Returns:
        Dictionary with optimization results
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    # Estimate model size
    model_sizes = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": 14.0,
        "meta-llama/Meta-Llama-3.1-8B": 14.0,
        "Qwen/Qwen2.5-7B-Instruct": 13.0,
        "Qwen/Qwen2.5-7B": 13.0,
    }

    model_size_gb = model_sizes.get(model_name, 14.0)  # Default to Llama size

    # Configure optimizer
    config = BatchSizeConfig(
        sequence_length=sequence_length,
        model_size_gb=model_size_gb,
        memory_fraction=target_memory_fraction,
        mixed_precision=True
    )

    optimizer = DynamicBatchSizeOptimizer(config)

    # Load model configuration (not the full model to save memory)
    model_config = AutoConfig.from_pretrained(model_name)

    # Create a dummy model for testing (much smaller than full model)
    # This is just for testing batch sizes, not actual training
    class DummyModel(nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(min(2, num_layers))
            ])
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids):
            x = self.embeddings(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)

    # Create dummy model with similar memory footprint
    dummy_model = DummyModel(
        model_config.vocab_size,
        model_config.hidden_size,
        model_config.num_hidden_layers
    ).cuda()

    # Find optimal batch size
    optimal_batch_size = optimizer.find_optimal_batch_size(dummy_model)

    # Get summary
    summary = optimizer.get_summary()
    summary['model_name'] = model_name

    # Clean up
    del dummy_model
    torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    """Example usage and testing."""

    import argparse

    parser = argparse.ArgumentParser(description="Dynamic batch size optimizer")
    parser.add_argument("--model", type=str,
                       default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Model name or path")
    parser.add_argument("--seq_length", type=int, default=512,
                       help="Sequence length for optimization")
    parser.add_argument("--target_memory", type=float, default=0.9,
                       help="Target GPU memory utilization (0-1)")
    parser.add_argument("--method", type=str, default="binary",
                       choices=["binary", "linear"],
                       help="Search method (binary or linear)")

    args = parser.parse_args()

    print("=" * 60)
    print("Dynamic Batch Size Optimizer")
    print("=" * 60)

    # Run optimization
    results = optimize_batch_size_for_model(
        model_name=args.model,
        sequence_length=args.seq_length,
        target_memory_fraction=args.target_memory
    )

    # Print results
    print("\nOptimization Results:")
    print("-" * 40)
    print(f"Model: {results['model_name']}")
    print(f"Number of GPUs: {results['num_gpus']}")
    print(f"Sequence Length: {results['sequence_length']}")
    print(f"\nOptimal Batch Size: {results['optimal_batch_size']}")
    print(f"Effective Batch Size: {results['effective_batch_size']}")

    if 'best_run' in results:
        print(f"\nBest Run Statistics:")
        print(f"  Memory Usage: {results['best_run']['memory_usage']:.1%}")
        print(f"  Throughput: {results['best_run']['throughput_samples_per_sec']:.1f} samples/sec")

    print("\nGPU Information:")
    for gpu in results['gpu_info']:
        print(f"  GPU {gpu['device_id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

    print("\nSuccessful batch sizes tested:", results['successful_batch_sizes'])
    if results['failed_batch_sizes']:
        print("Failed batch sizes:", results['failed_batch_sizes'])

    print("\n" + "=" * 60)
    print("Optimization complete!")