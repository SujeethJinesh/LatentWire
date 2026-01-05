"""
Latency measurement utilities for Telepathy experiments.

Provides accurate GPU timing, memory profiling, and throughput metrics
for comparing Bridge, Text-relay, and Zero-shot inference approaches.
"""

import time
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import gc


@dataclass
class LatencyMetrics:
    """Container for latency and throughput metrics."""

    # Timing metrics (in milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    # Throughput metrics
    samples_per_second: float
    tokens_per_second: Optional[float] = None

    # Memory metrics (in MB)
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float

    # Additional metadata
    num_samples: int
    num_warmup: int
    total_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class LatencyProfiler:
    """
    GPU-aware latency profiler with warmup handling and memory tracking.

    Usage:
        profiler = LatencyProfiler(warmup_iterations=5)

        # Profile a function
        metrics = profiler.profile(
            fn=lambda: model.generate(inputs),
            num_iterations=100,
            description="Bridge inference"
        )
    """

    def __init__(
        self,
        warmup_iterations: int = 5,
        device: str = "cuda",
        enable_memory_tracking: bool = True,
        verbose: bool = True
    ):
        """
        Initialize profiler.

        Args:
            warmup_iterations: Number of warmup runs before measurement
            device: Device to profile ("cuda" or "cpu")
            enable_memory_tracking: Whether to track GPU memory
            verbose: Print progress during profiling
        """
        self.warmup_iterations = warmup_iterations
        self.device = device
        self.enable_memory_tracking = enable_memory_tracking
        self.verbose = verbose

        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available() and device == "cuda"
        if device == "cuda" and not self.cuda_available:
            print("Warning: CUDA requested but not available, falling back to CPU timing")

    def profile(
        self,
        fn: Callable,
        num_iterations: int = 100,
        description: str = "Function",
        return_individual_times: bool = False
    ) -> LatencyMetrics:
        """
        Profile a function's latency and memory usage.

        Args:
            fn: Function to profile (should include all data movement)
            num_iterations: Number of iterations to measure
            description: Description for logging
            return_individual_times: If True, include raw timings in metrics

        Returns:
            LatencyMetrics object with timing and memory statistics
        """
        if self.verbose:
            print(f"\nProfiling {description}...")
            print(f"  Warmup iterations: {self.warmup_iterations}")
            print(f"  Measurement iterations: {num_iterations}")

        # Reset memory stats if tracking
        if self.cuda_available and self.enable_memory_tracking:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Warmup phase
        if self.verbose and self.warmup_iterations > 0:
            print(f"  Running warmup...")

        for _ in range(self.warmup_iterations):
            _ = fn()
            if self.cuda_available:
                torch.cuda.synchronize()

        # Measurement phase
        timings_ms = []

        if self.verbose:
            print(f"  Measuring latency...")

        # Record initial memory state
        if self.cuda_available and self.enable_memory_tracking:
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated()

        total_start = time.perf_counter()

        for i in range(num_iterations):
            if self.cuda_available:
                # GPU timing with CUDA events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                start_event.record()

                _ = fn()

                end_event.record()
                torch.cuda.synchronize()

                # Get elapsed time in milliseconds
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                # CPU timing
                start = time.perf_counter()
                _ = fn()
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000

            timings_ms.append(elapsed_ms)

            # Progress update
            if self.verbose and (i + 1) % max(1, num_iterations // 10) == 0:
                print(f"    Progress: {i + 1}/{num_iterations}")

        total_end = time.perf_counter()
        total_time_s = total_end - total_start

        # Calculate statistics
        timings_np = np.array(timings_ms)

        metrics = LatencyMetrics(
            mean_ms=float(np.mean(timings_np)),
            std_ms=float(np.std(timings_np)),
            min_ms=float(np.min(timings_np)),
            max_ms=float(np.max(timings_np)),
            p50_ms=float(np.percentile(timings_np, 50)),
            p95_ms=float(np.percentile(timings_np, 95)),
            p99_ms=float(np.percentile(timings_np, 99)),
            samples_per_second=num_iterations / total_time_s,
            num_samples=num_iterations,
            num_warmup=self.warmup_iterations,
            total_time_s=total_time_s,
            peak_memory_mb=0.0,
            allocated_memory_mb=0.0,
            reserved_memory_mb=0.0
        )

        # Get memory statistics
        if self.cuda_available and self.enable_memory_tracking:
            torch.cuda.synchronize()
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            metrics.allocated_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            metrics.reserved_memory_mb = torch.cuda.memory_reserved() / 1024 / 1024

        if self.verbose:
            self._print_metrics(metrics, description)

        return metrics

    def _print_metrics(self, metrics: LatencyMetrics, description: str):
        """Pretty print metrics to console."""
        print(f"\n{description} Results:")
        print(f"  Latency (ms):")
        print(f"    Mean: {metrics.mean_ms:.2f} ± {metrics.std_ms:.2f}")
        print(f"    Min/Max: {metrics.min_ms:.2f} / {metrics.max_ms:.2f}")
        print(f"    Percentiles - p50: {metrics.p50_ms:.2f}, p95: {metrics.p95_ms:.2f}, p99: {metrics.p99_ms:.2f}")
        print(f"  Throughput:")
        print(f"    Samples/sec: {metrics.samples_per_second:.2f}")
        if metrics.tokens_per_second:
            print(f"    Tokens/sec: {metrics.tokens_per_second:.2f}")
        if self.cuda_available and self.enable_memory_tracking:
            print(f"  Memory (MB):")
            print(f"    Peak: {metrics.peak_memory_mb:.2f}")
            print(f"    Allocated: {metrics.allocated_memory_mb:.2f}")
            print(f"    Reserved: {metrics.reserved_memory_mb:.2f}")


class ComparativeLatencyBenchmark:
    """
    Benchmark multiple inference approaches for fair comparison.

    Designed to integrate with run_unified_comparison.py for comparing:
    - Bridge (compressed latent)
    - Text-relay (full text baseline)
    - Zero-shot (no context baseline)
    """

    def __init__(
        self,
        warmup_iterations: int = 5,
        measurement_iterations: int = 100,
        device: str = "cuda",
        output_dir: Optional[Path] = None
    ):
        """
        Initialize comparative benchmark.

        Args:
            warmup_iterations: Warmup runs per approach
            measurement_iterations: Measurement runs per approach
            device: Device for benchmarking
            output_dir: Directory to save results JSON
        """
        self.profiler = LatencyProfiler(
            warmup_iterations=warmup_iterations,
            device=device,
            enable_memory_tracking=True,
            verbose=True
        )
        self.measurement_iterations = measurement_iterations
        self.output_dir = Path(output_dir) if output_dir else None
        self.results = {}

    def benchmark_approach(
        self,
        approach_name: str,
        inference_fn: Callable,
        num_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LatencyMetrics:
        """
        Benchmark a single inference approach.

        Args:
            approach_name: Name of the approach ("bridge", "text_relay", "zero_shot")
            inference_fn: Function that performs inference
            num_tokens: Optional number of tokens generated (for throughput calc)
            metadata: Optional metadata to store with results

        Returns:
            LatencyMetrics for this approach
        """
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {approach_name}")
        print(f"{'=' * 60}")

        # Profile the approach
        metrics = self.profiler.profile(
            fn=inference_fn,
            num_iterations=self.measurement_iterations,
            description=approach_name
        )

        # Calculate token throughput if provided
        if num_tokens:
            tokens_per_sample = num_tokens
            metrics.tokens_per_second = (
                tokens_per_sample * metrics.samples_per_second
            )

        # Store results
        self.results[approach_name] = {
            "metrics": metrics.to_dict(),
            "metadata": metadata or {}
        }

        return metrics

    def compare_approaches(
        self,
        bridge_fn: Callable,
        text_relay_fn: Callable,
        zero_shot_fn: Callable,
        num_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comparative benchmark across all three approaches.

        Args:
            bridge_fn: Bridge inference function
            text_relay_fn: Text-relay inference function
            zero_shot_fn: Zero-shot inference function
            num_tokens: Number of tokens generated per sample

        Returns:
            Dictionary with comparison results and speedup factors
        """
        print("\n" + "=" * 80)
        print("COMPARATIVE LATENCY BENCHMARK")
        print("=" * 80)

        # Benchmark each approach
        bridge_metrics = self.benchmark_approach(
            "bridge",
            bridge_fn,
            num_tokens=num_tokens,
            metadata={"type": "compressed_latent"}
        )

        text_relay_metrics = self.benchmark_approach(
            "text_relay",
            text_relay_fn,
            num_tokens=num_tokens,
            metadata={"type": "full_text_baseline"}
        )

        zero_shot_metrics = self.benchmark_approach(
            "zero_shot",
            zero_shot_fn,
            num_tokens=num_tokens,
            metadata={"type": "no_context_baseline"}
        )

        # Calculate speedup factors
        speedup_vs_text = text_relay_metrics.mean_ms / bridge_metrics.mean_ms
        speedup_vs_zero = zero_shot_metrics.mean_ms / bridge_metrics.mean_ms

        # Memory savings
        memory_savings_vs_text = (
            1 - bridge_metrics.peak_memory_mb / text_relay_metrics.peak_memory_mb
        ) * 100

        # Compile comparison
        comparison = {
            "approaches": self.results,
            "speedup_factors": {
                "bridge_vs_text_relay": speedup_vs_text,
                "bridge_vs_zero_shot": speedup_vs_zero
            },
            "memory_savings": {
                "bridge_vs_text_relay_percent": memory_savings_vs_text
            },
            "summary": {
                "fastest_approach": min(
                    self.results.keys(),
                    key=lambda x: self.results[x]["metrics"]["mean_ms"]
                ),
                "most_memory_efficient": min(
                    self.results.keys(),
                    key=lambda x: self.results[x]["metrics"]["peak_memory_mb"]
                ),
                "bridge_latency_ms": bridge_metrics.mean_ms,
                "text_relay_latency_ms": text_relay_metrics.mean_ms,
                "zero_shot_latency_ms": zero_shot_metrics.mean_ms
            }
        }

        # Print summary
        self._print_comparison_summary(comparison)

        # Save to file if output directory specified
        if self.output_dir:
            self._save_results(comparison)

        return comparison

    def _print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print formatted comparison summary."""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        summary = comparison["summary"]
        print(f"\nLatency Comparison:")
        print(f"  Bridge:      {summary['bridge_latency_ms']:.2f} ms")
        print(f"  Text-relay:  {summary['text_relay_latency_ms']:.2f} ms")
        print(f"  Zero-shot:   {summary['zero_shot_latency_ms']:.2f} ms")

        print(f"\nSpeedup Factors:")
        print(f"  Bridge vs Text-relay: {comparison['speedup_factors']['bridge_vs_text_relay']:.2f}x")
        print(f"  Bridge vs Zero-shot:  {comparison['speedup_factors']['bridge_vs_zero_shot']:.2f}x")

        print(f"\nMemory Efficiency:")
        print(f"  Bridge saves {comparison['memory_savings']['bridge_vs_text_relay_percent']:.1f}% memory vs Text-relay")

        print(f"\nWinner:")
        print(f"  Fastest: {summary['fastest_approach']}")
        print(f"  Most memory efficient: {summary['most_memory_efficient']}")

    def _save_results(self, comparison: Dict[str, Any]):
        """Save results to JSON file."""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "latency_comparison.json"

        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def create_mock_inference_functions(
    batch_size: int = 1,
    seq_len: int = 512,
    device: str = "cuda"
) -> tuple[Callable, Callable, Callable]:
    """
    Create mock inference functions for testing.

    Returns:
        Tuple of (bridge_fn, text_relay_fn, zero_shot_fn)
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    def bridge_fn():
        # Simulate compressed latent inference
        latent = torch.randn(batch_size, 32, 256).to(device_obj)  # 32 soft tokens
        output = torch.randn(batch_size, seq_len, 50000).to(device_obj)
        return output

    def text_relay_fn():
        # Simulate full text inference
        input_ids = torch.randint(0, 50000, (batch_size, seq_len)).to(device_obj)
        output = torch.randn(batch_size, seq_len, 50000).to(device_obj)
        return output

    def zero_shot_fn():
        # Simulate zero-shot inference (no context)
        input_ids = torch.randint(0, 50000, (batch_size, 64)).to(device_obj)  # Shorter
        output = torch.randn(batch_size, seq_len, 50000).to(device_obj)
        return output

    return bridge_fn, text_relay_fn, zero_shot_fn


def main():
    """Example usage and testing."""
    print("Telepathy Latency Profiler - Test Run")
    print("=" * 80)

    # Create benchmark
    benchmark = ComparativeLatencyBenchmark(
        warmup_iterations=3,
        measurement_iterations=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=Path("runs/latency_test")
    )

    # Create mock functions for testing
    bridge_fn, text_relay_fn, zero_shot_fn = create_mock_inference_functions(
        batch_size=4,
        seq_len=256
    )

    # Run comparison
    results = benchmark.compare_approaches(
        bridge_fn=bridge_fn,
        text_relay_fn=text_relay_fn,
        zero_shot_fn=zero_shot_fn,
        num_tokens=256
    )

    print("\n" + "=" * 80)
    print("Test complete! Latency profiler is working correctly.")
    print("=" * 80)


if __name__ == "__main__":
    main()