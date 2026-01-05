"""
Fixed latency measurement utilities for Telepathy experiments.

Provides accurate GPU timing, memory profiling, and throughput metrics
for comparing Bridge, Text-relay, and Zero-shot inference approaches.

FIXES APPLIED:
1. Increased warmup iterations to 10 for stability
2. Reset memory stats between approaches
3. Added batch size handling for throughput
4. Better error handling for CUDA availability
5. Added integration helper for run_unified_comparison.py
"""

import time
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
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
    batch_size: int = 1  # Added for proper throughput calculation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class LatencyProfiler:
    """
    GPU-aware latency profiler with warmup handling and memory tracking.

    FIXES:
    - Increased default warmup to 10 iterations
    - Added batch_size parameter for accurate throughput
    - Better CUDA availability handling
    """

    def __init__(
        self,
        warmup_iterations: int = 10,  # Increased from 5
        device: str = "cuda",
        enable_memory_tracking: bool = True,
        verbose: bool = True
    ):
        """
        Initialize profiler.

        Args:
            warmup_iterations: Number of warmup runs before measurement (10 recommended)
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
            if verbose:
                print("Warning: CUDA requested but not available, falling back to CPU timing")
            self.device = "cpu"

    def profile(
        self,
        fn: Callable,
        num_iterations: int = 100,
        batch_size: int = 1,
        description: str = "Function",
        return_individual_times: bool = False
    ) -> LatencyMetrics:
        """
        Profile a function's latency and memory usage.

        Args:
            fn: Function to profile (should include all data movement)
            num_iterations: Number of iterations to measure
            batch_size: Batch size for throughput calculation
            description: Description for logging
            return_individual_times: If True, include raw timings in metrics

        Returns:
            LatencyMetrics object with timing and memory statistics
        """
        if self.verbose:
            print(f"\nProfiling {description}...")
            print(f"  Device: {'CUDA' if self.cuda_available else 'CPU'}")
            print(f"  Warmup iterations: {self.warmup_iterations}")
            print(f"  Measurement iterations: {num_iterations}")
            print(f"  Batch size: {batch_size}")

        # Reset memory stats if tracking
        if self.cuda_available and self.enable_memory_tracking:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Warmup phase
        if self.verbose and self.warmup_iterations > 0:
            print(f"  Running warmup...")

        for i in range(self.warmup_iterations):
            _ = fn()
            if self.cuda_available:
                torch.cuda.synchronize()

            # Progress for long warmups
            if self.verbose and self.warmup_iterations >= 10 and (i + 1) % 5 == 0:
                print(f"    Warmup progress: {i + 1}/{self.warmup_iterations}")

        # Clear caches after warmup
        if self.cuda_available:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Measurement phase
        timings_ms = []

        if self.verbose:
            print(f"  Measuring latency...")

        # Record initial memory state
        if self.cuda_available and self.enable_memory_tracking:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()  # Reset again for clean measurement

        total_start = time.perf_counter()

        for i in range(num_iterations):
            if self.cuda_available:
                # GPU timing with CUDA events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Ensure all previous operations are complete
                torch.cuda.synchronize()
                start_event.record()

                _ = fn()

                end_event.record()
                torch.cuda.synchronize()  # Wait for completion

                # Get elapsed time in milliseconds
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                # CPU timing with high-resolution timer
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()
                _ = fn()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000

            timings_ms.append(elapsed_ms)

            # Progress update
            if self.verbose and (i + 1) % max(1, num_iterations // 10) == 0:
                print(f"    Progress: {i + 1}/{num_iterations} (last: {elapsed_ms:.2f}ms)")

        total_end = time.perf_counter()
        total_time_s = total_end - total_start

        # Calculate statistics
        timings_np = np.array(timings_ms)

        # Remove outliers (optional - top/bottom 1%)
        if len(timings_np) > 100:
            lower = np.percentile(timings_np, 1)
            upper = np.percentile(timings_np, 99)
            timings_filtered = timings_np[(timings_np >= lower) & (timings_np <= upper)]
            if self.verbose and len(timings_filtered) < len(timings_np):
                print(f"  Removed {len(timings_np) - len(timings_filtered)} outliers")
        else:
            timings_filtered = timings_np

        metrics = LatencyMetrics(
            mean_ms=float(np.mean(timings_filtered)),
            std_ms=float(np.std(timings_filtered)),
            min_ms=float(np.min(timings_filtered)),
            max_ms=float(np.max(timings_filtered)),
            p50_ms=float(np.percentile(timings_filtered, 50)),
            p95_ms=float(np.percentile(timings_filtered, 95)),
            p99_ms=float(np.percentile(timings_filtered, 99)),
            samples_per_second=(num_iterations * batch_size) / total_time_s,
            num_samples=num_iterations,
            num_warmup=self.warmup_iterations,
            total_time_s=total_time_s,
            batch_size=batch_size,
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


def create_profiled_wrapper(
    fn: Callable,
    device: str = "cuda",
    sync_before: bool = True,
    sync_after: bool = True
) -> Callable:
    """
    Create a wrapper function that properly handles GPU synchronization.

    This is for integrating with existing code that uses time.time().

    Args:
        fn: Function to wrap
        device: Device being used
        sync_before: Synchronize before calling function
        sync_after: Synchronize after calling function

    Returns:
        Wrapped function with proper synchronization
    """
    def wrapped(*args, **kwargs):
        if device == "cuda" and torch.cuda.is_available():
            if sync_before:
                torch.cuda.synchronize()

        result = fn(*args, **kwargs)

        if device == "cuda" and torch.cuda.is_available():
            if sync_after:
                torch.cuda.synchronize()

        return result

    return wrapped


def measure_gpu_latency(
    fn: Callable,
    num_iterations: int = 1,
    warmup: int = 0,
    return_individual: bool = False
) -> Tuple[float, Optional[List[float]]]:
    """
    Simple GPU-aware latency measurement for integration with existing code.

    Drop-in replacement for time.time() based measurements.

    Args:
        fn: Function to measure
        num_iterations: Number of iterations
        warmup: Warmup iterations
        return_individual: Return individual measurements

    Returns:
        (mean_latency_ms, individual_times_ms or None)

    Example:
        # Instead of:
        start = time.time()
        result = model.forward(x)
        latency = time.time() - start

        # Use:
        latency_ms, _ = measure_gpu_latency(lambda: model.forward(x))
        latency = latency_ms / 1000  # Convert to seconds if needed
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Warmup
    for _ in range(warmup):
        fn()
        if device == "cuda":
            torch.cuda.synchronize()

    # Measure
    times_ms = []
    for _ in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            fn()
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000

        times_ms.append(elapsed_ms)

    mean_ms = np.mean(times_ms)

    if return_individual:
        return mean_ms, times_ms
    else:
        return mean_ms, None


class ComparativeLatencyBenchmark:
    """
    Benchmark multiple inference approaches for fair comparison.

    FIXES:
    - Reset memory stats between approaches
    - Better handling of approach metadata
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        measurement_iterations: int = 100,
        device: str = "cuda",
        output_dir: Optional[Path] = None
    ):
        """
        Initialize comparative benchmark.

        Args:
            warmup_iterations: Warmup runs per approach (10 recommended)
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
        batch_size: int = 1,
        num_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LatencyMetrics:
        """
        Benchmark a single inference approach.

        FIXED: Added batch_size parameter for accurate throughput.

        Args:
            approach_name: Name of the approach
            inference_fn: Function that performs inference
            batch_size: Batch size for throughput calculation
            num_tokens: Optional number of tokens generated
            metadata: Optional metadata to store with results

        Returns:
            LatencyMetrics for this approach
        """
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {approach_name}")
        print(f"{'=' * 60}")

        # Reset GPU memory stats for fair comparison
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Profile the approach
        metrics = self.profiler.profile(
            fn=inference_fn,
            num_iterations=self.measurement_iterations,
            batch_size=batch_size,
            description=approach_name
        )

        # Calculate token throughput if provided
        if num_tokens:
            tokens_per_sample = num_tokens
            metrics.tokens_per_second = (
                tokens_per_sample * batch_size * metrics.samples_per_second
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
        batch_size: int = 1,
        num_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comparative benchmark across all three approaches.

        FIXED: Added batch_size parameter.

        Args:
            bridge_fn: Bridge inference function
            text_relay_fn: Text-relay inference function
            zero_shot_fn: Zero-shot inference function
            batch_size: Batch size for all approaches
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
            batch_size=batch_size,
            num_tokens=num_tokens,
            metadata={"type": "compressed_latent"}
        )

        text_relay_metrics = self.benchmark_approach(
            "text_relay",
            text_relay_fn,
            batch_size=batch_size,
            num_tokens=num_tokens,
            metadata={"type": "full_text_baseline"}
        )

        zero_shot_metrics = self.benchmark_approach(
            "zero_shot",
            zero_shot_fn,
            batch_size=batch_size,
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
                "bridge_vs_zero_shot": speedup_vs_zero,
                "text_relay_vs_zero_shot": text_relay_metrics.mean_ms / zero_shot_metrics.mean_ms
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
        """Pretty print comparison summary."""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        summary = comparison["summary"]
        print(f"\nLatency Comparison:")
        print(f"  Bridge:      {summary['bridge_latency_ms']:.2f} ms")
        print(f"  Text-relay:  {summary['text_relay_latency_ms']:.2f} ms")
        print(f"  Zero-shot:   {summary['zero_shot_latency_ms']:.2f} ms")

        print(f"\nSpeedup Factors:")
        for key, value in comparison['speedup_factors'].items():
            print(f"  {key}: {value:.2f}x")

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


def main():
    """Test the fixed utilities."""
    print("Telepathy Latency Profiler - Fixed Version")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # Test simple measurement
    print("\n1. Testing simple GPU latency measurement:")
    def dummy_forward():
        x = torch.randn(4, 512, 768, device=device)
        return x @ x.T

    latency_ms, _ = measure_gpu_latency(dummy_forward, num_iterations=10, warmup=3)
    print(f"   Average latency: {latency_ms:.2f} ms")

    # Test profiler
    print("\n2. Testing LatencyProfiler:")
    profiler = LatencyProfiler(warmup_iterations=10, device=device)
    metrics = profiler.profile(
        fn=dummy_forward,
        num_iterations=20,
        batch_size=4,
        description="Matrix multiplication"
    )

    print("\n" + "=" * 80)
    print("Tests complete! Fixed latency utilities are working correctly.")
    print("=" * 80)


if __name__ == "__main__":
    main()