"""
GPU monitoring and optimization utilities.

Features:
- Real-time GPU utilization tracking
- Memory usage monitoring
- Temperature monitoring
- Performance bottleneck detection
- Automatic batch size optimization
- Multi-GPU coordination
- Power usage tracking
- Kernel timing analysis
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from datetime import datetime
import subprocess

import torch
import torch.cuda as cuda
import numpy as np


class GPUMonitor:
    """
    Comprehensive GPU monitoring for training optimization.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        sample_interval: float = 1.0,
        history_size: int = 1000,
        enable_profiling: bool = False,
    ):
        """
        Initialize GPU monitor.

        Args:
            log_dir: Directory to save monitoring logs
            sample_interval: Sampling interval in seconds
            history_size: Number of samples to keep in memory
            enable_profiling: Whether to enable detailed profiling
        """
        self.log_dir = Path(log_dir) if log_dir else Path("runs/gpu_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.sample_interval = sample_interval
        self.enable_profiling = enable_profiling

        # Check GPU availability
        self.gpu_available = cuda.is_available()
        self.gpu_count = cuda.device_count() if self.gpu_available else 0

        # Monitoring data
        self.history = {i: deque(maxlen=history_size) for i in range(self.gpu_count)}
        self.current_stats = {}

        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None

        # Log file
        self.log_file = self.log_dir / f"gpu_monitor_{int(time.time())}.jsonl"

    def start(self) -> None:
        """Start GPU monitoring in background thread."""
        if not self.gpu_available:
            print("No GPU available for monitoring")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"GPU monitoring started (logging to {self.log_file})")

    def stop(self) -> None:
        """Stop GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("GPU monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                stats = self._collect_stats()
                self._save_stats(stats)

                # Update history
                for gpu_id, gpu_stats in stats.items():
                    if isinstance(gpu_id, int):
                        self.history[gpu_id].append(gpu_stats)

                self.current_stats = stats
                time.sleep(self.sample_interval)

            except Exception as e:
                print(f"Error in GPU monitoring: {e}", flush=True)
                time.sleep(self.sample_interval)

    def _collect_stats(self) -> Dict[str, Any]:
        """Collect current GPU statistics."""
        stats = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
        }

        for i in range(self.gpu_count):
            gpu_stats = self._get_gpu_stats(i)
            stats[i] = gpu_stats

        # Add system-wide stats
        if self.gpu_count > 0:
            stats['total'] = self._get_aggregate_stats(stats)

        return stats

    def _get_gpu_stats(self, device_id: int) -> Dict[str, Any]:
        """Get statistics for a specific GPU."""
        cuda.synchronize(device_id)

        stats = {
            'device_id': device_id,
            'name': cuda.get_device_name(device_id),
        }

        # Memory stats
        mem_allocated = cuda.memory_allocated(device_id)
        mem_reserved = cuda.memory_reserved(device_id)
        mem_total = cuda.get_device_properties(device_id).total_memory

        stats.update({
            'memory_allocated_gb': mem_allocated / (1024 ** 3),
            'memory_reserved_gb': mem_reserved / (1024 ** 3),
            'memory_total_gb': mem_total / (1024 ** 3),
            'memory_utilization': (mem_allocated / mem_total) * 100,
        })

        # Try to get utilization via nvidia-smi
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits',
                    '-i', str(device_id)
                ],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 4:
                    stats['gpu_utilization'] = float(values[0])
                    stats['memory_utilization_smi'] = float(values[1])
                    stats['temperature_c'] = float(values[2])
                    stats['power_draw_w'] = float(values[3])
        except:
            # nvidia-smi not available or failed
            pass

        return stats

    def _get_aggregate_stats(self, all_stats: Dict) -> Dict[str, float]:
        """Calculate aggregate statistics across all GPUs."""
        aggregate = {}

        metrics_to_average = [
            'memory_allocated_gb', 'memory_utilization',
            'gpu_utilization', 'temperature_c'
        ]

        metrics_to_sum = ['memory_allocated_gb', 'power_draw_w']

        for metric in metrics_to_average:
            values = []
            for i in range(self.gpu_count):
                if metric in all_stats[i]:
                    values.append(all_stats[i][metric])
            if values:
                aggregate[f'avg_{metric}'] = np.mean(values)
                aggregate[f'max_{metric}'] = np.max(values)

        for metric in metrics_to_sum:
            total = 0
            for i in range(self.gpu_count):
                if metric in all_stats[i]:
                    total += all_stats[i][metric]
            aggregate[f'total_{metric}'] = total

        return aggregate

    def _save_stats(self, stats: Dict) -> None:
        """Save statistics to log file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')

    def get_current_stats(self) -> Dict:
        """Get current GPU statistics."""
        if not self.monitoring:
            # Collect stats once if not monitoring
            return self._collect_stats()
        return self.current_stats

    def get_memory_summary(self) -> Dict[int, Dict[str, float]]:
        """Get memory usage summary for all GPUs."""
        summary = {}

        for i in range(self.gpu_count):
            allocated = cuda.memory_allocated(i) / (1024 ** 3)
            reserved = cuda.memory_reserved(i) / (1024 ** 3)
            total = cuda.get_device_properties(i).total_memory / (1024 ** 3)

            summary[i] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': total - reserved,
                'total_gb': total,
                'utilization_pct': (allocated / total) * 100,
            }

        return summary

    def get_utilization_summary(self) -> Dict[int, Dict[str, float]]:
        """Get utilization summary from history."""
        summary = {}

        for gpu_id, history in self.history.items():
            if not history:
                continue

            gpu_utils = [s.get('gpu_utilization', 0) for s in history]
            mem_utils = [s.get('memory_utilization', 0) for s in history]

            summary[gpu_id] = {
                'avg_gpu_utilization': np.mean(gpu_utils),
                'max_gpu_utilization': np.max(gpu_utils),
                'min_gpu_utilization': np.min(gpu_utils),
                'avg_memory_utilization': np.mean(mem_utils),
                'max_memory_utilization': np.max(mem_utils),
            }

        return summary

    def detect_bottlenecks(self) -> Dict[str, Any]:
        """Detect performance bottlenecks."""
        bottlenecks = {
            'gpu_underutilized': [],
            'memory_pressure': [],
            'thermal_throttling': [],
            'recommendations': []
        }

        utilization = self.get_utilization_summary()
        memory = self.get_memory_summary()

        for gpu_id in range(self.gpu_count):
            # Check GPU underutilization
            if gpu_id in utilization:
                avg_util = utilization[gpu_id].get('avg_gpu_utilization', 100)
                if avg_util < 70:
                    bottlenecks['gpu_underutilized'].append(gpu_id)
                    bottlenecks['recommendations'].append(
                        f"GPU {gpu_id} underutilized ({avg_util:.1f}%). "
                        "Consider increasing batch size or using data parallelism."
                    )

            # Check memory pressure
            if gpu_id in memory:
                mem_util = memory[gpu_id]['utilization_pct']
                if mem_util > 90:
                    bottlenecks['memory_pressure'].append(gpu_id)
                    bottlenecks['recommendations'].append(
                        f"GPU {gpu_id} high memory usage ({mem_util:.1f}%). "
                        "Consider gradient checkpointing or reducing batch size."
                    )

            # Check temperature
            if self.current_stats and gpu_id in self.current_stats:
                temp = self.current_stats[gpu_id].get('temperature_c', 0)
                if temp > 80:
                    bottlenecks['thermal_throttling'].append(gpu_id)
                    bottlenecks['recommendations'].append(
                        f"GPU {gpu_id} running hot ({temp}°C). "
                        "May experience thermal throttling."
                    )

        return bottlenecks

    def optimize_batch_size(
        self,
        model: torch.nn.Module,
        initial_batch_size: int = 32,
        max_batch_size: int = 512,
        target_memory_usage: float = 0.8
    ) -> int:
        """
        Find optimal batch size for given model.

        Args:
            model: Model to optimize for
            initial_batch_size: Starting batch size
            max_batch_size: Maximum batch size to try
            target_memory_usage: Target memory utilization (0-1)

        Returns:
            Optimal batch size
        """
        if not self.gpu_available:
            return initial_batch_size

        device = next(model.parameters()).device
        device_id = device.index if device.type == 'cuda' else 0

        # Binary search for optimal batch size
        low = 1
        high = max_batch_size
        optimal = initial_batch_size

        while low <= high:
            batch_size = (low + high) // 2

            try:
                # Clear cache
                cuda.empty_cache()
                cuda.synchronize()

                # Test forward pass
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
                with torch.no_grad():
                    _ = model(dummy_input)

                # Check memory usage
                mem_allocated = cuda.memory_allocated(device_id)
                mem_total = cuda.get_device_properties(device_id).total_memory
                mem_usage = mem_allocated / mem_total

                if mem_usage < target_memory_usage:
                    optimal = batch_size
                    low = batch_size + 1
                else:
                    high = batch_size - 1

                del dummy_input
                cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = batch_size - 1
                    cuda.empty_cache()
                else:
                    raise

        return optimal

    def profile_forward_pass(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Profile model forward pass performance.

        Args:
            model: Model to profile
            input_shape: Input tensor shape
            num_iterations: Number of iterations for timing

        Returns:
            Profiling results
        """
        if not self.gpu_available:
            return {}

        device = next(model.parameters()).device

        # Warmup
        dummy_input = torch.randn(*input_shape).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)

        # Time forward passes
        cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)

        cuda.synchronize()
        total_time = time.time() - start_time

        # Memory stats
        peak_memory = cuda.max_memory_allocated(device.index if device.type == 'cuda' else 0)

        return {
            'avg_forward_time_ms': (total_time / num_iterations) * 1000,
            'throughput_samples_per_sec': (num_iterations * input_shape[0]) / total_time,
            'peak_memory_gb': peak_memory / (1024 ** 3),
        }

    def get_report(self) -> str:
        """Generate comprehensive GPU monitoring report."""
        lines = ["GPU Monitoring Report", "=" * 50]

        # Current stats
        current = self.get_current_stats()
        lines.append(f"\nTimestamp: {current.get('datetime', 'N/A')}")

        # Per-GPU stats
        for i in range(self.gpu_count):
            if i in current:
                gpu = current[i]
                lines.append(f"\nGPU {i}: {gpu.get('name', 'Unknown')}")
                lines.append(f"  Memory: {gpu.get('memory_allocated_gb', 0):.2f}/"
                           f"{gpu.get('memory_total_gb', 0):.2f} GB "
                           f"({gpu.get('memory_utilization', 0):.1f}%)")
                lines.append(f"  Utilization: {gpu.get('gpu_utilization', 0):.1f}%")
                lines.append(f"  Temperature: {gpu.get('temperature_c', 0):.1f}°C")
                lines.append(f"  Power: {gpu.get('power_draw_w', 0):.1f}W")

        # Bottleneck analysis
        bottlenecks = self.detect_bottlenecks()
        if bottlenecks['recommendations']:
            lines.append("\nRecommendations:")
            for rec in bottlenecks['recommendations']:
                lines.append(f"  - {rec}")

        return "\n".join(lines)