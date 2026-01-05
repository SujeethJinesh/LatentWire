#!/usr/bin/env python3
"""
GPU Utilization Monitoring System for LatentWire Training

Tracks real-time GPU metrics to identify bottlenecks and ensure maximum utilization.
Runs alongside training to provide insights into resource usage patterns.

Usage:
    # Run standalone monitoring
    python telepathy/gpu_monitor.py --output_dir runs/monitor --interval 1.0

    # Integrate with training
    from telepathy.gpu_monitor import GPUMonitor
    monitor = GPUMonitor(output_dir="runs/experiment")
    monitor.start()
    # ... training code ...
    monitor.stop()
    stats = monitor.get_summary()
"""

import os
import sys
import json
import time
import threading
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings

# Try importing nvidia-ml-py for robust GPU monitoring
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    warnings.warn("pynvml not installed. Using nvidia-smi fallback. Install with: pip install nvidia-ml-py")

import torch


class GPUMetrics:
    """Container for GPU metrics at a point in time."""

    def __init__(
        self,
        timestamp: float,
        gpu_id: int,
        utilization: float,
        memory_used_gb: float,
        memory_total_gb: float,
        temperature: float,
        power_draw: float,
        sm_clock: int,
        memory_clock: int,
    ):
        self.timestamp = timestamp
        self.gpu_id = gpu_id
        self.utilization = utilization
        self.memory_used_gb = memory_used_gb
        self.memory_total_gb = memory_total_gb
        self.memory_percent = (memory_used_gb / memory_total_gb * 100) if memory_total_gb > 0 else 0
        self.temperature = temperature
        self.power_draw = power_draw
        self.sm_clock = sm_clock
        self.memory_clock = memory_clock

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "gpu_id": self.gpu_id,
            "utilization": self.utilization,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "memory_percent": self.memory_percent,
            "temperature": self.temperature,
            "power_draw": self.power_draw,
            "sm_clock": self.sm_clock,
            "memory_clock": self.memory_clock,
        }


class GPUMonitor:
    """
    Real-time GPU monitoring system with bottleneck detection.

    Features:
    - Tracks GPU utilization, memory, temperature, power
    - Identifies bottlenecks (low utilization patterns)
    - Logs metrics to file for analysis
    - Provides optimization recommendations
    """

    def __init__(
        self,
        output_dir: str = "runs/monitor",
        interval: float = 1.0,
        window_size: int = 60,
        alert_threshold: float = 80.0,
        verbose: bool = True,
    ):
        """
        Args:
            output_dir: Directory to save monitoring logs
            interval: Sampling interval in seconds
            window_size: Window size for rolling statistics
            alert_threshold: Alert if utilization drops below this %
            verbose: Print alerts to console
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.interval = interval
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.verbose = verbose

        # Monitoring state
        self.monitoring = False
        self.thread = None
        self.start_time = None

        # Metrics storage
        self.metrics_history = []
        self.recent_metrics = deque(maxlen=window_size)

        # Alert tracking
        self.low_util_count = 0
        self.alerts = []

        # Initialize NVML if available
        if HAS_NVML:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
        else:
            self.device_count = torch.cuda.device_count()
            self.handles = None

        # Log file setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"gpu_monitor_{timestamp}.jsonl"
        self.summary_file = self.output_dir / f"gpu_summary_{timestamp}.json"

        if self.verbose:
            print(f"GPU Monitor initialized")
            print(f"  - GPUs detected: {self.device_count}")
            print(f"  - Log file: {self.log_file}")
            print(f"  - Alert threshold: {self.alert_threshold}%")

    def _get_gpu_metrics_nvml(self, gpu_id: int) -> GPUMetrics:
        """Get GPU metrics using NVML (nvidia-ml-py)."""
        handle = self.handles[gpu_id]

        # Get utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Get temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temp = 0

        # Get power draw
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
        except:
            power = 0

        # Get clocks
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except:
            sm_clock = 0
            mem_clock = 0

        return GPUMetrics(
            timestamp=time.time(),
            gpu_id=gpu_id,
            utilization=util.gpu,
            memory_used_gb=mem_info.used / 1e9,
            memory_total_gb=mem_info.total / 1e9,
            temperature=temp,
            power_draw=power,
            sm_clock=sm_clock,
            memory_clock=mem_clock,
        )

    def _get_gpu_metrics_smi(self, gpu_id: int) -> GPUMetrics:
        """Get GPU metrics using nvidia-smi (fallback)."""
        try:
            # Query specific metrics for efficiency
            query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory"
            cmd = f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits -i {gpu_id}"

            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

            values = result.stdout.strip().split(', ')

            return GPUMetrics(
                timestamp=time.time(),
                gpu_id=gpu_id,
                utilization=float(values[0]) if values[0] != 'N/A' else 0,
                memory_used_gb=float(values[1]) / 1024 if values[1] != 'N/A' else 0,
                memory_total_gb=float(values[2]) / 1024 if values[2] != 'N/A' else 0,
                temperature=float(values[3]) if values[3] != 'N/A' else 0,
                power_draw=float(values[4]) if values[4] != 'N/A' else 0,
                sm_clock=int(values[5]) if values[5] != 'N/A' else 0,
                memory_clock=int(values[6]) if values[6] != 'N/A' else 0,
            )
        except Exception as e:
            # Return empty metrics on failure
            return GPUMetrics(
                timestamp=time.time(),
                gpu_id=gpu_id,
                utilization=0,
                memory_used_gb=0,
                memory_total_gb=0,
                temperature=0,
                power_draw=0,
                sm_clock=0,
                memory_clock=0,
            )

    def get_gpu_metrics(self, gpu_id: int) -> GPUMetrics:
        """Get current GPU metrics."""
        if HAS_NVML:
            return self._get_gpu_metrics_nvml(gpu_id)
        else:
            return self._get_gpu_metrics_smi(gpu_id)

    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.monitoring:
            try:
                # Collect metrics for all GPUs
                for gpu_id in range(self.device_count):
                    metrics = self.get_gpu_metrics(gpu_id)

                    # Store metrics
                    self.metrics_history.append(metrics)
                    self.recent_metrics.append(metrics)

                    # Write to log file
                    with open(self.log_file, 'a') as f:
                        f.write(json.dumps(metrics.to_dict()) + '\n')

                    # Check for low utilization
                    if metrics.utilization < self.alert_threshold:
                        self.low_util_count += 1

                        # Generate alert after sustained low utilization
                        if self.low_util_count >= 10:  # 10 seconds of low util
                            alert = self._generate_alert(gpu_id, metrics)
                            self.alerts.append(alert)

                            if self.verbose:
                                print(f"\n⚠️  GPU Alert: {alert['message']}")

                            self.low_util_count = 0
                    else:
                        self.low_util_count = max(0, self.low_util_count - 1)

                # Sleep until next sample
                time.sleep(self.interval)

            except Exception as e:
                if self.verbose:
                    print(f"Monitor error: {e}")
                time.sleep(self.interval)

    def _generate_alert(self, gpu_id: int, metrics: GPUMetrics) -> Dict:
        """Generate alert with bottleneck analysis."""
        # Analyze recent history for patterns
        recent_gpu_metrics = [m for m in self.recent_metrics if m.gpu_id == gpu_id]

        if len(recent_gpu_metrics) > 0:
            avg_util = sum(m.utilization for m in recent_gpu_metrics) / len(recent_gpu_metrics)
            avg_mem = sum(m.memory_percent for m in recent_gpu_metrics) / len(recent_gpu_metrics)
        else:
            avg_util = metrics.utilization
            avg_mem = metrics.memory_percent

        # Determine likely bottleneck
        bottleneck = "unknown"
        recommendation = ""

        if avg_util < 50 and avg_mem < 50:
            bottleneck = "data_loading"
            recommendation = "Increase DataLoader workers, use prefetching, or optimize data pipeline"
        elif avg_util < 50 and avg_mem > 80:
            bottleneck = "memory_bound"
            recommendation = "Reduce batch size or optimize memory usage (gradient checkpointing)"
        elif avg_util > 80 and metrics.temperature > 80:
            bottleneck = "thermal_throttling"
            recommendation = "Check cooling, reduce power limit, or add delays"
        elif avg_util < self.alert_threshold:
            bottleneck = "cpu_bound"
            recommendation = "Profile CPU operations, use torch.compile, or optimize preprocessing"

        alert = {
            "timestamp": metrics.timestamp,
            "gpu_id": gpu_id,
            "utilization": metrics.utilization,
            "memory_percent": metrics.memory_percent,
            "temperature": metrics.temperature,
            "bottleneck": bottleneck,
            "message": f"GPU {gpu_id} utilization at {metrics.utilization:.1f}%",
            "recommendation": recommendation,
        }

        return alert

    def start(self):
        """Start monitoring in background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        if self.verbose:
            print(f"GPU monitoring started (interval={self.interval}s)")

    def stop(self):
        """Stop monitoring and save summary."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5)

        # Save summary
        summary = self.get_summary()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print(f"\nGPU monitoring stopped")
            print(f"Summary saved to: {self.summary_file}")
            self._print_summary(summary)

    def get_summary(self) -> Dict:
        """Generate monitoring summary with statistics and recommendations."""
        if not self.metrics_history:
            return {"error": "No metrics collected"}

        total_time = time.time() - self.start_time if self.start_time else 0

        # Calculate per-GPU statistics
        gpu_stats = {}
        for gpu_id in range(self.device_count):
            gpu_metrics = [m for m in self.metrics_history if m.gpu_id == gpu_id]

            if gpu_metrics:
                gpu_stats[gpu_id] = {
                    "samples": len(gpu_metrics),
                    "avg_utilization": sum(m.utilization for m in gpu_metrics) / len(gpu_metrics),
                    "min_utilization": min(m.utilization for m in gpu_metrics),
                    "max_utilization": max(m.utilization for m in gpu_metrics),
                    "avg_memory_percent": sum(m.memory_percent for m in gpu_metrics) / len(gpu_metrics),
                    "max_memory_gb": max(m.memory_used_gb for m in gpu_metrics),
                    "avg_temperature": sum(m.temperature for m in gpu_metrics) / len(gpu_metrics),
                    "max_temperature": max(m.temperature for m in gpu_metrics),
                    "avg_power_draw": sum(m.power_draw for m in gpu_metrics) / len(gpu_metrics),
                    "time_below_threshold": sum(1 for m in gpu_metrics if m.utilization < self.alert_threshold) * self.interval,
                }

        # Overall statistics
        all_utils = [m.utilization for m in self.metrics_history]
        overall_avg_util = sum(all_utils) / len(all_utils) if all_utils else 0

        # Bottleneck analysis
        bottleneck_counts = {}
        for alert in self.alerts:
            bottleneck = alert["bottleneck"]
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1

        # Generate recommendations
        recommendations = []

        if overall_avg_util < 80:
            recommendations.append(f"⚠️  Average GPU utilization only {overall_avg_util:.1f}% - significant optimization potential")

        if "data_loading" in bottleneck_counts and bottleneck_counts["data_loading"] > 5:
            recommendations.append("📊 Frequent data loading bottlenecks detected - increase DataLoader workers or use persistent workers")

        if "memory_bound" in bottleneck_counts:
            recommendations.append("💾 Memory bottlenecks detected - consider gradient checkpointing or smaller batch size")

        if "cpu_bound" in bottleneck_counts:
            recommendations.append("🖥️  CPU bottlenecks detected - profile with torch.profiler to identify slow operations")

        # Check for GPU imbalance in multi-GPU setup
        if len(gpu_stats) > 1:
            utils = [stats["avg_utilization"] for stats in gpu_stats.values()]
            if max(utils) - min(utils) > 20:
                recommendations.append("⚖️  GPU utilization imbalance detected - check data parallelism or load distribution")

        summary = {
            "monitoring_duration_seconds": total_time,
            "total_samples": len(self.metrics_history),
            "overall_avg_utilization": overall_avg_util,
            "gpu_stats": gpu_stats,
            "alerts_count": len(self.alerts),
            "bottleneck_counts": bottleneck_counts,
            "recommendations": recommendations,
            "log_file": str(self.log_file),
        }

        return summary

    def _print_summary(self, summary: Dict):
        """Print formatted summary to console."""
        print("\n" + "="*60)
        print("GPU MONITORING SUMMARY")
        print("="*60)

        print(f"\nDuration: {summary['monitoring_duration_seconds']:.1f}s")
        print(f"Samples: {summary['total_samples']}")
        print(f"Overall Avg Utilization: {summary['overall_avg_utilization']:.1f}%")

        print("\nPer-GPU Statistics:")
        for gpu_id, stats in summary['gpu_stats'].items():
            print(f"\n  GPU {gpu_id}:")
            print(f"    Utilization: {stats['avg_utilization']:.1f}% (min={stats['min_utilization']:.1f}%, max={stats['max_utilization']:.1f}%)")
            print(f"    Memory: {stats['avg_memory_percent']:.1f}% (max={stats['max_memory_gb']:.2f}GB)")
            print(f"    Temperature: {stats['avg_temperature']:.1f}°C (max={stats['max_temperature']:.1f}°C)")
            print(f"    Time below {self.alert_threshold}%: {stats['time_below_threshold']:.1f}s")

        if summary['bottleneck_counts']:
            print("\nBottleneck Analysis:")
            for bottleneck, count in summary['bottleneck_counts'].items():
                print(f"  {bottleneck}: {count} occurrences")

        if summary['recommendations']:
            print("\nOptimization Recommendations:")
            for rec in summary['recommendations']:
                print(f"  {rec}")

        print("\n" + "="*60)


class TrainingMonitor:
    """
    Integration helper for monitoring during training.

    Usage:
        from telepathy.gpu_monitor import TrainingMonitor

        with TrainingMonitor(output_dir="runs/exp") as monitor:
            # Training code here
            for epoch in range(epochs):
                train_epoch()

                # Get current stats
                stats = monitor.get_current_stats()
                print(f"GPU util: {stats['utilization']:.1f}%")
    """

    def __init__(self, output_dir: str = "runs/monitor", interval: float = 1.0, verbose: bool = False):
        self.monitor = GPUMonitor(
            output_dir=output_dir,
            interval=interval,
            verbose=verbose
        )

    def __enter__(self):
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()

    def get_current_stats(self) -> Dict:
        """Get current GPU statistics."""
        if not self.monitor.recent_metrics:
            return {"utilization": 0, "memory_percent": 0}

        recent = list(self.monitor.recent_metrics)[-self.monitor.device_count:]

        return {
            "utilization": sum(m.utilization for m in recent) / len(recent),
            "memory_percent": sum(m.memory_percent for m in recent) / len(recent),
            "temperature": max(m.temperature for m in recent),
        }

    def check_health(self) -> Tuple[bool, str]:
        """Check if GPUs are healthy and well-utilized."""
        stats = self.get_current_stats()

        if stats["utilization"] < 50:
            return False, f"Low GPU utilization: {stats['utilization']:.1f}%"

        if stats["temperature"] > 85:
            return False, f"High GPU temperature: {stats['temperature']:.1f}°C"

        if stats["memory_percent"] > 95:
            return False, f"GPU memory nearly full: {stats['memory_percent']:.1f}%"

        return True, "GPUs healthy"


def main():
    """Standalone monitoring mode."""
    parser = argparse.ArgumentParser(description="GPU Utilization Monitor")
    parser.add_argument("--output_dir", type=str, default="runs/monitor",
                        help="Directory to save monitoring logs")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Monitoring duration in seconds (None = until Ctrl+C)")
    parser.add_argument("--alert_threshold", type=float, default=80.0,
                        help="Alert if utilization drops below this %")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console output")

    args = parser.parse_args()

    print("Starting GPU Monitor...")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling interval: {args.interval}s")
    print(f"Alert threshold: {args.alert_threshold}%")

    if args.duration:
        print(f"Duration: {args.duration}s")
    else:
        print("Duration: Until Ctrl+C")

    print("\nPress Ctrl+C to stop monitoring\n")

    # Create monitor
    monitor = GPUMonitor(
        output_dir=args.output_dir,
        interval=args.interval,
        alert_threshold=args.alert_threshold,
        verbose=not args.quiet
    )

    # Start monitoring
    monitor.start()

    try:
        if args.duration:
            time.sleep(args.duration)
        else:
            # Run until interrupted
            while True:
                time.sleep(1)

                # Print live stats every 10 seconds
                if int(time.time()) % 10 == 0 and not args.quiet:
                    stats = monitor.get_summary()
                    if "overall_avg_utilization" in stats:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Avg GPU Util: {stats['overall_avg_utilization']:.1f}%",
                              end='\r')

    except KeyboardInterrupt:
        print("\n\nStopping monitor...")

    finally:
        monitor.stop()

        # Print final summary
        summary = monitor.get_summary()
        if "overall_avg_utilization" in summary:
            print(f"\nFinal Average GPU Utilization: {summary['overall_avg_utilization']:.1f}%")

            if summary['recommendations']:
                print("\n🎯 Key Recommendations:")
                for rec in summary['recommendations']:
                    print(f"  {rec}")


if __name__ == "__main__":
    main()