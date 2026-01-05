"""
Enhanced logging utilities for preemptible training.

Features:
- Structured logging with JSON format
- Automatic log rotation
- Performance metrics tracking
- Memory usage monitoring
- Training progress visualization
- Checkpoint event logging
- Error tracking and recovery
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque
import psutil
import torch


class PreemptibleLogger:
    """
    Advanced logger for preemptible training with structured logging,
    metrics tracking, and progress monitoring.
    """

    def __init__(
        self,
        log_file: Path,
        enable_console: bool = True,
        enable_json: bool = True,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
    ):
        """
        Initialize the preemptible logger.

        Args:
            log_file: Path to log file
            enable_console: Whether to log to console
            enable_json: Whether to use JSON format for structured logging
            max_file_size_mb: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.enable_json = enable_json
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count

        # Set up Python logger
        self.logger = logging.getLogger("PreemptibleTraining")
        self.logger.setLevel(logging.DEBUG)

        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

        # Formatters
        if enable_json:
            file_handler.setFormatter(self.JSONFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # Metrics tracking
        self.metrics_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.last_checkpoint_time = None

    class JSONFormatter(logging.Formatter):
        """JSON formatter for structured logging."""

        def format(self, record):
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
            }

            # Add extra fields if present
            if hasattr(record, 'metrics'):
                log_data['metrics'] = record.metrics
            if hasattr(record, 'checkpoint'):
                log_data['checkpoint'] = record.checkpoint

            return json.dumps(log_data)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional structured data."""
        extra = {}
        if kwargs:
            extra['metrics'] = kwargs
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        extra = {}
        if kwargs:
            extra['metrics'] = kwargs
        self.logger.warning(message, extra=extra)

    def error(self, message: str, exc_info=None, **kwargs) -> None:
        """Log error message."""
        extra = {}
        if kwargs:
            extra['metrics'] = kwargs
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        extra = {}
        if kwargs:
            extra['metrics'] = kwargs
        self.logger.debug(message, extra=extra)

    def log_metrics(
        self,
        epoch: int,
        batch: int,
        loss: float,
        lr: float,
        **extra_metrics
    ) -> None:
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'lr': lr,
            'timestamp': time.time(),
            **extra_metrics
        }

        self.metrics_history.append(metrics)

        # Log to file
        self.info(
            f"Training metrics - Epoch: {epoch}, Batch: {batch}, Loss: {loss:.4f}",
            **metrics
        )

    def log_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        global_step: int,
        reason: str = "periodic"
    ) -> None:
        """Log checkpoint save event."""
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'global_step': global_step,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        }

        self.last_checkpoint_time = time.time()

        self.logger.info(
            f"Checkpoint saved: {checkpoint_path}",
            extra={'checkpoint': checkpoint_info}
        )

    def log_memory_usage(self) -> Dict[str, float]:
        """Log current memory usage."""
        memory_info = {}

        # System memory
        mem = psutil.virtual_memory()
        memory_info['system_memory_gb'] = mem.used / (1024 ** 3)
        memory_info['system_memory_percent'] = mem.percent

        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                memory_info[f'gpu_{i}_allocated_gb'] = allocated
                memory_info[f'gpu_{i}_reserved_gb'] = reserved

        self.info("Memory usage", **memory_info)
        return memory_info

    def log_training_summary(
        self,
        total_epochs: int,
        total_steps: int,
        best_metrics: Dict[str, float]
    ) -> None:
        """Log training summary at completion."""
        elapsed_time = time.time() - self.start_time
        hours = elapsed_time / 3600

        summary = {
            'total_epochs': total_epochs,
            'total_steps': total_steps,
            'training_time_hours': hours,
            'best_metrics': best_metrics,
        }

        self.info("Training completed", **summary)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        if not self.metrics_history:
            return {}

        losses = [m['loss'] for m in self.metrics_history if 'loss' in m]
        lrs = [m['lr'] for m in self.metrics_history if 'lr' in m]

        return {
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'min_loss': min(losses) if losses else 0,
            'max_loss': max(losses) if losses else 0,
            'final_lr': lrs[-1] if lrs else 0,
            'total_entries': len(self.metrics_history),
        }


class ProgressTracker:
    """
    Track and visualize training progress with ETA estimation.
    """

    def __init__(self, total_epochs: int, total_steps: int):
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.current_epoch = 0
        self.current_step = 0
        self.epoch_start_time = None
        self.training_start_time = time.time()

        self.step_times = deque(maxlen=100)
        self.epoch_times = deque(maxlen=10)

    def start_epoch(self, epoch: int) -> None:
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def end_epoch(self) -> float:
        """Mark the end of an epoch and return epoch time."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0

    def update_step(self, step: int, batch_time: float) -> None:
        """Update step progress."""
        self.current_step = step
        self.step_times.append(batch_time)

    def get_eta(self) -> Dict[str, float]:
        """Calculate ETA for training completion."""
        if not self.step_times:
            return {'eta_hours': 0, 'eta_minutes': 0}

        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps * avg_step_time

        return {
            'eta_hours': eta_seconds / 3600,
            'eta_minutes': (eta_seconds % 3600) / 60,
            'steps_per_second': 1 / avg_step_time if avg_step_time > 0 else 0,
        }

    def get_progress_string(self) -> str:
        """Get formatted progress string."""
        progress_pct = (self.current_step / self.total_steps) * 100
        eta = self.get_eta()

        return (
            f"Progress: {self.current_step}/{self.total_steps} "
            f"({progress_pct:.1f}%) | "
            f"ETA: {eta['eta_hours']:.1f}h {eta['eta_minutes']:.0f}m | "
            f"Speed: {eta['steps_per_second']:.2f} steps/s"
        )


class ExperimentTracker:
    """
    Track experiments and results across multiple runs.
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.experiment_dir / "results.json"

        # Load existing results
        self.results = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)

    def add_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a new run to the tracker."""
        run_data = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'metadata': metadata or {},
        }

        self.results.append(run_data)
        self._save_results()

    def _save_results(self) -> None:
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_best_run(self, metric: str, mode: str = 'max') -> Optional[Dict]:
        """Get the best run based on a metric."""
        if not self.results:
            return None

        key_func = lambda r: r['metrics'].get(metric, float('-inf' if mode == 'max' else 'inf'))

        if mode == 'max':
            return max(self.results, key=key_func)
        else:
            return min(self.results, key=key_func)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        if not self.results:
            return {}

        all_metrics = {}
        for run in self.results:
            for metric, value in run['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        summary = {
            'total_runs': len(self.results),
            'metrics_summary': {}
        }

        for metric, values in all_metrics.items():
            summary['metrics_summary'][metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': torch.std(torch.tensor(values)).item() if len(values) > 1 else 0,
            }

        return summary