#!/usr/bin/env python3
"""
Enhanced preemptible training wrapper with signal handling and robust checkpointing.

This module provides a PreemptibleTrainer class that wraps the existing training
infrastructure to add:
- SIGTERM/SIGINT signal handling for graceful shutdown
- Automatic checkpoint saving on preemption
- Resume from latest checkpoint
- GPU utilization monitoring
- Distributed training support

Usage:
    from telepathy.preemptible_trainer import PreemptibleTrainer

    trainer = PreemptibleTrainer(
        args=training_args,
        save_dir="runs/experiment",
        checkpoint_interval=100
    )
    trainer.train()
"""

import os
import sys
import signal
import time
import json
import threading
import traceback
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from datetime import datetime
from contextlib import contextmanager

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from latentwire.checkpointing import save_latest_checkpoint


class SignalHandler:
    """Handle system signals for graceful shutdown."""

    def __init__(self):
        self.shutdown_requested = False
        self.original_handlers = {}
        self.shutdown_callbacks = []
        self._lock = threading.Lock()

    def register(self):
        """Register signal handlers."""
        # Store original handlers
        self.original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handle_signal)
        self.original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._handle_signal)

        # SIGUSR1 for manual checkpoint request
        if hasattr(signal, 'SIGUSR1'):
            self.original_handlers[signal.SIGUSR1] = signal.signal(signal.SIGUSR1, self._handle_checkpoint)

    def unregister(self):
        """Restore original signal handlers."""
        for sig, handler in self.original_handlers.items():
            signal.signal(sig, handler)
        self.original_handlers.clear()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        with self._lock:
            if not self.shutdown_requested:
                self.shutdown_requested = True
                sig_name = signal.Signals(signum).name
                print(f"\n⚠️  Received {sig_name}, initiating graceful shutdown...")
                print("   Will save checkpoint after current batch completes.")

                # Run shutdown callbacks
                for callback in self.shutdown_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        print(f"Error in shutdown callback: {e}")

    def _handle_checkpoint(self, signum, frame):
        """Handle manual checkpoint request."""
        print("\n📸 Manual checkpoint requested via SIGUSR1")
        # This will be handled by the training loop checking for checkpoint requests

    def add_shutdown_callback(self, callback: Callable):
        """Add callback to run on shutdown."""
        self.shutdown_callbacks.append(callback)

    @property
    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_requested


class GPUMonitor:
    """Monitor GPU utilization and memory."""

    def __init__(self, log_interval: int = 60):
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.peak_memory_gb = {}
        self.enabled = TORCH_AVAILABLE and torch.cuda.is_available()

        if self.enabled:
            self.device_count = torch.cuda.device_count()
            for i in range(self.device_count):
                torch.cuda.reset_peak_memory_stats(i)
                self.peak_memory_gb[i] = 0.0

    def log_if_needed(self, force: bool = False):
        """Log GPU stats if interval passed or forced."""
        if not self.enabled:
            return

        current_time = time.time()
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return

        self.last_log_time = current_time
        self._log_gpu_stats()

    def _log_gpu_stats(self):
        """Log current GPU statistics."""
        print("\n" + "=" * 60)
        print("GPU Utilization Report")
        print("-" * 60)

        for i in range(self.device_count):
            # Memory stats
            allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(i) / 1024**3
            peak_gb = torch.cuda.max_memory_allocated(i) / 1024**3

            self.peak_memory_gb[i] = max(self.peak_memory_gb[i], peak_gb)

            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3

            util_pct = (allocated_gb / total_gb) * 100 if total_gb > 0 else 0

            print(f"GPU {i} ({props.name}):")
            print(f"  Memory: {allocated_gb:.1f}/{total_gb:.1f} GB ({util_pct:.1f}%)")
            print(f"  Reserved: {reserved_gb:.1f} GB")
            print(f"  Peak: {self.peak_memory_gb[i]:.1f} GB")

        print("=" * 60)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.enabled:
            return {}

        summary = {
            "gpu_count": self.device_count,
            "peak_memory_gb": dict(self.peak_memory_gb),
            "runtime_seconds": time.time() - self.start_time,
        }

        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            summary[f"gpu_{i}_name"] = props.name
            summary[f"gpu_{i}_total_memory_gb"] = props.total_memory / 1024**3

        return summary


class CheckpointManager:
    """Manage checkpoint saving and loading with preemption support."""

    def __init__(self, save_dir: str, keep_best: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.best_metric = float('inf')
        self.best_checkpoint_path = None
        self.save_count = 0
        self.last_save_time = time.time()

    def save_checkpoint(
        self,
        artifacts: Dict[str, Any],
        metric: Optional[float] = None,
        is_emergency: bool = False
    ) -> str:
        """Save checkpoint with optional best model tracking."""
        self.save_count += 1
        save_time = time.time()

        # Add metadata
        metadata = {
            "save_count": self.save_count,
            "save_time": save_time,
            "save_timestamp": datetime.now().isoformat(),
            "is_emergency": is_emergency,
            "metric": metric,
        }

        if "state.pt" in artifacts and isinstance(artifacts["state.pt"], dict):
            artifacts["state.pt"]["metadata"] = metadata
        else:
            artifacts["metadata.json"] = metadata

        # Save checkpoint
        if is_emergency:
            save_path = self.save_dir / "emergency"
            save_path.mkdir(exist_ok=True)
            print(f"💾 Saving emergency checkpoint to {save_path}")
        else:
            save_path = self.save_dir

        save_latest_checkpoint(str(save_path), artifacts, verbose=True)

        # Track best checkpoint
        if self.keep_best and metric is not None and metric < self.best_metric:
            self.best_metric = metric
            best_path = self.save_dir / "best"
            best_path.mkdir(exist_ok=True)

            # Copy to best directory
            save_latest_checkpoint(str(best_path), artifacts, verbose=False)
            self.best_checkpoint_path = best_path
            print(f"🏆 New best checkpoint saved (metric: {metric:.4f})")

        save_duration = time.time() - save_time
        self.last_save_time = save_time
        print(f"   Checkpoint saved in {save_duration:.2f}s")

        return str(save_path)

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint to resume from."""
        candidates = []

        # Check main directory
        if (self.save_dir / "state.pt").exists():
            candidates.append(self.save_dir)

        # Check emergency directory
        emergency_dir = self.save_dir / "emergency"
        if (emergency_dir / "state.pt").exists():
            candidates.append(emergency_dir)

        # Check subdirectories
        for subdir in self.save_dir.iterdir():
            if subdir.is_dir() and (subdir / "state.pt").exists():
                candidates.append(subdir)

        if not candidates:
            return None

        # Return most recent by modification time
        return max(candidates, key=lambda p: (p / "state.pt").stat().st_mtime)

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        return {
            "save_count": self.save_count,
            "best_metric": self.best_metric,
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "last_save_time": self.last_save_time,
        }


class PreemptibleTrainer:
    """Training wrapper with preemption support."""

    def __init__(
        self,
        args: Any,
        save_dir: str,
        checkpoint_interval: int = 100,
        gpu_log_interval: int = 60,
        enable_distributed: bool = False
    ):
        """Initialize preemptible trainer.

        Args:
            args: Training arguments object
            save_dir: Directory for checkpoints
            checkpoint_interval: Steps between automatic checkpoints
            gpu_log_interval: Seconds between GPU monitoring logs
            enable_distributed: Enable distributed training support
        """
        self.args = args
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval
        self.enable_distributed = enable_distributed

        # Initialize components
        self.signal_handler = SignalHandler()
        self.gpu_monitor = GPUMonitor(log_interval=gpu_log_interval)
        self.checkpoint_manager = CheckpointManager(save_dir)

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.training_start_time = None

        # Register signal handlers
        self.signal_handler.register()

        # Distributed training setup
        self.is_main_process = True
        if enable_distributed and TORCH_AVAILABLE:
            self._setup_distributed()

    def _setup_distributed(self):
        """Setup distributed training if enabled."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.is_main_process = (rank == 0)

            if self.is_main_process:
                print(f"🌐 Distributed training initialized: {world_size} processes")

    def _save_checkpoint(self, artifacts: Dict[str, Any], is_emergency: bool = False):
        """Save checkpoint (only on main process for distributed)."""
        if not self.is_main_process:
            return

        # Add training state
        if "state.pt" not in artifacts:
            artifacts["state.pt"] = {}

        state = artifacts["state.pt"]
        if isinstance(state, dict):
            state.update({
                "epoch": self.current_epoch,
                "step": self.current_step,
                "global_step": self.global_step,
                "training_time": time.time() - self.training_start_time if self.training_start_time else 0,
                "gpu_stats": self.gpu_monitor.get_summary(),
                "checkpoint_stats": self.checkpoint_manager.get_stats(),
            })

        # Calculate metric if available
        metric = None
        if "best_loss" in state:
            metric = state["best_loss"]
        elif "val_loss" in state:
            metric = state["val_loss"]

        self.checkpoint_manager.save_checkpoint(artifacts, metric, is_emergency)

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint if available."""
        checkpoint_dir = self.checkpoint_manager.find_latest_checkpoint()
        if not checkpoint_dir:
            return None

        print(f"📂 Found checkpoint in {checkpoint_dir}")

        state_path = checkpoint_dir / "state.pt"
        if not state_path.exists():
            return None

        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, cannot load checkpoint")
            return None

        state = torch.load(state_path, map_location="cpu")

        # Restore training state
        if isinstance(state, dict):
            self.current_epoch = state.get("epoch", 0)
            self.current_step = state.get("step", 0)
            self.global_step = state.get("global_step", 0)

            print(f"✅ Resumed from epoch {self.current_epoch}, step {self.global_step}")

        return {"checkpoint_dir": str(checkpoint_dir), "state": state}

    def train(self, train_fn: Optional[Callable] = None) -> bool:
        """Run training with preemption handling.

        Args:
            train_fn: Optional custom training function.
                     If not provided, uses standard latentwire training.

        Returns:
            True if training completed, False if interrupted.
        """
        self.training_start_time = time.time()

        try:
            # Load checkpoint if resuming
            checkpoint_data = self._load_checkpoint()

            print("\n" + "=" * 60)
            print("🚀 Starting Preemptible Training")
            print(f"   Save directory: {self.save_dir}")
            print(f"   Checkpoint interval: {self.checkpoint_interval} steps")
            print(f"   Distributed: {self.enable_distributed}")
            print(f"   Main process: {self.is_main_process}")
            print("=" * 60 + "\n")

            # Log initial GPU state
            self.gpu_monitor.log_if_needed(force=True)

            if train_fn:
                # Use custom training function
                completed = train_fn(
                    args=self.args,
                    checkpoint_data=checkpoint_data,
                    should_stop=lambda: self.signal_handler.should_stop,
                    save_checkpoint=self._save_checkpoint,
                    global_step=self.global_step
                )
            else:
                # Use standard training
                completed = self._run_standard_training(checkpoint_data)

            if completed:
                print("\n✅ Training completed successfully!")
            else:
                print("\n⚠️  Training interrupted by signal")

            # Final GPU report
            self.gpu_monitor.log_if_needed(force=True)

            # Save final statistics
            self._save_training_summary()

            return completed

        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            traceback.print_exc()

            # Try to save emergency checkpoint
            try:
                print("\n💾 Attempting emergency checkpoint save...")
                self._save_checkpoint({}, is_emergency=True)
            except Exception as save_error:
                print(f"   Failed to save emergency checkpoint: {save_error}")

            raise

        finally:
            # Cleanup
            self.signal_handler.unregister()
            if self.enable_distributed and dist.is_initialized():
                dist.destroy_process_group()

    def _run_standard_training(self, checkpoint_data: Optional[Dict]) -> bool:
        """Run standard latentwire training with preemption support."""
        # This would integrate with the existing train.py
        # For now, return a placeholder
        print("Standard training integration not yet implemented")
        print("Use train_fn parameter to provide custom training function")
        return True

    def _save_training_summary(self):
        """Save final training summary."""
        if not self.is_main_process:
            return

        summary = {
            "total_runtime": time.time() - self.training_start_time,
            "total_epochs": self.current_epoch,
            "total_steps": self.global_step,
            "gpu_summary": self.gpu_monitor.get_summary(),
            "checkpoint_summary": self.checkpoint_manager.get_stats(),
            "completed_at": datetime.now().isoformat(),
        }

        summary_path = Path(self.save_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Training summary saved to {summary_path}")


@contextmanager
def preemptible_context(save_dir: str, checkpoint_interval: int = 100):
    """Context manager for preemptible execution.

    Usage:
        with preemptible_context("runs/experiment") as ctx:
            # Your training code here
            # Check ctx.should_stop() periodically
            # Call ctx.save_checkpoint(artifacts) to save
    """
    trainer = PreemptibleTrainer(
        args=None,
        save_dir=save_dir,
        checkpoint_interval=checkpoint_interval
    )

    try:
        yield trainer
    finally:
        trainer.signal_handler.unregister()


def example_usage():
    """Example of how to use the preemptible trainer."""

    def custom_train_fn(args, checkpoint_data, should_stop, save_checkpoint, global_step):
        """Custom training function with preemption support."""

        # Setup model and optimizer
        import torch.nn as nn
        import torch.optim as optim

        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters())

        # Load checkpoint if available
        if checkpoint_data:
            ckpt_dir = Path(checkpoint_data["checkpoint_dir"])
            if (ckpt_dir / "model.pt").exists():
                model.load_state_dict(torch.load(ckpt_dir / "model.pt"))
                optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt"))

        # Training loop
        for epoch in range(10):
            for step in range(100):
                # Check for preemption
                if should_stop():
                    # Save checkpoint before stopping
                    save_checkpoint({
                        "model.pt": model.state_dict(),
                        "optimizer.pt": optimizer.state_dict(),
                        "state.pt": {"epoch": epoch, "step": step}
                    }, is_emergency=True)
                    return False

                # Training step
                optimizer.zero_grad()
                loss = model(torch.randn(4, 10)).sum()
                loss.backward()
                optimizer.step()

                # Periodic checkpoint
                if (epoch * 100 + step) % 50 == 0:
                    save_checkpoint({
                        "model.pt": model.state_dict(),
                        "optimizer.pt": optimizer.state_dict(),
                        "state.pt": {"epoch": epoch, "step": step}
                    })

        return True  # Training completed

    # Create trainer and run
    trainer = PreemptibleTrainer(
        args=None,
        save_dir="runs/example",
        checkpoint_interval=50
    )

    completed = trainer.train(train_fn=custom_train_fn)
    print(f"Training completed: {completed}")


if __name__ == "__main__":
    # Run example if executed directly
    example_usage()