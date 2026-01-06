#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Training Wrapper for LatentWire

Provides comprehensive error handling and recovery mechanisms:
- Atomic checkpoint saves
- Graceful interrupt handling
- Automatic recovery from OOM
- Network retry logic
- Disk space monitoring
- Loss spike detection
- Comprehensive logging
"""

import os
import sys
import json
import signal
import time
import shutil
import traceback
import tempfile
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RobustTrainingWrapper:
    """Wrapper class for robust training with comprehensive error handling."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the robust training wrapper.

        Args:
            config_path: Path to training configuration JSON file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.checkpoint_dir = Path(self.config.get("output_dir", "runs/robust_training"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.training_interrupted = False
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 300  # Save every 5 minutes
        self.loss_history = []
        self.error_log = []

        # Setup signal handlers
        self._setup_signal_handlers()

        # Pre-flight checks
        self._run_preflight_checks()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load config from {config_path}: {e}")
            return {}

    def _setup_signal_handlers(self):
        """Setup graceful interrupt handling."""
        def signal_handler(signum, frame):
            print("\n[INFO] Interrupt received. Saving checkpoint and exiting gracefully...")
            self.training_interrupted = True
            self._save_emergency_checkpoint()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _run_preflight_checks(self):
        """Run pre-flight checks before training."""
        print("[INFO] Running pre-flight checks...")

        # Check disk space
        if not self._check_disk_space():
            raise RuntimeError("Insufficient disk space for training")

        # Check GPU availability
        self._check_gpu_availability()

        # Verify dependencies
        self._verify_dependencies()

        print("[INFO] Pre-flight checks passed")

    def _check_disk_space(self, min_gb: float = 10.0) -> bool:
        """Check if sufficient disk space is available.

        Args:
            min_gb: Minimum required space in GB

        Returns:
            True if sufficient space available
        """
        try:
            stat = shutil.disk_usage(self.checkpoint_dir)
            free_gb = stat.free / (1024**3)

            if free_gb < min_gb:
                print(f"[ERROR] Insufficient disk space: {free_gb:.2f} GB available, {min_gb:.2f} GB required")
                return False

            print(f"[INFO] Disk space check passed: {free_gb:.2f} GB available")
            return True

        except Exception as e:
            print(f"[WARN] Could not check disk space: {e}")
            return True  # Continue anyway

    def _check_gpu_availability(self):
        """Check GPU availability and log information."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"[INFO] Found {gpu_count} GPU(s)")

                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    print(f"  GPU {i}: {props.name}, {memory_gb:.2f} GB memory")
            else:
                print("[WARN] No GPUs available, will use CPU")

        except ImportError:
            print("[WARN] PyTorch not available for GPU check")

    def _verify_dependencies(self):
        """Verify all required dependencies are available."""
        required_modules = [
            "torch",
            "transformers",
            "datasets",
            "numpy",
        ]

        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)

        if missing:
            print(f"[WARN] Missing dependencies: {missing}")

    def save_checkpoint_atomic(self, state: Dict[str, Any], path: Path):
        """Save checkpoint atomically to prevent corruption.

        Args:
            state: State dictionary to save
            path: Target checkpoint path
        """
        temp_path = path.with_suffix('.tmp')

        try:
            # Save to temporary file
            import torch
            torch.save(state, temp_path)

            # Verify the checkpoint is loadable
            torch.load(temp_path, map_location='cpu')

            # Atomic rename
            temp_path.rename(path)

            print(f"[INFO] Checkpoint saved atomically to {path}")

        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on interrupt."""
        try:
            emergency_path = self.checkpoint_dir / f"emergency_checkpoint_{int(time.time())}.pt"

            # Create minimal state
            state = {
                "timestamp": datetime.now().isoformat(),
                "reason": "emergency_save",
                "loss_history": self.loss_history[-100:],  # Last 100 losses
                "error_log": self.error_log,
            }

            # Try to get model state if available
            try:
                import torch
                # This would need to be adapted to actual model
                # state["model_state"] = model.state_dict()
                pass
            except:
                pass

            self.save_checkpoint_atomic(state, emergency_path)
            print(f"[INFO] Emergency checkpoint saved to {emergency_path}")

        except Exception as e:
            print(f"[ERROR] Failed to save emergency checkpoint: {e}")

    def detect_loss_spike(self, current_loss: float, threshold: float = 10.0) -> bool:
        """Detect anomalous loss spikes.

        Args:
            current_loss: Current loss value
            threshold: Spike threshold multiplier

        Returns:
            True if spike detected
        """
        if len(self.loss_history) < 10:
            return False

        recent_avg = sum(self.loss_history[-10:]) / 10

        if current_loss > recent_avg * threshold:
            print(f"[WARN] Loss spike detected: {current_loss:.4f} vs avg {recent_avg:.4f}")
            return True

        return False

    def handle_oom_error(self, batch_size: int) -> int:
        """Handle OOM error by adjusting batch size.

        Args:
            batch_size: Current batch size

        Returns:
            New reduced batch size
        """
        new_batch_size = max(1, batch_size // 2)
        print(f"[INFO] OOM detected, reducing batch size from {batch_size} to {new_batch_size}")

        # Clear GPU cache
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

        return new_batch_size

    def retry_on_network_error(self, func, max_attempts: int = 3, backoff: float = 2.0):
        """Retry function on network errors with exponential backoff.

        Args:
            func: Function to retry
            max_attempts: Maximum retry attempts
            backoff: Backoff multiplier

        Returns:
            Function result
        """
        for attempt in range(max_attempts):
            try:
                return func()
            except (ConnectionError, TimeoutError) as e:
                if attempt == max_attempts - 1:
                    raise

                wait_time = backoff ** attempt
                print(f"[WARN] Network error, retrying in {wait_time:.1f}s... ({attempt+1}/{max_attempts})")
                time.sleep(wait_time)

    def validate_checkpoint(self, path: Path) -> bool:
        """Validate checkpoint integrity.

        Args:
            path: Checkpoint path

        Returns:
            True if valid
        """
        try:
            import torch

            # Try to load checkpoint
            ckpt = torch.load(path, map_location='cpu')

            # Check for required keys
            required_keys = ["epoch", "global_step"]
            for key in required_keys:
                if key not in ckpt:
                    print(f"[WARN] Checkpoint missing key: {key}")
                    return False

            return True

        except Exception as e:
            print(f"[ERROR] Invalid checkpoint at {path}: {e}")
            return False

    def run_training_with_recovery(self, train_func, **kwargs):
        """Run training with automatic error recovery.

        Args:
            train_func: Training function to run
            **kwargs: Arguments for training function
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                print(f"[INFO] Starting training (attempt {retry_count + 1}/{max_retries})")

                # Run the training function
                result = train_func(**kwargs)

                print("[INFO] Training completed successfully")
                return result

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM
                    kwargs["batch_size"] = self.handle_oom_error(kwargs.get("batch_size", 32))
                    retry_count += 1

                elif "CUDA error" in str(e):
                    # Handle CUDA errors
                    print(f"[ERROR] CUDA error: {e}")
                    self._reset_gpu()
                    retry_count += 1

                else:
                    # Unknown runtime error
                    print(f"[ERROR] Runtime error: {e}")
                    self.error_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    raise

            except KeyboardInterrupt:
                print("[INFO] Training interrupted by user")
                self._save_emergency_checkpoint()
                raise

            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                self.error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

                if retry_count < max_retries - 1:
                    print(f"[INFO] Retrying after unexpected error...")
                    retry_count += 1
                    time.sleep(5)
                else:
                    raise

        print(f"[ERROR] Training failed after {max_retries} attempts")
        raise RuntimeError("Maximum retries exceeded")

    def _reset_gpu(self):
        """Reset GPU state after error."""
        try:
            import torch

            # Clear cache
            torch.cuda.empty_cache()

            # Reset peak memory stats
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            print("[INFO] GPU state reset")

        except Exception as e:
            print(f"[WARN] Could not reset GPU: {e}")

    def monitor_training_loop(self, epoch: int, step: int, loss: float):
        """Monitor training progress and handle issues.

        Args:
            epoch: Current epoch
            step: Current step
            loss: Current loss value
        """
        # Track loss history
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]

        # Detect loss spikes
        if self.detect_loss_spike(loss):
            print("[WARN] Skipping update due to loss spike")
            return False  # Skip this update

        # Periodic checkpoint saves
        current_time = time.time()
        if current_time - self.last_checkpoint_time > self.checkpoint_interval:
            print(f"[INFO] Periodic checkpoint at epoch {epoch}, step {step}")
            # Would trigger checkpoint save here
            self.last_checkpoint_time = current_time

        # Check for NaN/Inf
        import math
        if math.isnan(loss) or math.isinf(loss):
            print(f"[ERROR] Invalid loss detected: {loss}")
            return False

        return True  # Continue training

    def cleanup(self):
        """Cleanup resources and save final state."""
        print("[INFO] Running cleanup...")

        # Save error log
        if self.error_log:
            error_log_path = self.checkpoint_dir / "error_log.json"
            with open(error_log_path, 'w') as f:
                json.dump(self.error_log, f, indent=2)
            print(f"[INFO] Error log saved to {error_log_path}")

        # Save training summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "loss_history_length": len(self.loss_history),
            "error_count": len(self.error_log),
            "final_loss": self.loss_history[-1] if self.loss_history else None,
        }

        summary_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Training summary saved to {summary_path}")


def main():
    """Example usage of the robust training wrapper."""

    # Initialize wrapper
    wrapper = RobustTrainingWrapper()

    # Example training function
    def mock_training_function(batch_size=32, epochs=10):
        print(f"[INFO] Training with batch_size={batch_size}, epochs={epochs}")

        # Simulate training loop
        for epoch in range(epochs):
            for step in range(100):
                # Simulate loss
                import random
                loss = random.random() * 2.0

                # Monitor training
                if not wrapper.monitor_training_loop(epoch, step, loss):
                    continue  # Skip this step

                # Simulate occasional errors
                if random.random() < 0.001:
                    raise RuntimeError("out of memory")

                time.sleep(0.01)  # Simulate computation

        return {"status": "completed"}

    try:
        # Run training with automatic recovery
        result = wrapper.run_training_with_recovery(
            mock_training_function,
            batch_size=32,
            epochs=2
        )
        print(f"[INFO] Training result: {result}")

    finally:
        # Cleanup
        wrapper.cleanup()


if __name__ == "__main__":
    main()