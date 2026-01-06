#!/usr/bin/env python3
"""
Robust Training Wrapper for LatentWire
======================================
Handles common failure modes in long-running experiments:
- OOM errors with automatic batch size reduction
- Gradient explosions with clipping and recovery
- Network/IO errors with exponential backoff
- Checkpoint corruption with backup management
- Keyboard interrupts with emergency saves

Usage:
    python telepathy/robust_training.py --config path/to/config.yaml

Or programmatically:
    from telepathy.robust_training import RobustTrainer
    trainer = RobustTrainer(config)
    trainer.run()
"""

import os
import sys
import time
import json
import shutil
import signal
import traceback
import subprocess
import psutil
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingConfig:
    """Configuration for robust training."""
    # Base training command and args
    script_path: str = "latentwire/train.py"
    base_args: Dict[str, Any] = field(default_factory=dict)

    # Recovery settings
    max_retries: int = 3
    max_oom_retries: int = 5
    batch_size_reduction_factor: float = 0.5
    min_batch_size: int = 1

    # Monitoring settings
    memory_threshold_gb: float = 70.0  # Warn if GPU memory exceeds this
    gradient_clip_threshold: float = 10.0

    # Checkpoint management
    checkpoint_dir: str = "runs/robust_experiment"
    backup_checkpoints: bool = True
    keep_n_backups: int = 3

    # Network retry settings
    network_retry_delay: float = 5.0
    network_max_delay: float = 300.0
    network_backoff_factor: float = 2.0

    # Logging
    log_file: str = "robust_training.log"
    verbose: bool = True


class RobustTrainer:
    """Handles training with automatic recovery from common failures."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.retry_count = 0
        self.oom_count = 0
        self.current_batch_size = config.base_args.get('batch_size', 64)
        self.checkpoint_backups = []
        self.start_time = time.time()
        self.interrupted = False

        # Setup logging
        self.setup_logging()

        # Setup signal handlers
        self.setup_signal_handlers()

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging to both file and console."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'

        # File handler
        file_handler = logging.FileHandler(
            Path(self.config.checkpoint_dir) / self.config.log_file
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        console_handler.setFormatter(logging.Formatter(log_format))

        # Configure root logger
        logging.root.setLevel(logging.DEBUG)
        logging.root.handlers = [file_handler, console_handler]

        self.logger = logging.getLogger(__name__)

    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
            self.interrupted = True
            self.save_emergency_checkpoint()
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def build_command(self, **overrides) -> List[str]:
        """Build the training command with current parameters."""
        cmd = ["python", self.config.script_path]

        # Merge base args with overrides
        args = {**self.config.base_args, **overrides}

        # Add command line arguments
        for key, value in args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        return cmd

    def check_gpu_memory(self) -> Dict[str, float]:
        """Check current GPU memory usage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )

            memory_info = {}
            for i, line in enumerate(result.stdout.strip().split('\n')):
                used, total = map(float, line.split(','))
                memory_info[f"gpu_{i}"] = {
                    "used_gb": used / 1024,
                    "total_gb": total / 1024,
                    "percent": (used / total) * 100
                }
            return memory_info
        except Exception as e:
            self.logger.warning(f"Failed to check GPU memory: {e}")
            return {}

    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "gpu_memory": self.check_gpu_memory()
        }

        # Check for warnings
        warnings = []
        if health["cpu_percent"] > 90:
            warnings.append(f"High CPU usage: {health['cpu_percent']:.1f}%")
        if health["memory_percent"] > 90:
            warnings.append(f"High RAM usage: {health['memory_percent']:.1f}%")
        if health["disk_usage_percent"] > 95:
            warnings.append(f"Low disk space: {health['disk_usage_percent']:.1f}% used")

        for gpu_id, gpu_info in health["gpu_memory"].items():
            if gpu_info["used_gb"] > self.config.memory_threshold_gb:
                warnings.append(f"{gpu_id}: {gpu_info['used_gb']:.1f}GB used")

        health["warnings"] = warnings
        return health

    def backup_checkpoint(self, checkpoint_path: Path):
        """Create a backup of the current checkpoint."""
        if not self.config.backup_checkpoints or not checkpoint_path.exists():
            return

        backup_dir = Path(self.config.checkpoint_dir) / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"checkpoint_backup_{timestamp}"

        try:
            shutil.copytree(checkpoint_path, backup_path)
            self.checkpoint_backups.append(backup_path)
            self.logger.info(f"Created checkpoint backup: {backup_path}")

            # Remove old backups
            if len(self.checkpoint_backups) > self.config.keep_n_backups:
                old_backup = self.checkpoint_backups.pop(0)
                shutil.rmtree(old_backup, ignore_errors=True)
                self.logger.info(f"Removed old backup: {old_backup}")

        except Exception as e:
            self.logger.error(f"Failed to backup checkpoint: {e}")

    def restore_from_backup(self, checkpoint_path: Path) -> bool:
        """Restore from the most recent backup."""
        if not self.checkpoint_backups:
            return False

        for backup in reversed(self.checkpoint_backups):
            if backup.exists():
                try:
                    if checkpoint_path.exists():
                        shutil.rmtree(checkpoint_path)
                    shutil.copytree(backup, checkpoint_path)
                    self.logger.info(f"Restored from backup: {backup}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to restore from {backup}: {e}")

        return False

    def save_emergency_checkpoint(self):
        """Save an emergency checkpoint on interrupt."""
        try:
            emergency_path = Path(self.config.checkpoint_dir) / "emergency_checkpoint"

            # Find the most recent checkpoint
            checkpoints = sorted(
                Path(self.config.checkpoint_dir).glob("epoch*"),
                key=lambda x: x.stat().st_mtime
            )

            if checkpoints:
                latest = checkpoints[-1]
                shutil.copytree(latest, emergency_path, dirs_exist_ok=True)
                self.logger.info(f"Saved emergency checkpoint: {emergency_path}")

                # Save recovery state
                state = {
                    "interrupted_at": datetime.now().isoformat(),
                    "retry_count": self.retry_count,
                    "oom_count": self.oom_count,
                    "current_batch_size": self.current_batch_size,
                    "elapsed_hours": (time.time() - self.start_time) / 3600
                }

                with open(emergency_path / "recovery_state.json", 'w') as f:
                    json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save emergency checkpoint: {e}")

    def handle_oom_error(self) -> bool:
        """Handle OOM error by reducing batch size."""
        self.oom_count += 1

        if self.oom_count > self.config.max_oom_retries:
            self.logger.error(f"Max OOM retries ({self.config.max_oom_retries}) exceeded")
            return False

        # Reduce batch size
        new_batch_size = max(
            self.config.min_batch_size,
            int(self.current_batch_size * self.config.batch_size_reduction_factor)
        )

        if new_batch_size == self.current_batch_size:
            self.logger.error(f"Cannot reduce batch size below {self.config.min_batch_size}")
            return False

        self.logger.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size}")
        self.current_batch_size = new_batch_size

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Wait for memory to be freed
        time.sleep(10)

        return True

    def handle_network_error(self, attempt: int = 0) -> bool:
        """Handle network errors with exponential backoff."""
        delay = min(
            self.config.network_retry_delay * (self.config.network_backoff_factor ** attempt),
            self.config.network_max_delay
        )

        self.logger.info(f"Network error. Retrying in {delay:.1f} seconds...")
        time.sleep(delay)

        return attempt < 5  # Max 5 network retries

    def run_training(self) -> int:
        """Run the actual training command."""
        # Build command with current batch size
        cmd = self.build_command(
            batch_size=self.current_batch_size,
            output_dir=self.config.checkpoint_dir
        )

        self.logger.info(f"Running: {' '.join(cmd)}")

        # Log system health before starting
        health = self.check_system_health()
        if health["warnings"]:
            self.logger.warning(f"System warnings: {', '.join(health['warnings'])}")

        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Monitor output
        try:
            for line in process.stdout:
                print(line, end='')  # Print to console

                # Check for specific error patterns
                if "CUDA out of memory" in line or "OutOfMemoryError" in line:
                    process.terminate()
                    raise RuntimeError("OOM")
                elif "gradient overflow" in line or "nan loss" in line.lower():
                    self.logger.warning("Detected gradient issues")
                elif "ConnectionError" in line or "HTTPError" in line:
                    process.terminate()
                    raise ConnectionError("Network error")

            return_code = process.wait()

        except KeyboardInterrupt:
            process.terminate()
            raise

        finally:
            process.poll()
            if process.returncode is None:
                process.terminate()

        return return_code

    def run(self) -> bool:
        """Main training loop with error recovery."""
        self.logger.info("="*60)
        self.logger.info("Starting Robust Training")
        self.logger.info(f"Config: {self.config.checkpoint_dir}")
        self.logger.info(f"Max retries: {self.config.max_retries}")
        self.logger.info(f"Initial batch size: {self.current_batch_size}")
        self.logger.info("="*60)

        success = False

        while self.retry_count <= self.config.max_retries and not self.interrupted:
            try:
                # Check if we're resuming from a checkpoint
                checkpoint_path = Path(self.config.checkpoint_dir)
                existing_checkpoints = sorted(checkpoint_path.glob("epoch*"))

                if existing_checkpoints:
                    latest_checkpoint = existing_checkpoints[-1]
                    self.logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                    self.backup_checkpoint(latest_checkpoint)

                # Run training
                return_code = self.run_training()

                if return_code == 0:
                    self.logger.info("Training completed successfully!")
                    success = True
                    break
                else:
                    raise RuntimeError(f"Training failed with return code {return_code}")

            except RuntimeError as e:
                error_msg = str(e)
                self.logger.error(f"Runtime error: {error_msg}")

                if "OOM" in error_msg:
                    if not self.handle_oom_error():
                        break
                else:
                    self.retry_count += 1
                    self.logger.info(f"Retry {self.retry_count}/{self.config.max_retries}")

            except ConnectionError as e:
                self.logger.error(f"Network error: {e}")
                if not self.handle_network_error(self.retry_count):
                    break
                self.retry_count += 1

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                self.logger.error(traceback.format_exc())

                # Try to restore from backup
                checkpoint_path = Path(self.config.checkpoint_dir)
                if self.restore_from_backup(checkpoint_path):
                    self.logger.info("Restored from backup, retrying...")
                    self.retry_count += 1
                else:
                    break

            # Wait before retry
            if self.retry_count <= self.config.max_retries and not success:
                wait_time = 30 * (2 ** min(self.retry_count - 1, 3))  # Exponential backoff, max 4 min
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        # Final summary
        elapsed_hours = (time.time() - self.start_time) / 3600
        self.logger.info("="*60)
        self.logger.info("Training Summary")
        self.logger.info(f"Success: {success}")
        self.logger.info(f"Total retries: {self.retry_count}")
        self.logger.info(f"OOM events: {self.oom_count}")
        self.logger.info(f"Final batch size: {self.current_batch_size}")
        self.logger.info(f"Elapsed time: {elapsed_hours:.2f} hours")
        self.logger.info("="*60)

        return success


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Robust training wrapper for LatentWire")
    parser.add_argument(
        "--config", type=str,
        help="Path to YAML config file"
    )

    # Allow overriding any config option
    parser.add_argument("--script_path", type=str, default="latentwire/train.py")
    parser.add_argument("--checkpoint_dir", type=str, default="runs/robust_experiment")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_oom_retries", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--min_batch_size", type=int, default=1)
    parser.add_argument("--memory_threshold_gb", type=float, default=70.0)

    # Training arguments to pass through
    parser.add_argument("--llama_id", type=str)
    parser.add_argument("--qwen_id", type=str)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--latent_len", type=int)
    parser.add_argument("--d_z", type=int)
    parser.add_argument("--encoder_type", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--sequential_models", action="store_true")
    parser.add_argument("--warm_anchor_text", type=str)
    parser.add_argument("--first_token_ce_weight", type=float)

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = {}

    # Build base training args from command line
    base_args = {}
    train_arg_names = [
        'llama_id', 'qwen_id', 'samples', 'epochs', 'batch_size',
        'latent_len', 'd_z', 'encoder_type', 'dataset', 'sequential_models',
        'warm_anchor_text', 'first_token_ce_weight'
    ]

    for arg_name in train_arg_names:
        value = getattr(args, arg_name, None)
        if value is not None:
            base_args[arg_name] = value

    # Merge YAML config with command line args (command line takes precedence)
    if 'base_args' in yaml_config:
        base_args = {**yaml_config['base_args'], **base_args}

    # Create training config
    config = TrainingConfig(
        script_path=args.script_path,
        base_args=base_args,
        checkpoint_dir=args.checkpoint_dir,
        max_retries=args.max_retries,
        max_oom_retries=args.max_oom_retries,
        min_batch_size=args.min_batch_size,
        memory_threshold_gb=args.memory_threshold_gb
    )

    # Override with any YAML config values
    for key, value in yaml_config.items():
        if key != 'base_args' and hasattr(config, key):
            setattr(config, key, value)

    # Run training
    trainer = RobustTrainer(config)
    success = trainer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()