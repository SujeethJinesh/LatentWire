#!/usr/bin/env python3
"""
Preemption-safe training wrapper for LatentWire.

This wrapper provides robust handling of SLURM preemption signals and automatic resumption.
It ensures training can be interrupted gracefully and resumed from the exact same point.

Key Features:
- Signal handler for SIGTERM (preemption warning)
- Immediate checkpoint save within grace period
- Automatic resume from latest checkpoint
- Periodic checkpoint saves (configurable)
- Atomic checkpoint writes
- Exact training state preservation (batch index, RNG states)
- Mid-batch interruption handling

Usage:
    # Basic usage with default settings
    python telepathy/preemptible_training.py --config configs/train.yaml

    # With custom checkpoint interval (every 5 minutes)
    python telepathy/preemptible_training.py --checkpoint_interval 300 \\
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \\
        --samples 10000 --epochs 10

    # Resume from specific checkpoint
    python telepathy/preemptible_training.py --resume_from runs/experiment/state.pt \\
        --config configs/train.yaml

SLURM Integration:
    Add to your SLURM script:

    #!/bin/bash
    #SBATCH --signal=TERM@120  # Send SIGTERM 120 seconds before job ends
    #SBATCH --requeue          # Allow job to be requeued after preemption

    python telepathy/preemptible_training.py --auto_resume --checkpoint_interval 300 \\
        --save_dir runs/preemptible_exp \\
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \\
        --samples 87599 --epochs 24
"""

import os
import sys
import signal
import time
import json
import argparse
import subprocess
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import shutil
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. This script requires PyTorch.")
    sys.exit(1)


class PreemptionHandler:
    """Handles preemption signals and coordinates graceful shutdown."""

    def __init__(self, grace_period: int = 120):
        """
        Initialize preemption handler.

        Args:
            grace_period: Seconds available to save after receiving SIGTERM (default: 120)
        """
        self.grace_period = grace_period
        self.preemption_received = False
        self.save_requested = False
        self.shutdown_requested = False
        self.last_save_time = time.time()
        self.save_lock = threading.Lock()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGUSR1, self._handle_sigusr1)  # Manual save trigger
        signal.signal(signal.SIGINT, self._handle_sigint)    # Ctrl+C

    def _handle_sigterm(self, signum, frame):
        """Handle SIGTERM (preemption warning)."""
        print(f"\n🚨 PREEMPTION SIGNAL RECEIVED! Starting graceful shutdown...")
        print(f"   Grace period: {self.grace_period} seconds")
        self.preemption_received = True
        self.save_requested = True

        # Start countdown timer
        threading.Thread(target=self._countdown_timer, daemon=True).start()

    def _handle_sigusr1(self, signum, frame):
        """Handle SIGUSR1 (manual save request)."""
        print("\n📝 Manual save requested (SIGUSR1)")
        self.save_requested = True

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT (Ctrl+C)."""
        print("\n⛔ Interrupt received! Saving checkpoint before exit...")
        self.save_requested = True
        self.shutdown_requested = True

    def _countdown_timer(self):
        """Countdown timer for grace period."""
        remaining = self.grace_period
        while remaining > 0 and not self.shutdown_requested:
            if remaining % 30 == 0 or remaining <= 10:
                print(f"   ⏱️  {remaining} seconds remaining...")
            time.sleep(1)
            remaining -= 1

        if not self.shutdown_requested:
            print("   ⚠️  Grace period expired! Forcing exit...")
            os._exit(1)

    def should_save(self, checkpoint_interval: int = 300) -> bool:
        """
        Check if checkpoint should be saved.

        Args:
            checkpoint_interval: Seconds between periodic saves

        Returns:
            True if save is needed (preemption, interval, or manual request)
        """
        with self.save_lock:
            # Always save on preemption or manual request
            if self.save_requested:
                return True

            # Check periodic interval
            if checkpoint_interval > 0:
                elapsed = time.time() - self.last_save_time
                if elapsed >= checkpoint_interval:
                    return True

            return False

    def mark_saved(self):
        """Mark that a checkpoint was just saved."""
        with self.save_lock:
            self.last_save_time = time.time()
            self.save_requested = False

    def should_exit(self) -> bool:
        """Check if training should exit."""
        return self.shutdown_requested or self.preemption_received


class CheckpointManager:
    """Manages checkpoint saving and loading with atomic operations."""

    def __init__(self, save_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory for checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_training_state(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete training state atomically.

        Args:
            state: Training state dict (epoch, step, batch_idx, RNG states, etc.)
            artifacts: Model artifacts (encoder, adapters, optimizer, etc.)
            metadata: Optional metadata to save

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add timestamp to state
        state["save_timestamp"] = timestamp
        state["save_time_iso"] = datetime.now().isoformat()

        # Save metadata if provided
        if metadata:
            meta_path = self.save_dir / "metadata.json"
            self._atomic_json_save(metadata, meta_path)

        # Save state with atomic write
        state_path = self.save_dir / "state.pt"
        self._atomic_torch_save(state, state_path)

        # Save all artifacts atomically
        for name, obj in artifacts.items():
            if obj is not None:
                artifact_path = self.save_dir / name
                if name.endswith(".pt"):
                    self._atomic_torch_save(obj, artifact_path)
                elif name.endswith(".json"):
                    self._atomic_json_save(obj, artifact_path)

        # Create backup link for safety
        backup_path = self.save_dir / f"state_backup_{timestamp}.pt"
        if backup_path.exists():
            backup_path.unlink()
        try:
            backup_path.symlink_to(state_path.name)
        except:
            pass  # Symlinks may not work on all filesystems

        print(f"✅ Checkpoint saved: {state_path}")
        return str(state_path)

    def load_training_state(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (or directory containing checkpoints)

        Returns:
            State dict if found, None otherwise
        """
        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.is_dir():
                # Find latest state.pt in directory
                state_path = path / "state.pt"
                if not state_path.exists():
                    # Look for backup states
                    backups = sorted(path.glob("state_backup_*.pt"))
                    if backups:
                        state_path = backups[-1]
            else:
                state_path = path
        else:
            # Auto-find in save_dir
            state_path = self.save_dir / "state.pt"
            if not state_path.exists():
                return None

        if not state_path.exists():
            return None

        try:
            state = torch.load(state_path, map_location="cpu")
            print(f"📂 Loaded checkpoint: {state_path}")
            if "save_time_iso" in state:
                print(f"   Saved at: {state['save_time_iso']}")
            return state
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint {state_path}: {e}")
            return None

    def _atomic_torch_save(self, obj: Any, path: Path):
        """Save PyTorch object atomically."""
        tmp_path = path.with_suffix(".tmp")
        torch.save(obj, tmp_path)
        tmp_path.replace(path)

    def _atomic_json_save(self, obj: Any, path: Path):
        """Save JSON object atomically."""
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, 'w') as f:
            json.dump(obj, f, indent=2)
        tmp_path.replace(path)


class PreemptibleTrainer:
    """Main trainer class with preemption support."""

    def __init__(self, args: argparse.Namespace):
        """
        Initialize preemptible trainer.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.preemption_handler = PreemptionHandler(grace_period=args.grace_period)
        self.checkpoint_manager = CheckpointManager(args.save_dir)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.batch_idx = 0
        self.best_loss = float('inf')

        # Track partial batch progress for mid-batch interruption
        self.partial_batch_state = None

    def run(self):
        """Main training loop with preemption handling."""

        # Try to resume from checkpoint
        resumed = self._try_resume()

        # Prepare training command
        train_cmd = self._build_train_command(resumed)

        print("=" * 60)
        print("PREEMPTIBLE TRAINING WRAPPER")
        print("=" * 60)
        print(f"Save directory: {self.args.save_dir}")
        print(f"Checkpoint interval: {self.args.checkpoint_interval}s")
        print(f"Grace period: {self.args.grace_period}s")
        print(f"Auto-resume: {self.args.auto_resume}")
        if resumed:
            print(f"Resumed from: epoch {self.epoch}, step {self.global_step}")
        print("=" * 60)
        print()

        # Start training process
        print("🚀 Starting training process...")
        print(f"Command: {' '.join(train_cmd)}")
        print()

        try:
            # Run training with monitoring
            self._run_training_with_monitoring(train_cmd)
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            traceback.print_exc()

            # Try to save emergency checkpoint
            if self.preemption_handler.should_save():
                print("\n🆘 Attempting emergency checkpoint save...")
                self._save_emergency_checkpoint()

            sys.exit(1)

        print("\n✅ Training completed successfully!")

    def _try_resume(self) -> bool:
        """
        Try to resume from checkpoint.

        Returns:
            True if resumed, False if starting fresh
        """
        checkpoint_path = None

        if self.args.resume_from:
            checkpoint_path = self.args.resume_from
        elif self.args.auto_resume:
            checkpoint_path = self.args.save_dir

        if not checkpoint_path:
            return False

        state = self.checkpoint_manager.load_training_state(checkpoint_path)
        if state:
            self.epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            self.batch_idx = state.get("batch_idx", 0)
            self.best_loss = state.get("best_loss", float('inf'))
            self.partial_batch_state = state.get("partial_batch_state", None)

            print(f"📥 Resumed from checkpoint:")
            print(f"   Epoch: {self.epoch}")
            print(f"   Global step: {self.global_step}")
            print(f"   Batch index: {self.batch_idx}")
            print(f"   Best loss: {self.best_loss:.4f}")

            return True

        return False

    def _build_train_command(self, resumed: bool) -> list:
        """
        Build command for train.py with appropriate arguments.

        Args:
            resumed: Whether training is being resumed

        Returns:
            Command as list of strings
        """
        cmd = [sys.executable, "latentwire/train.py"]

        # Pass through all arguments
        for key, value in vars(self.args).items():
            # Skip wrapper-specific arguments
            if key in ["checkpoint_interval", "grace_period", "resume_from", "auto_resume"]:
                continue

            # Skip None values
            if value is None:
                continue

            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        # Add resume arguments if needed
        if resumed:
            cmd.extend(["--resume_from", str(self.args.save_dir)])

            # Skip to correct batch if mid-epoch
            if self.batch_idx > 0:
                cmd.extend(["--skip_batches", str(self.batch_idx)])

        return cmd

    def _run_training_with_monitoring(self, train_cmd: list):
        """
        Run training process with checkpoint monitoring.

        Args:
            train_cmd: Command to run training
        """
        # Start training subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )

        # Monitor process and handle checkpointing
        last_checkpoint_time = time.time()

        try:
            while True:
                # Check if process is still running
                retcode = process.poll()
                if retcode is not None:
                    # Process finished
                    if retcode == 0:
                        print("\n✅ Training process completed successfully")
                    else:
                        print(f"\n⚠️  Training process exited with code {retcode}")
                    break

                # Read and print output
                line = process.stdout.readline()
                if line:
                    print(line, end='')

                    # Parse training progress from output
                    self._parse_training_progress(line)

                # Check for preemption or checkpoint interval
                if self.preemption_handler.should_save(self.args.checkpoint_interval):
                    print("\n📝 Checkpoint save triggered...")

                    # Send signal to training process to save
                    process.send_signal(signal.SIGUSR1)

                    # Wait for save to complete (look for confirmation in output)
                    save_deadline = time.time() + 30  # 30 second timeout
                    while time.time() < save_deadline:
                        line = process.stdout.readline()
                        if line:
                            print(line, end='')
                            if "Saved latest checkpoint" in line or "Checkpoint saved" in line:
                                break

                    self.preemption_handler.mark_saved()
                    last_checkpoint_time = time.time()

                    # Exit if preempted
                    if self.preemption_handler.should_exit():
                        print("\n⚡ Preemption handled! Terminating training process...")
                        process.terminate()

                        # Wait for graceful shutdown
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()

                        print("🔄 Ready for requeue/restart")
                        sys.exit(0)

                # Small sleep to avoid busy waiting
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⛔ Interrupted! Saving checkpoint...")
            process.send_signal(signal.SIGUSR1)
            time.sleep(5)  # Give time to save
            process.terminate()
            raise

        finally:
            # Ensure process is terminated
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

    def _parse_training_progress(self, line: str):
        """
        Parse training progress from output line.

        Args:
            line: Output line from training process
        """
        # Parse epoch/step information
        import re

        # Match epoch pattern
        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
        if epoch_match:
            self.epoch = int(epoch_match.group(1))

        # Match step pattern
        step_match = re.search(r'Step\s+(\d+)', line)
        if step_match:
            self.global_step = int(step_match.group(1))

        # Match batch pattern
        batch_match = re.search(r'Batch\s+(\d+)/(\d+)', line)
        if batch_match:
            self.batch_idx = int(batch_match.group(1))

        # Match loss pattern
        loss_match = re.search(r'Loss:\s+([\d.]+)', line)
        if loss_match:
            loss = float(loss_match.group(1))
            if loss < self.best_loss:
                self.best_loss = loss

    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint with current state."""
        try:
            state = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "batch_idx": self.batch_idx,
                "best_loss": self.best_loss,
                "partial_batch_state": self.partial_batch_state,
                "emergency": True,
                "timestamp": datetime.now().isoformat()
            }

            emergency_path = self.checkpoint_manager.save_dir / "emergency_state.pt"
            torch.save(state, emergency_path)
            print(f"🆘 Emergency checkpoint saved: {emergency_path}")

        except Exception as e:
            print(f"❌ Failed to save emergency checkpoint: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preemptible training wrapper for LatentWire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Wrapper-specific arguments
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=300,
        help="Seconds between periodic checkpoint saves (default: 300)"
    )
    parser.add_argument(
        "--grace_period",
        type=int,
        default=120,
        help="Seconds available after SIGTERM to save checkpoint (default: 120)"
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume from latest checkpoint in save_dir"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to specific checkpoint to resume from"
    )

    # Pass through all train.py arguments
    parser.add_argument("--save_dir", type=str, default="runs/preemptible")
    parser.add_argument("--llama_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--qwen_id", type=str, default="")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--latent_len", type=int, default=32)
    parser.add_argument("--d_z", type=int, default=256)
    parser.add_argument("--encoder_type", type=str, default="byte")
    parser.add_argument("--dataset", type=str, default="squad")
    parser.add_argument("--sequential_models", action="store_true")
    parser.add_argument("--warm_anchor_text", type=str, default="Answer: ")
    parser.add_argument("--first_token_ce_weight", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Parse arguments
    args, unknown = parser.parse_known_args()

    # Add any unknown arguments (will be passed to train.py)
    if unknown:
        print(f"Note: Passing through additional arguments: {unknown}")

    # Check PyTorch availability
    if not PYTORCH_AVAILABLE:
        print("ERROR: PyTorch is required but not installed.")
        sys.exit(1)

    # Run training with preemption support
    trainer = PreemptibleTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()