#!/usr/bin/env python3
"""
Test script for preemption handling.

This script demonstrates the preemption-safe training wrapper in action.
Run this locally to test signal handling and checkpoint saving.

Usage:
    # Basic test (runs for 60 seconds with 10-second checkpoint interval)
    python telepathy/test_preemption.py

    # Test with custom parameters
    python telepathy/test_preemption.py --duration 120 --interval 5

    # Test preemption signal
    # In another terminal, send: kill -TERM <pid>
    # Or press Ctrl+C to test interrupt handling
"""

import os
import sys
import time
import signal
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telepathy.training_signals import (
    install_signal_handlers,
    should_save_checkpoint,
    mark_checkpoint_saved,
    should_exit_training
)


class MockTrainingState:
    """Mock training state for demonstration."""

    def __init__(self, save_dir: str = "runs/test_preemption"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.step = 0
        self.loss = 10.0
        self.checkpoints_saved = 0

        # Try to load existing state
        self.state_file = self.save_dir / "state.json"
        if self.state_file.exists():
            self.load_state()
            print(f"📂 Resumed from checkpoint: epoch={self.epoch}, step={self.step}")
        else:
            print("🆕 Starting fresh training")

    def save_state(self):
        """Save current state to checkpoint."""
        state = {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "checkpoints_saved": self.checkpoints_saved,
            "timestamp": datetime.now().isoformat()
        }

        # Atomic save
        tmp_file = self.state_file.with_suffix(".tmp")
        with open(tmp_file, 'w') as f:
            json.dump(state, f, indent=2)
        tmp_file.replace(self.state_file)

        self.checkpoints_saved += 1
        print(f"💾 Checkpoint #{self.checkpoints_saved} saved: epoch={self.epoch}, step={self.step}, loss={self.loss:.4f}")

    def load_state(self):
        """Load state from checkpoint."""
        with open(self.state_file, 'r') as f:
            state = json.load(f)

        self.epoch = state["epoch"]
        self.step = state["step"]
        self.loss = state["loss"]
        self.checkpoints_saved = state.get("checkpoints_saved", 0)

    def train_step(self):
        """Simulate one training step."""
        self.step += 1
        if self.step % 10 == 0:
            self.epoch += 1

        # Simulate loss decreasing
        self.loss *= 0.99

        # Simulate some work
        time.sleep(1)

        print(f"Step {self.step:04d} | Epoch {self.epoch:02d} | Loss: {self.loss:.4f}", end="\r")


def run_mock_training(duration: int = 60, checkpoint_interval: int = 10):
    """
    Run mock training with preemption handling.

    Args:
        duration: Total duration in seconds
        checkpoint_interval: Seconds between checkpoints
    """
    print("=" * 60)
    print("PREEMPTION TEST")
    print("=" * 60)
    print(f"Duration: {duration}s")
    print(f"Checkpoint interval: {checkpoint_interval}s")
    print("Send SIGTERM (kill -TERM <pid>) to test preemption")
    print("Press Ctrl+C to test interrupt handling")
    print("=" * 60)
    print()

    # Install signal handlers
    install_signal_handlers()

    # Initialize training state
    state = MockTrainingState()

    # Training loop
    start_time = time.time()
    last_checkpoint = time.time()

    try:
        while time.time() - start_time < duration:
            # Perform training step
            state.train_step()

            # Check if we should save checkpoint
            if should_save_checkpoint(checkpoint_interval):
                print()  # New line for checkpoint message
                state.save_state()
                mark_checkpoint_saved()
                last_checkpoint = time.time()

                # Check if we should exit
                if should_exit_training():
                    print("\n🔄 Exiting due to signal - ready for resume")
                    return 99  # Preemption exit code

        print(f"\n\n✅ Training completed successfully!")
        print(f"Final state: epoch={state.epoch}, step={state.step}, loss={state.loss:.4f}")
        print(f"Total checkpoints saved: {state.checkpoints_saved}")
        return 0

    except KeyboardInterrupt:
        print("\n\n⛔ Training interrupted by user")
        state.save_state()
        return 1

    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test preemption handling system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Total duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Checkpoint interval in seconds (default: 10)"
    )

    args = parser.parse_args()

    # Run mock training
    exit_code = run_mock_training(args.duration, args.interval)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
