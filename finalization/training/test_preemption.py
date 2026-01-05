#!/usr/bin/env python3
"""
Test script to demonstrate preemption handling.

This script simulates a training loop and shows how checkpoints are saved
when a SIGTERM signal is received.

Usage:
    # Run the test
    python test_preemption.py

    # In another terminal, send SIGTERM to test preemption:
    kill -TERM <pid>

    # Or use timeout to auto-terminate after 10 seconds:
    timeout --signal=TERM 10 python test_preemption.py
"""

import os
import sys
import signal
import time
import json
import threading
from pathlib import Path
from datetime import datetime

# Simple state tracking
class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.loss = 0.0

state = TrainingState()
preemption_requested = False
checkpoint_lock = threading.Lock()


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal."""
    global preemption_requested
    print("\n" + "="*60)
    print("SIGTERM RECEIVED - Simulating preemption!")
    print("="*60)

    preemption_requested = True

    # Save checkpoint
    save_checkpoint("preemption")

    print("\nCheckpoint saved successfully!", flush=True)
    print("Exiting gracefully...")
    sys.exit(0)


def save_checkpoint(reason="periodic"):
    """Save a checkpoint."""
    with checkpoint_lock:
        checkpoint_dir = Path("test_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().isoformat()

        checkpoint = {
            "timestamp": timestamp,
            "reason": reason,
            "epoch": state.epoch,
            "step": state.step,
            "loss": state.loss,
            "pid": os.getpid()
        }

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_{reason}_{int(time.time())}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"\n[{timestamp}] Checkpoint saved: {checkpoint_path}", flush=True)
        print(f"  Reason: {reason}")
        print(f"  Epoch: {state.epoch}, Step: {state.step}, Loss: {state.loss:.4f}", flush=True)

        return checkpoint_path


def simulate_training():
    """Simulate a training loop."""
    global state, preemption_requested

    print("Starting simulated training...", flush=True)
    print(f"Process ID: {os.getpid()}")
    print("Send SIGTERM to this process to test preemption handling")
    print("e.g., in another terminal: kill -TERM", os.getpid())
    print("\n" + "-"*60 + "\n")

    # Register signal handler
    signal.signal(signal.SIGTERM, handle_sigterm)

    # Training parameters
    num_epochs = 10
    steps_per_epoch = 100
    checkpoint_interval = 5  # seconds
    last_checkpoint_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        state.epoch = epoch

        print(f"Epoch {epoch + 1}/{num_epochs}", flush=True)

        for step in range(steps_per_epoch):
            state.step = step

            # Check for preemption
            if preemption_requested:
                print("\nPreemption requested - stopping training", flush=True)
                return

            # Simulate training step
            time.sleep(0.1)  # Simulate computation
            state.loss = 1.0 / (epoch * steps_per_epoch + step + 1)  # Fake decreasing loss

            # Print progress occasionally
            if step % 20 == 0:
                print(f"  Step {step}/{steps_per_epoch}, Loss: {state.loss:.6f}", flush=True)

            # Periodic checkpoint
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval:
                save_checkpoint("periodic")
                last_checkpoint_time = current_time

        # End of epoch checkpoint
        save_checkpoint(f"epoch_{epoch + 1}")

    print("\nTraining completed!", flush=True)
    save_checkpoint("final")


def cleanup_test_checkpoints():
    """Clean up test checkpoints from previous runs."""
    checkpoint_dir = Path("test_checkpoints")
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"Cleaned up {checkpoint_dir}", flush=True)


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("PREEMPTION HANDLER TEST")
    print("="*60 + "\n")

    # Clean up old checkpoints
    cleanup_test_checkpoints()

    try:
        # Run simulated training
        simulate_training()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received")
        save_checkpoint("interrupt")
    except Exception as e:
        print(f"\nError during training: {e}", flush=True)
        save_checkpoint("error")
        raise

    # Show saved checkpoints
    checkpoint_dir = Path("test_checkpoints")
    if checkpoint_dir.exists():
        print("\n" + "="*60)
        print("SAVED CHECKPOINTS:", flush=True)
        print("="*60)

        for checkpoint_file in sorted(checkpoint_dir.glob("*.json")):
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            print(f"\n{checkpoint_file.name}:", flush=True)
            print(f"  Timestamp: {data['timestamp']}")
            print(f"  Reason: {data['reason']}")
            print(f"  Epoch: {data['epoch']}, Step: {data['step']}", flush=True)
            print(f"  Loss: {data['loss']:.6f}", flush=True)


if __name__ == "__main__":
    main()