#!/usr/bin/env python3
"""
Simplified preemption test script.
Tests signal handling and checkpoint saving without complex dependencies.
"""

import os
import sys
import time
import signal
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


def test_basic_signals():
    """Test basic signal handling functionality."""
    print("=" * 60)
    print("BASIC SIGNAL HANDLING TEST")
    print("=" * 60)

    # Install handlers
    install_signal_handlers()

    print("\nTest 1: Checkpoint interval")
    print("-" * 40)

    # Test interval-based checkpointing
    for i in range(6):
        time.sleep(1)
        if should_save_checkpoint(interval_seconds=3):
            print(f"✅ Step {i}: Checkpoint triggered by interval")
            mark_checkpoint_saved()
        else:
            print(f"   Step {i}: No checkpoint needed")

    print("\nTest completed successfully!")
    print("=" * 60)


def test_checkpoint_saving():
    """Test checkpoint save/load functionality."""
    print("\n" + "=" * 60)
    print("CHECKPOINT SAVE/LOAD TEST")
    print("=" * 60)

    save_dir = Path("runs/test_checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create test state
    state = {
        "epoch": 5,
        "step": 100,
        "loss": 0.123,
        "timestamp": datetime.now().isoformat()
    }

    # Save checkpoint
    checkpoint_file = save_dir / "test_state.json"
    print(f"\nSaving checkpoint to {checkpoint_file}")

    with open(checkpoint_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"✅ Saved state: epoch={state['epoch']}, step={state['step']}, loss={state['loss']:.3f}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    with open(checkpoint_file, 'r') as f:
        loaded_state = json.load(f)

    # Verify
    assert loaded_state["epoch"] == state["epoch"]
    assert loaded_state["step"] == state["step"]
    print(f"✅ Loaded state matches: epoch={loaded_state['epoch']}, step={loaded_state['step']}")

    print("\nTest completed successfully!")
    print("=" * 60)


def test_preemption_simulation():
    """Simulate preemption with mock training."""
    print("\n" + "=" * 60)
    print("PREEMPTION SIMULATION TEST")
    print("=" * 60)
    print("This test will run for 10 seconds")
    print("You can send SIGTERM to test preemption:")
    print(f"  kill -TERM {os.getpid()}")
    print("Or press Ctrl+C to test interrupt")
    print("=" * 60)

    # Install handlers
    install_signal_handlers()

    save_dir = Path("runs/test_preemption_sim")
    save_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    checkpoint_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 10:
            # Simulate training step
            step += 1
            time.sleep(0.5)

            print(f"Step {step:03d}", end="\r")

            # Check for checkpoint
            if should_save_checkpoint(interval_seconds=2):
                checkpoint_count += 1
                print(f"\n💾 Saving checkpoint #{checkpoint_count} at step {step}")

                # Save state
                state_file = save_dir / f"checkpoint_{checkpoint_count}.json"
                with open(state_file, 'w') as f:
                    json.dump({"step": step, "time": time.time()}, f)

                mark_checkpoint_saved()

                # Check if we should exit
                if should_exit_training():
                    print("\n🔄 Exiting due to signal - ready for resume")
                    return

        print(f"\n\n✅ Simulation completed: {step} steps, {checkpoint_count} checkpoints")

    except KeyboardInterrupt:
        print(f"\n\n⛔ Interrupted at step {step}")

    print("=" * 60)


def main():
    """Run all tests."""
    print("\n🧪 PREEMPTION SYSTEM VALIDATION TESTS")
    print("=" * 60)

    # Run tests
    try:
        test_basic_signals()
        test_checkpoint_saving()
        test_preemption_simulation()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())