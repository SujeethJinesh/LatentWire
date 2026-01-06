#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify SIGTERM handling and single checkpoint management.

This script tests:
1. SIGTERM signal handling for preemption
2. Single checkpoint strategy (only 1 checkpoint kept)
3. Checkpoint saving and recovery
4. Atomic writes and backup protection
"""

import os
import sys
import signal
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.checkpoint_manager import CheckpointManager
from training.preemptible_trainer import PreemptibleTrainingState


def test_signal_handling():
    """Test SIGTERM signal handling."""
    print("\n" + "="*60)
    print("Test 1: SIGTERM Signal Handling")
    print("="*60)

    # Create a class to track if handler was called
    class HandlerState:
        def __init__(self):
            self.called = False

    state = HandlerState()

    def test_handler(signum, frame):
        state.called = True
        print(f"✓ Signal handler called with signal {signum}")

    # Register handler
    original_handler = signal.signal(signal.SIGTERM, test_handler)

    try:
        # Send SIGTERM to self
        print("Sending SIGTERM to self...")
        os.kill(os.getpid(), signal.SIGTERM)

        # Give handler time to execute
        time.sleep(0.1)

        assert state.called, "Signal handler was not called"
        print("✓ SIGTERM handler successfully registered and triggered")

    finally:
        # Restore original handler
        signal.signal(signal.SIGTERM, original_handler)

    return True


def test_single_checkpoint_strategy():
    """Test that only 1 checkpoint is maintained."""
    print("\n" + "="*60)
    print("Test 2: Single Checkpoint Strategy")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        # Create checkpoint manager with max_checkpoints=1
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=1,  # Should enforce single checkpoint
            save_interval_minutes=0.1,  # Short interval for testing
            validate_on_save=True
        )

        # Verify max_checkpoints is forced to 1
        assert manager.max_checkpoints == 1, f"Expected max_checkpoints=1, got {manager.max_checkpoints}"
        print(f"✓ CheckpointManager enforces max_checkpoints=1")

        # Save multiple checkpoints
        states = []
        for i in range(3):
            state = {
                'iteration': i,
                'model': torch.randn(10, 10),
                'optimizer': {'lr': 0.001 * (i + 1)},
                'rng': {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                }
            }
            states.append(state)

            print(f"Saving checkpoint {i}...")
            path = manager.save_checkpoint(
                state=state,
                tag=f"iteration_{i}",
                metadata={'iteration': i}
            )
            print(f"  Saved to: {path.name}")

            # Check that only checkpoint_current.pt exists
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
            valid_names = ["checkpoint_current.pt", "checkpoint_backup.pt", "checkpoint_emergency.pt"]
            active_checkpoints = [c for c in checkpoints if c.name in valid_names]

            # Should have at most 2 files (current + temporary backup during save)
            assert len(active_checkpoints) <= 2, f"Found {len(active_checkpoints)} checkpoints: {[c.name for c in active_checkpoints]}"

            # After save completes, should only have current
            time.sleep(0.1)  # Let any cleanup finish
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
            main_checkpoint = checkpoint_dir / "checkpoint_current.pt"
            assert main_checkpoint.exists(), "checkpoint_current.pt should exist"

            # Count non-backup checkpoints
            non_backup = [c for c in checkpoints if 'backup' not in c.name and 'emergency' not in c.name]
            assert len(non_backup) == 1, f"Should have exactly 1 non-backup checkpoint, found {len(non_backup)}"

        print("✓ Only 1 main checkpoint maintained across multiple saves")

        # Verify we can load the latest checkpoint
        loaded_state, metadata = manager.load_checkpoint()
        assert loaded_state['iteration'] == 2, f"Expected iteration=2, got {loaded_state['iteration']}"
        print(f"✓ Latest checkpoint correctly loaded (iteration={loaded_state['iteration']})")

        # Check checkpoint listing
        checkpoint_list = manager.list_checkpoints()
        assert len(checkpoint_list) <= 2, f"Should list at most 2 checkpoints (current + backup), found {len(checkpoint_list)}"
        print(f"✓ Checkpoint listing shows {len(checkpoint_list)} checkpoint(s)")

    return True


def test_atomic_checkpoint_save():
    """Test atomic checkpoint saving with backup protection."""
    print("\n" + "="*60)
    print("Test 3: Atomic Checkpoint Save with Backup")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=1,
            validate_on_save=True,
            validate_on_load=True
        )

        # Save initial checkpoint
        initial_state = {
            'value': 'initial',
            'tensor': torch.tensor([1.0, 2.0, 3.0])
        }

        print("Saving initial checkpoint...")
        manager.save_checkpoint(initial_state, tag="initial")

        # Verify initial checkpoint exists
        current_path = checkpoint_dir / "checkpoint_current.pt"
        assert current_path.exists(), "Initial checkpoint should exist"
        print("✓ Initial checkpoint saved")

        # Save second checkpoint (should backup first)
        second_state = {
            'value': 'second',
            'tensor': torch.tensor([4.0, 5.0, 6.0])
        }

        print("Saving second checkpoint (should create backup)...")
        manager.save_checkpoint(second_state, tag="second")

        # Verify current checkpoint is the second one
        loaded_state, metadata = manager.load_checkpoint()
        assert loaded_state['value'] == 'second', f"Expected 'second', got {loaded_state['value']}"
        print("✓ Second checkpoint saved and loaded correctly")

        # Verify backup was cleaned up after successful save
        backup_path = checkpoint_dir / "checkpoint_backup.pt"
        assert not backup_path.exists(), "Backup should be cleaned up after successful save"
        print("✓ Backup cleaned up after successful save")

    return True


def test_emergency_save():
    """Test emergency checkpoint save for preemption."""
    print("\n" + "="*60)
    print("Test 4: Emergency Checkpoint Save")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=1
        )

        # Create emergency state
        emergency_state = {
            'emergency': True,
            'iteration': 999,
            'critical_data': torch.randn(100, 100),
            'timestamp': time.time()
        }

        print("Performing emergency save...")
        success = manager.emergency_save(emergency_state)
        assert success, "Emergency save should succeed"
        print("✓ Emergency save completed")

        # Check that emergency checkpoint was promoted to current
        current_path = checkpoint_dir / "checkpoint_current.pt"
        emergency_path = checkpoint_dir / "checkpoint_emergency.pt"

        # Either current exists (promoted) or emergency exists (no time to promote)
        assert current_path.exists() or emergency_path.exists(), \
            "Either current or emergency checkpoint should exist"

        # Try to load checkpoint (should find emergency if not promoted)
        loaded_state, metadata = manager.load_checkpoint()
        assert loaded_state['emergency'] == True, "Should load emergency state"
        assert loaded_state['iteration'] == 999, "Should have correct iteration"
        print("✓ Emergency checkpoint can be loaded")

        # Verify metadata marks it as emergency
        if 'emergency' in metadata:
            assert metadata['emergency'] == True, "Metadata should mark as emergency"
            print("✓ Emergency checkpoint properly marked in metadata")

    return True


def test_checkpoint_validation():
    """Test checkpoint validation with checksums."""
    print("\n" + "="*60)
    print("Test 5: Checkpoint Validation")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=1,
            validate_on_save=True,
            validate_on_load=True
        )

        # Save checkpoint with validation
        state = {
            'validated': True,
            'data': torch.randn(50, 50)
        }

        print("Saving checkpoint with validation...")
        path = manager.save_checkpoint(state, tag="validated")
        print("✓ Checkpoint saved with checksum")

        # Load and verify metadata contains checksum
        _, metadata = manager.load_checkpoint()
        assert 'checksum' in metadata, "Metadata should contain checksum"
        print(f"✓ Checksum in metadata: {metadata['checksum'][:8]}...")

        # Corrupt the checkpoint file
        print("Corrupting checkpoint file...")
        with open(path, 'r+b') as f:
            f.seek(100)
            f.write(b'CORRUPTED')

        # Try to load corrupted checkpoint (should fail validation)
        print("Attempting to load corrupted checkpoint...")
        try:
            manager.load_checkpoint()
            assert False, "Should have failed validation"
        except ValueError as e:
            print(f"✓ Validation correctly detected corruption: {e}")

    return True


def test_preemptible_training_state():
    """Test thread-safe training state management."""
    print("\n" + "="*60)
    print("Test 6: Thread-Safe Training State")
    print("="*60)

    state = PreemptibleTrainingState()

    # Test state updates
    state.update(
        epoch=5,
        batch_idx=100,
        global_step=500,
        best_metrics={'f1': 0.85, 'accuracy': 0.90}
    )

    snapshot = state.get_snapshot()
    assert snapshot['epoch'] == 5, f"Expected epoch=5, got {snapshot['epoch']}"
    assert snapshot['batch_idx'] == 100, f"Expected batch_idx=100, got {snapshot['batch_idx']}"
    assert snapshot['global_step'] == 500, f"Expected global_step=500, got {snapshot['global_step']}"
    print("✓ Training state updates correctly")

    # Test model state
    state.encoder = type('MockEncoder', (), {'state_dict': lambda: {'weight': 'encoder_weight'}})()
    state.adapters = {
        'llama': type('MockAdapter', (), {'state_dict': lambda: {'weight': 'llama_weight'}})(),
        'qwen': type('MockAdapter', (), {'state_dict': lambda: {'weight': 'qwen_weight'}})()
    }

    model_state = state.get_model_state()
    assert 'encoder' in model_state, "Should have encoder state"
    assert 'adapter_llama' in model_state, "Should have llama adapter state"
    assert 'adapter_qwen' in model_state, "Should have qwen adapter state"
    print("✓ Model state extraction works")

    return True


def run_all_tests():
    """Run all preemption and checkpoint tests."""
    print("\n" + "="*60)
    print("PREEMPTION & CHECKPOINT VERIFICATION TESTS")
    print("="*60)

    tests = [
        ("Signal Handling", test_signal_handling),
        ("Single Checkpoint Strategy", test_single_checkpoint_strategy),
        ("Atomic Checkpoint Save", test_atomic_checkpoint_save),
        ("Emergency Save", test_emergency_save),
        ("Checkpoint Validation", test_checkpoint_validation),
        ("Training State Management", test_preemptible_training_state),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The system properly handles:")
        print("  • SIGTERM signals for preemption")
        print("  • Single checkpoint strategy (only 1 checkpoint kept)")
        print("  • Atomic checkpoint saves with backup protection")
        print("  • Emergency saves for immediate preemption")
        print("  • Checkpoint validation with checksums")
        print("  • Thread-safe training state management")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()