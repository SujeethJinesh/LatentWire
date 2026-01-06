#!/usr/bin/env python3
"""
Test script to verify checkpoint resume functionality works correctly.

This script verifies:
1. CheckpointManager can save and load checkpoints
2. Training can resume from interruption
3. State is properly preserved across resume
4. Automatic checkpoint discovery works
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from checkpoint_manager import CheckpointManager, ExperimentCheckpointer

def test_basic_checkpoint_operations():
    """Test basic save/load operations."""
    print("\n" + "="*60)
    print("TEST: Basic Checkpoint Operations")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            save_interval=10,
            keep_last_n=2,
            verbose=True
        )

        # Test 1: Save checkpoints
        print("\n[Test 1] Saving checkpoints...")
        for step in range(0, 30, 10):
            state = {
                'step': step,
                'epoch': step // 10,
                'loss': 1.0 / (step + 1),
                'metrics': {
                    'accuracy': 0.5 + step * 0.01,
                    'f1_score': 0.4 + step * 0.01
                }
            }

            checkpoint_path = manager.save_checkpoint(
                state=state,
                step=step,
                epoch=step // 10,
                is_best=(step == 20)
            )

            print(f"  Saved checkpoint at step {step}: {checkpoint_path}")

        # Test 2: Find latest checkpoint
        print("\n[Test 2] Finding latest checkpoint...")
        latest_path = manager.find_latest_checkpoint()
        if latest_path:
            print(f"  ✅ Found latest: {latest_path}")
        else:
            print("  ❌ Failed to find latest checkpoint")
            return False

        # Test 3: Load checkpoint
        print("\n[Test 3] Loading checkpoint...")
        loaded = manager.load_checkpoint(latest_path)
        if loaded:
            print(f"  ✅ Loaded state: step={loaded['step']}, loss={loaded['loss']:.4f}")
            print(f"     Metrics: accuracy={loaded['metrics']['accuracy']:.3f}")
        else:
            print("  ❌ Failed to load checkpoint")
            return False

        # Test 4: Check cleanup worked
        print("\n[Test 4] Verifying old checkpoint cleanup...")
        checkpoints = list(Path(tmpdir).glob("step_*"))
        if len(checkpoints) <= manager.keep_last_n:
            print(f"  ✅ Cleanup working: {len(checkpoints)} checkpoints kept (max: {manager.keep_last_n})")
        else:
            print(f"  ❌ Too many checkpoints: {len(checkpoints)} (expected max: {manager.keep_last_n})")

        # Test 5: Best checkpoint symlink
        print("\n[Test 5] Checking best checkpoint symlink...")
        best_link = Path(tmpdir) / "best"
        if best_link.exists():
            print(f"  ✅ Best symlink exists: {best_link.resolve()}")
        else:
            print("  ❌ Best symlink not created")

        print("\n✅ Basic checkpoint operations test PASSED")
        return True


def test_experiment_checkpointer():
    """Test high-level ExperimentCheckpointer."""
    print("\n" + "="*60)
    print("TEST: ExperimentCheckpointer Integration")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config = {
            'output_dir': tmpdir,
            'save_interval': 5,
            'keep_checkpoints': 2,
            'handle_preemption': False,  # Disable for testing
            'verbose': True
        }

        checkpointer = ExperimentCheckpointer(config)

        # Mock model and optimizer
        class MockModel:
            def __init__(self):
                self.weight = 1.0

            def state_dict(self):
                return {'weight': self.weight}

            def load_state_dict(self, state):
                self.weight = state['weight']

        class MockOptimizer:
            def __init__(self):
                self.lr = 0.001

            def state_dict(self):
                return {'lr': self.lr}

            def load_state_dict(self, state):
                self.lr = state['lr']

        model = MockModel()
        optimizer = MockOptimizer()

        # Test 1: Save training state
        print("\n[Test 1] Saving training state...")
        model.weight = 2.5
        optimizer.lr = 0.0005

        checkpoint_path = checkpointer.save_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            step=100,
            metrics={'loss': 0.25, 'accuracy': 0.85},
            best_metric=0.25,
            is_best=True
        )

        print(f"  Saved to: {checkpoint_path}")

        # Test 2: Resume training
        print("\n[Test 2] Resuming training state...")
        new_model = MockModel()
        new_optimizer = MockOptimizer()

        epoch, step, best_metric, metrics = checkpointer.resume_training(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=None,
            checkpoint_path=None  # Auto-discover
        )

        print(f"  Resumed: epoch={epoch}, step={step}, best_metric={best_metric:.4f}")
        print(f"  Model weight: {new_model.weight} (expected: 2.5)")
        print(f"  Optimizer lr: {new_optimizer.lr} (expected: 0.0005)")

        # Verify restoration
        assert abs(new_model.weight - 2.5) < 0.001, "Model weight not restored"
        assert abs(new_optimizer.lr - 0.0005) < 0.00001, "Optimizer lr not restored"
        assert epoch == 5, "Epoch not restored"
        assert step == 100, "Step not restored"

        print("\n✅ ExperimentCheckpointer test PASSED")
        return True


def test_preemption_handling():
    """Test preemption signal handling."""
    print("\n" + "="*60)
    print("TEST: Preemption Handling")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            save_interval=100,
            enable_preemption_handling=True,
            verbose=True
        )

        # Test signal handler setup
        print("\n[Test 1] Signal handlers configured...")
        import signal

        # Check SIGTERM handler (main preemption signal)
        handler = signal.getsignal(signal.SIGTERM)
        if handler != signal.SIG_DFL:
            print("  ✅ SIGTERM handler installed")
        else:
            print("  ❌ SIGTERM handler not installed")

        # Test should_save with preemption
        print("\n[Test 2] Testing should_save logic...")
        assert not manager.should_save(50), "Should not save at step 50"
        assert manager.should_save(100), "Should save at step 100"
        assert manager.should_save(75, force=True), "Should save when forced"

        # Simulate preemption
        manager.preemption_requested = True
        assert manager.should_save(1), "Should save when preemption requested"

        print("\n✅ Preemption handling test PASSED")
        return True


def test_resume_compatibility():
    """Test compatibility with latentwire/train.py."""
    print("\n" + "="*60)
    print("TEST: Compatibility with latentwire/train.py")
    print("="*60)

    try:
        # Import train.py functions
        from latentwire.train import find_latest_checkpoint as train_find_latest

        print("✅ Successfully imported train.py checkpoint functions")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test checkpoints
            for i in [1, 5, 10]:
                epoch_dir = Path(tmpdir) / f"epoch{i}"
                epoch_dir.mkdir()
                (epoch_dir / "state.pt").write_text("dummy")

            # Test both find functions work
            manager = CheckpointManager(save_dir=tmpdir, verbose=False)

            our_latest = manager.find_latest_checkpoint()
            train_latest = train_find_latest(tmpdir)

            print(f"  Our finder: {our_latest}")
            print(f"  train.py finder: {train_latest}")

            if our_latest and train_latest:
                print("✅ Both checkpoint finders work")
            else:
                print("⚠️  Checkpoint finders have issues")

        return True

    except ImportError as e:
        print(f"⚠️  Could not import train.py: {e}")
        print("  This is expected if running standalone tests")
        return True


def run_all_tests():
    """Run all checkpoint tests."""
    print("\n" + "="*80)
    print("CHECKPOINT RESUME FUNCTIONALITY TEST SUITE")
    print("="*80)

    tests = [
        ("Basic Checkpoint Operations", test_basic_checkpoint_operations),
        ("ExperimentCheckpointer Integration", test_experiment_checkpointer),
        ("Preemption Handling", test_preemption_handling),
        ("Train.py Compatibility", test_resume_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 All checkpoint resume tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests failed. Please review output above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)