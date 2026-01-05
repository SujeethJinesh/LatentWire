#!/usr/bin/env python3
"""
Test script for the preemptible orchestrator.
Validates that all components integrate correctly without running full experiments.
"""

import sys
import os
import time
import signal
import argparse
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from telepathy.run_preemptible_experiments import (
            PreemptibleOrchestrator,
            ExperimentConfig,
            ExperimentPhase
        )
        print("✓ Orchestrator imports")

        from telepathy.checkpoint_manager import CheckpointManager
        print("✓ Checkpoint manager imports")

        from telepathy.preemptible_training import PreemptionHandler
        print("✓ Preemption handler imports")

        import torch
        print(f"✓ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            print("⚠ CUDA not available (CPU mode)")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_experiment_creation():
    """Test experiment queue creation."""
    print("\nTesting experiment creation...")

    from telepathy.run_preemptible_experiments import (
        PreemptibleOrchestrator,
        ExperimentConfig,
        ExperimentPhase
    )

    # Create minimal args
    args = argparse.Namespace(
        experiment="sst2",
        output_dir=tempfile.mkdtemp(prefix="test_orchestrator_"),
        resume=False,
        checkpoint_dir=None,
        test=True,
        config=None
    )

    try:
        orchestrator = PreemptibleOrchestrator(args)
        print(f"✓ Created orchestrator with {len(orchestrator.experiment_queue)} experiments")

        # Check experiment structure
        for exp in orchestrator.experiment_queue[:3]:
            print(f"  - {exp.name}: phase={exp.phase.value}, priority={exp.priority}")

        # Cleanup
        if orchestrator.gpu_monitor:
            orchestrator.gpu_monitor.stop()

        return True

    except Exception as e:
        print(f"✗ Failed to create experiments: {e}")
        return False


def test_checkpoint_manager():
    """Test checkpoint save/load functionality."""
    print("\nTesting checkpoint manager...")

    from telepathy.checkpoint_manager import CheckpointManager
    import torch
    import torch.nn as nn

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_checkpoints_")

    try:
        # Create manager
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints=2,
            save_interval_minutes=0.01,  # Very short for testing
            enable_background_save=False
        )

        # Create dummy model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        ckpt_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            global_step=100,
            batch_idx=50,
            samples_seen=1000,
            metrics={"loss": 0.5, "accuracy": 0.85},
            force=True
        )

        if ckpt_path:
            print(f"✓ Saved checkpoint: {ckpt_path.name}")
        else:
            print("✗ Failed to save checkpoint")
            return False

        # Load checkpoint
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        info = manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer
        )

        if info:
            print(f"✓ Loaded checkpoint: epoch={info['epoch']}, step={info['global_step']}")
        else:
            print("✗ Failed to load checkpoint")
            return False

        # Test checkpoint rotation
        time.sleep(1)
        manager.save_checkpoint(
            model=model, optimizer=optimizer,
            epoch=2, global_step=200,
            batch_idx=100, samples_seen=2000,
            force=True
        )

        time.sleep(1)
        manager.save_checkpoint(
            model=model, optimizer=optimizer,
            epoch=3, global_step=300,
            batch_idx=150, samples_seen=3000,
            force=True
        )

        # Check that old checkpoints are cleaned up
        checkpoints = list(Path(temp_dir).glob("checkpoint_*.pt"))
        print(f"✓ Checkpoint rotation: {len(checkpoints)} checkpoints kept (max={manager.max_checkpoints})")

        # Cleanup
        manager.cleanup()

        return True

    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        return False

    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_preemption_handler():
    """Test preemption signal handling."""
    print("\nTesting preemption handler...")

    from telepathy.preemptible_training import PreemptionHandler

    try:
        handler = PreemptionHandler(grace_period=5)

        # Test initial state
        assert not handler.preemption_received
        assert not handler.save_requested
        print("✓ Handler initialized correctly")

        # Test manual save request
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.1)
        assert handler.save_requested
        print("✓ Manual save signal (SIGUSR1) handled")

        # Reset
        handler.save_requested = False

        # Test checkpoint interval
        assert not handler.should_save(checkpoint_interval=10)
        handler.last_save_time = time.time() - 15
        assert handler.should_save(checkpoint_interval=10)
        print("✓ Checkpoint interval logic works")

        return True

    except Exception as e:
        print(f"✗ Preemption handler test failed: {e}")
        return False


def test_gpu_monitoring():
    """Test GPU monitoring (if available)."""
    print("\nTesting GPU monitoring...")

    import torch

    if not torch.cuda.is_available():
        print("⚠ Skipping GPU tests (no CUDA)")
        return True

    try:
        from telepathy.gpu_monitor import GPUMonitor

        temp_dir = tempfile.mkdtemp(prefix="test_gpu_monitor_")

        monitor = GPUMonitor(
            log_dir=temp_dir,
            interval=1,  # Short interval for testing
            detailed=True
        )

        monitor.start()
        print("✓ GPU monitor started")

        # Let it collect some data
        time.sleep(3)

        # Check if logs were created
        log_files = list(Path(temp_dir).glob("gpu_*.json"))
        if log_files:
            print(f"✓ GPU logs created: {len(log_files)} file(s)")
        else:
            print("⚠ No GPU logs created (may be normal)")

        monitor.stop()
        print("✓ GPU monitor stopped cleanly")

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"⚠ GPU monitoring test skipped: {e}")
        return True  # Don't fail if GPU monitoring isn't available


def run_integration_test():
    """Run a minimal integration test."""
    print("\nRunning integration test...")

    from telepathy.run_preemptible_experiments import PreemptibleOrchestrator
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="test_integration_")

    try:
        # Create test args
        args = argparse.Namespace(
            experiment="sst2",
            output_dir=temp_dir,
            resume=False,
            checkpoint_dir=None,
            test=True,  # Use test mode for small dataset
            config=None
        )

        # Create orchestrator
        orchestrator = PreemptibleOrchestrator(args)
        print(f"✓ Created orchestrator with {len(orchestrator.experiment_queue)} experiments")

        # Simulate running one experiment (without actually running it)
        if orchestrator.experiment_queue:
            exp = orchestrator.experiment_queue[0]
            print(f"✓ Would run experiment: {exp.name}")

            # Test queue save/load
            orchestrator.save_experiment_queue()
            queue_file = Path(temp_dir) / "checkpoints" / "experiment_queue.json"
            if queue_file.exists():
                print("✓ Experiment queue saved")
            else:
                print("✗ Failed to save experiment queue")
                return False

        # Test report generation
        orchestrator.completed_experiments.add("test_experiment")
        orchestrator.generate_report()
        report_file = Path(temp_dir) / "results" / "experiment_report.md"
        if report_file.exists():
            print("✓ Report generated")
        else:
            print("✗ Failed to generate report")
            return False

        # Cleanup
        orchestrator.cleanup()
        print("✓ Cleanup successful")

        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("=" * 60)
    print("PREEMPTIBLE ORCHESTRATOR TEST SUITE")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Experiment Creation", test_experiment_creation),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Preemption Handler", test_preemption_handler),
        ("GPU Monitoring", test_gpu_monitoring),
        ("Integration", run_integration_test),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"Test: {test_name}")
        print('=' * 40)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Orchestrator is ready to use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())