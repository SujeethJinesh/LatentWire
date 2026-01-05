#!/usr/bin/env python3
"""
Comprehensive validation tests for preemptible training system.

Tests checkpoint management, signal handling, resume functionality,
and multi-GPU scenarios to ensure robust handling of preemption.

Usage:
    python telepathy/test_preemption.py                  # Run all tests
    python telepathy/test_preemption.py --quick          # Quick tests only (<1 minute)
    python telepathy/test_preemption.py --test checkpoint # Run specific test
"""

import os
import sys
import signal
import time
import json
import shutil
import tempfile
import unittest
import argparse
import threading
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Some tests will be skipped.")

from latentwire.checkpointing import (
    save_latest_checkpoint,
    prune_save_dir,
    _is_step_dir,
    _is_tmp_file,
    _atomic_save_torch,
    _atomic_save_json,
)


class MockModel(nn.Module):
    """Simple mock model for testing."""
    def __init__(self, d_in=10, d_out=5):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint manager functionality."""

    def setUp(self):
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp(prefix="test_ckpt_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def test_atomic_save(self):
        """Test atomic file saving."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Test torch save
        model = MockModel()
        path = os.path.join(self.test_dir, "model.pt")
        _atomic_save_torch(model.state_dict(), path)
        self.assertTrue(os.path.exists(path))

        # Verify no temp files left
        files = os.listdir(self.test_dir)
        temp_files = [f for f in files if f.endswith('.tmp') or f.startswith('.')]
        self.assertEqual(len(temp_files), 0, f"Found temp files: {temp_files}")

        # Test JSON save
        config = {"epoch": 1, "loss": 0.5}
        json_path = os.path.join(self.test_dir, "config.json")
        _atomic_save_json(config, json_path)
        self.assertTrue(os.path.exists(json_path))

        with open(json_path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded, config)

    def test_save_latest_checkpoint(self):
        """Test saving latest checkpoint with pruning."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Create old files that should be pruned
        old_files = [
            "step_100/model.pt",
            "model_old.pt",
            "temp.tmp",
            "backup.bak",
            "encoder_step500.pt",
        ]
        for fname in old_files:
            fpath = os.path.join(self.test_dir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, 'w') as f:
                f.write("old data")

        # Save new checkpoint
        model = MockModel()
        optimizer = optim.Adam(model.parameters())

        artifacts = {
            "encoder.pt": model.state_dict(),
            "optimizer.pt": optimizer.state_dict(),
            "config.json": {"epoch": 5, "step": 1000},
            "state.pt": {"epoch": 5, "global_step": 1000},
        }

        freed_pre, freed_post = save_latest_checkpoint(
            self.test_dir,
            artifacts,
            pre_prune=True,
            post_prune=True,
            verbose=False
        )

        # Check that only canonical files exist
        remaining = os.listdir(self.test_dir)
        for fname in remaining:
            self.assertIn(fname, ["encoder.pt", "optimizer.pt", "config.json", "state.pt"])

        # Verify old files were removed
        for fname in old_files:
            fpath = os.path.join(self.test_dir, fname)
            self.assertFalse(os.path.exists(fpath), f"Old file still exists: {fname}")

        # Verify freed bytes > 0
        self.assertGreater(freed_pre, 0, "Should have freed bytes during pre-prune")

    def test_directory_pruning(self):
        """Test that step directories are properly pruned."""
        # Create various directories
        dirs_to_create = [
            "step_100",
            "epoch_5",
            "ckpt_old",
            "global_step_500",
            "valid_dir",  # Should not be pruned
        ]

        for dirname in dirs_to_create:
            dirpath = os.path.join(self.test_dir, dirname)
            os.makedirs(dirpath, exist_ok=True)
            # Add a file to each directory
            with open(os.path.join(dirpath, "dummy.txt"), 'w') as f:
                f.write("test")

        # Prune directory
        freed = prune_save_dir(self.test_dir, keep_only=["valid_file.txt"])

        # Check that step directories were removed
        remaining = os.listdir(self.test_dir)
        for dirname in dirs_to_create:
            if _is_step_dir(dirname):
                self.assertNotIn(dirname, remaining, f"Step dir not removed: {dirname}")
            elif dirname == "valid_dir":
                # This should be removed because it's not in keep_only
                self.assertNotIn(dirname, remaining)

        self.assertGreater(freed, 0, "Should have freed bytes")


class TestSignalHandling(unittest.TestCase):
    """Test signal handling for preemption."""

    def setUp(self):
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp(prefix="test_signal_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def test_sigterm_handler(self):
        """Test SIGTERM handling during training."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        checkpoint_saved = False
        save_path = os.path.join(self.test_dir, "emergency.pt")

        def signal_handler(signum, frame):
            """Emergency checkpoint on SIGTERM."""
            nonlocal checkpoint_saved
            model = MockModel()
            torch.save(model.state_dict(), save_path)
            checkpoint_saved = True
            sys.exit(0)

        # Register handler
        old_handler = signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Simulate SIGTERM
            os.kill(os.getpid(), signal.SIGTERM)
        except SystemExit:
            pass
        finally:
            signal.signal(signal.SIGTERM, old_handler)

        self.assertTrue(checkpoint_saved, "Checkpoint not saved on SIGTERM")
        self.assertTrue(os.path.exists(save_path), "Checkpoint file not created")

    def test_graceful_shutdown(self):
        """Test graceful shutdown with checkpoint."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        class TrainingContext:
            def __init__(self, save_dir):
                self.save_dir = save_dir
                self.shutdown_requested = False
                self.checkpoint_saved = False
                self.model = MockModel()
                self.epoch = 0
                self.step = 0

            def request_shutdown(self, signum, frame):
                """Mark shutdown requested."""
                self.shutdown_requested = True

            def save_checkpoint(self):
                """Save checkpoint."""
                artifacts = {
                    "model.pt": self.model.state_dict(),
                    "state.pt": {"epoch": self.epoch, "step": self.step},
                }
                save_latest_checkpoint(self.save_dir, artifacts, verbose=False)
                self.checkpoint_saved = True

            def training_loop(self):
                """Simulated training loop."""
                for epoch in range(10):
                    self.epoch = epoch
                    for step in range(100):
                        self.step = step

                        # Check for shutdown
                        if self.shutdown_requested:
                            print(f"Shutdown requested at epoch {epoch}, step {step}")
                            self.save_checkpoint()
                            return

                        # Simulate training
                        time.sleep(0.001)

        ctx = TrainingContext(self.test_dir)

        # Register signal handler
        old_handler = signal.signal(signal.SIGUSR1, ctx.request_shutdown)

        try:
            # Start training in thread
            train_thread = threading.Thread(target=ctx.training_loop)
            train_thread.start()

            # Wait a bit then send signal
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGUSR1)

            # Wait for training to finish
            train_thread.join(timeout=1.0)

            self.assertTrue(ctx.shutdown_requested, "Shutdown not requested")
            self.assertTrue(ctx.checkpoint_saved, "Checkpoint not saved")
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "model.pt")))
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "state.pt")))

        finally:
            signal.signal(signal.SIGUSR1, old_handler)


class TestResumeFromCheckpoint(unittest.TestCase):
    """Test resume from checkpoint functionality."""

    def setUp(self):
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp(prefix="test_resume_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def create_checkpoint(self, epoch: int, step: int) -> str:
        """Create a checkpoint and return path."""
        if not TORCH_AVAILABLE:
            return ""

        ckpt_dir = os.path.join(self.test_dir, f"ckpt_epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)

        model = MockModel()
        optimizer = optim.Adam(model.parameters())

        artifacts = {
            "encoder.pt": model.state_dict(),
            "optimizer.pt": optimizer.state_dict(),
            "state.pt": {
                "epoch": epoch,
                "global_step": step,
                "best_loss": 0.1 * (10 - epoch),  # Improving loss
            },
            "config.json": {
                "d_z": 256,
                "latent_len": 32,
            },
        }

        for name, data in artifacts.items():
            path = os.path.join(ckpt_dir, name)
            if name.endswith(".pt"):
                torch.save(data, path)
            else:
                with open(path, 'w') as f:
                    json.dump(data, f)

        return ckpt_dir

    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint in directory."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Create multiple checkpoints
        ckpt1 = self.create_checkpoint(1, 100)
        time.sleep(0.01)  # Ensure different timestamps
        ckpt2 = self.create_checkpoint(2, 200)
        time.sleep(0.01)
        ckpt3 = self.create_checkpoint(3, 300)

        # Find latest by modification time
        all_ckpts = sorted(
            [d for d in os.listdir(self.test_dir) if d.startswith("ckpt_")],
            key=lambda x: os.path.getmtime(os.path.join(self.test_dir, x)),
            reverse=True
        )

        if all_ckpts:
            latest = os.path.join(self.test_dir, all_ckpts[0])
            self.assertEqual(latest, ckpt3)

    def test_resume_state_consistency(self):
        """Test that resumed state matches saved state."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Create and save checkpoint
        model1 = MockModel()
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

        # Do some training steps to change state
        for _ in range(5):
            optimizer1.zero_grad()
            loss = model1(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer1.step()

        # Save checkpoint
        ckpt_path = os.path.join(self.test_dir, "checkpoint")
        os.makedirs(ckpt_path, exist_ok=True)

        torch.save(model1.state_dict(), os.path.join(ckpt_path, "model.pt"))
        torch.save(optimizer1.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
        torch.save({"epoch": 5, "step": 500}, os.path.join(ckpt_path, "state.pt"))

        # Create new model and load checkpoint
        model2 = MockModel()
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

        model2.load_state_dict(torch.load(os.path.join(ckpt_path, "model.pt")))
        optimizer2.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt")))
        state = torch.load(os.path.join(ckpt_path, "state.pt"))

        # Verify state matches
        self.assertEqual(state["epoch"], 5)
        self.assertEqual(state["step"], 500)

        # Verify model weights match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # Verify optimizer state matches
        state1 = optimizer1.state_dict()
        state2 = optimizer2.state_dict()
        self.assertEqual(state1['param_groups'][0]['lr'], state2['param_groups'][0]['lr'])


class TestGPUUtilization(unittest.TestCase):
    """Test GPU utilization metrics and multi-GPU scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.gpu_available = torch.cuda.is_available() if TORCH_AVAILABLE else False
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0

    def test_gpu_memory_tracking(self):
        """Test GPU memory tracking during training."""
        if not self.gpu_available:
            self.skipTest("GPU not available")

        device = torch.device("cuda:0")

        # Reset stats
        torch.cuda.reset_peak_memory_stats(device)

        # Create model and move to GPU
        model = MockModel(d_in=1000, d_out=500).to(device)
        optimizer = optim.Adam(model.parameters())

        initial_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB

        # Run some iterations
        peak_memory = 0
        for i in range(10):
            # Large batch to use memory
            batch = torch.randn(256, 1000, device=device)

            optimizer.zero_grad()
            output = model(batch)
            loss = output.sum()
            loss.backward()
            optimizer.step()

            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            peak_memory = max(peak_memory, current_memory)

        # Check peak memory stats
        peak_stats = torch.cuda.max_memory_allocated(device) / 1024**3

        print(f"Initial memory: {initial_memory:.2f} GB")
        print(f"Peak memory: {peak_memory:.2f} GB")
        print(f"Peak stats: {peak_stats:.2f} GB")

        self.assertGreater(peak_memory, initial_memory, "Peak memory should exceed initial")
        self.assertGreater(peak_stats, 0, "Peak stats should be tracked")

    def test_multi_gpu_checkpoint(self):
        """Test checkpointing with multiple GPUs."""
        if self.gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs, found {self.gpu_count}")

        # Create model for DataParallel
        model = MockModel(d_in=100, d_out=50)
        model = nn.DataParallel(model, device_ids=list(range(self.gpu_count)))
        model = model.cuda()

        # Save checkpoint
        ckpt_dir = tempfile.mkdtemp(prefix="test_multigpu_")
        self.addCleanup(shutil.rmtree, ckpt_dir, ignore_errors=True)

        # Save model (need to save module for DataParallel)
        model_state = model.module.state_dict()
        torch.save(model_state, os.path.join(ckpt_dir, "model.pt"))

        # Load into new model
        model2 = MockModel(d_in=100, d_out=50)
        model2.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt")))
        model2 = nn.DataParallel(model2, device_ids=list(range(self.gpu_count)))
        model2 = model2.cuda()

        # Verify weights match
        for p1, p2 in zip(model.module.parameters(), model2.module.parameters()):
            self.assertTrue(torch.allclose(p1, p2))


class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for full training pipeline with preemption."""

    def setUp(self):
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp(prefix="test_integration_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def test_training_with_interruption(self):
        """Test training loop with simulated interruption."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        class PreemptibleTrainer:
            def __init__(self, save_dir, total_steps=100):
                self.save_dir = save_dir
                self.total_steps = total_steps
                self.model = MockModel()
                self.optimizer = optim.Adam(self.parameters())
                self.current_step = 0
                self.losses = []

            def parameters(self):
                return self.model.parameters()

            def save_checkpoint(self):
                """Save current state."""
                artifacts = {
                    "model.pt": self.model.state_dict(),
                    "optimizer.pt": self.optimizer.state_dict(),
                    "state.pt": {
                        "step": self.current_step,
                        "losses": self.losses,
                    },
                }
                save_latest_checkpoint(self.save_dir, artifacts, verbose=False)

            def load_checkpoint(self):
                """Load saved state."""
                model_path = os.path.join(self.save_dir, "model.pt")
                if os.path.exists(model_path):
                    self.model.load_state_dict(torch.load(model_path))
                    self.optimizer.load_state_dict(
                        torch.load(os.path.join(self.save_dir, "optimizer.pt"))
                    )
                    state = torch.load(os.path.join(self.save_dir, "state.pt"))
                    self.current_step = state["step"]
                    self.losses = state["losses"]
                    return True
                return False

            def train(self, interrupt_at=None):
                """Run training with optional interruption."""
                # Try to resume
                if self.load_checkpoint():
                    print(f"Resumed from step {self.current_step}")

                while self.current_step < self.total_steps:
                    # Simulate interruption
                    if interrupt_at and self.current_step == interrupt_at:
                        print(f"Interrupted at step {self.current_step}")
                        self.save_checkpoint()
                        return False  # Not complete

                    # Training step
                    self.optimizer.zero_grad()
                    batch = torch.randn(4, 10)
                    output = self.model(batch)
                    loss = output.sum()
                    loss.backward()
                    self.optimizer.step()

                    self.losses.append(loss.item())
                    self.current_step += 1

                    # Periodic checkpoint
                    if self.current_step % 20 == 0:
                        self.save_checkpoint()

                print(f"Training complete at step {self.current_step}")
                return True  # Complete

        # Run training with interruption
        trainer1 = PreemptibleTrainer(self.test_dir, total_steps=50)
        complete = trainer1.train(interrupt_at=30)
        self.assertFalse(complete, "Should be interrupted")
        self.assertEqual(trainer1.current_step, 30)

        # Resume training
        trainer2 = PreemptibleTrainer(self.test_dir, total_steps=50)
        complete = trainer2.train()
        self.assertTrue(complete, "Should complete after resume")
        self.assertEqual(trainer2.current_step, 50)
        self.assertEqual(len(trainer2.losses), 50)

        # Verify continuous loss history
        self.assertEqual(trainer2.losses[:30], trainer1.losses)


class PerformanceBenchmark:
    """Performance benchmarks for checkpointing."""

    @staticmethod
    def benchmark_checkpoint_speed():
        """Benchmark checkpoint save/load speed."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available for benchmarking")
            return

        sizes = [
            (100, 50, "Small"),
            (1000, 500, "Medium"),
            (5000, 2000, "Large"),
        ]

        print("\nCheckpoint Save/Load Benchmarks:")
        print("-" * 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            for d_in, d_out, label in sizes:
                model = MockModel(d_in=d_in, d_out=d_out)
                optimizer = optim.Adam(model.parameters())

                # Benchmark save
                artifacts = {
                    "model.pt": model.state_dict(),
                    "optimizer.pt": optimizer.state_dict(),
                    "state.pt": {"epoch": 1, "step": 100},
                }

                start = time.time()
                save_latest_checkpoint(tmpdir, artifacts, verbose=False)
                save_time = time.time() - start

                # Benchmark load
                start = time.time()
                loaded_model = torch.load(os.path.join(tmpdir, "model.pt"))
                loaded_opt = torch.load(os.path.join(tmpdir, "optimizer.pt"))
                load_time = time.time() - start

                # Get checkpoint size
                total_size = sum(
                    os.path.getsize(os.path.join(tmpdir, f))
                    for f in os.listdir(tmpdir)
                    if os.path.isfile(os.path.join(tmpdir, f))
                )
                size_mb = total_size / (1024 * 1024)

                print(f"{label:10} | Size: {size_mb:6.1f} MB | "
                      f"Save: {save_time*1000:6.1f} ms | "
                      f"Load: {load_time*1000:6.1f} ms")

    @staticmethod
    def benchmark_pruning_speed():
        """Benchmark directory pruning speed."""
        print("\nDirectory Pruning Benchmarks:")
        print("-" * 50)

        file_counts = [10, 50, 100, 500]

        for count in file_counts:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create many files and directories
                for i in range(count):
                    # Create step directories
                    step_dir = os.path.join(tmpdir, f"step_{i}")
                    os.makedirs(step_dir, exist_ok=True)
                    with open(os.path.join(step_dir, "model.pt"), 'w') as f:
                        f.write("x" * 1000)  # 1KB file

                    # Create temp files
                    with open(os.path.join(tmpdir, f"temp_{i}.tmp"), 'w') as f:
                        f.write("x" * 1000)

                # Benchmark pruning
                start = time.time()
                freed = prune_save_dir(tmpdir, keep_only=["final.pt"])
                prune_time = time.time() - start

                freed_mb = freed / (1024 * 1024)
                print(f"Files: {count:4} | Freed: {freed_mb:6.1f} MB | "
                      f"Time: {prune_time*1000:6.1f} ms")


def run_quick_tests():
    """Run only quick tests (<1 minute)."""
    suite = unittest.TestSuite()

    # Quick checkpoint tests
    suite.addTest(TestCheckpointManager('test_atomic_save'))
    suite.addTest(TestCheckpointManager('test_directory_pruning'))

    # Quick resume tests
    suite.addTest(TestResumeFromCheckpoint('test_resume_state_consistency'))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_specific_test(test_name: str):
    """Run a specific test by name."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_mapping = {
        'checkpoint': TestCheckpointManager,
        'signal': TestSignalHandling,
        'resume': TestResumeFromCheckpoint,
        'gpu': TestGPUUtilization,
        'integration': TestIntegrationPipeline,
    }

    if test_name in test_mapping:
        suite.addTests(loader.loadTestsFromTestCase(test_mapping[test_name]))
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {list(test_mapping.keys())}")
        return False

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test preemptible training system")
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (<1 minute)')
    parser.add_argument('--test', type=str, help='Run specific test (checkpoint/signal/resume/gpu/integration)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    args = parser.parse_args()

    print("=" * 60)
    print("Preemptible Training System Validation Tests")
    print("=" * 60)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)

    if args.benchmark:
        PerformanceBenchmark.benchmark_checkpoint_speed()
        PerformanceBenchmark.benchmark_pruning_speed()
        return

    if args.test:
        success = run_specific_test(args.test)
    elif args.quick:
        print("\nRunning quick tests only...")
        success = run_quick_tests()
    else:
        # Run all tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        success = result.wasSuccessful()

    # Print summary
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())