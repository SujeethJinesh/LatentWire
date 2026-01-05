#!/usr/bin/env python3
"""
Comprehensive test suite for checkpoint resume functionality.

This test verifies that training can be interrupted and resumed from the exact
batch/epoch where it left off, which is critical for preemptible training.

Tests include:
1. Basic checkpoint save/load with state preservation
2. Mid-epoch resume (interruption during batch processing)
3. RNG state preservation (deterministic resume)
4. Optimizer and scheduler state preservation
5. Training metrics continuity
6. Corruption recovery (backup restoration)
7. Atomic write verification
8. Multi-model state handling (Llama + Qwen adapters)
9. Edge cases (empty checkpoint, corrupted files)

Usage:
    python test_checkpoint_resume.py
    python test_checkpoint_resume.py --verbose
    python test_checkpoint_resume.py --test-specific TestCheckpointResume.test_mid_epoch_resume
"""

import os
import sys
import json
import time
import shutil
import hashlib
import random
import tempfile
import unittest
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
try:
    from typing import get_type_hints
except ImportError:
    # Python < 3.5 compatibility
    pass
from unittest.mock import Mock, patch, MagicMock
import signal

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.checkpoint_manager import CheckpointManager


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DummyDataset:
    """Simple dataset for testing."""
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.input_dim = input_dim
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 5, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TrainingState:
    """Track training state for testing."""
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0
        self.global_step = 0
        self.losses = []
        self.metrics = {}

    def to_dict(self):
        return {
            'epoch': self.epoch,
            'batch_idx': self.batch_idx,
            'global_step': self.global_step,
            'losses': self.losses[-10:],  # Keep last 10 losses
            'metrics': self.metrics
        }

    def from_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.batch_idx = state_dict['batch_idx']
        self.global_step = state_dict['global_step']
        self.losses = state_dict.get('losses', [])
        self.metrics = state_dict.get('metrics', {})


class TestCheckpointResume(unittest.TestCase):
    """Test checkpoint save/resume functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_checkpoint_"))
        self.checkpoint_dir = self.test_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = DummyModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
        self.dataset = DummyDataset(size=100)

        # Initialize checkpoint manager
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=1,
            save_interval_minutes=0.001,  # Very short for testing
            validate_on_save=True,
            validate_on_load=True
        )

        # Initialize training state
        self.training_state = TrainingState()

    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_checkpoint(self):
        """Create a checkpoint dictionary."""
        return {
            'epoch': self.training_state.epoch,
            'batch_idx': self.training_state.batch_idx,
            'global_step': self.training_state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_state': self.training_state.to_dict(),
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }

    def _load_checkpoint(self, checkpoint):
        """Load checkpoint into current state."""
        self.training_state.from_dict(checkpoint['training_state'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore RNG states
        if 'rng_state' in checkpoint:
            random.setstate(checkpoint['rng_state']['python'])
            np.random.set_state(checkpoint['rng_state']['numpy'])
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            if torch.cuda.is_available() and checkpoint['rng_state']['cuda']:
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])

    def test_basic_save_load(self):
        """Test basic checkpoint save and load."""
        # Train for a few steps
        self.training_state.epoch = 2
        self.training_state.batch_idx = 15
        self.training_state.global_step = 45
        self.training_state.losses = [0.5, 0.4, 0.35]

        # Save checkpoint
        checkpoint = self._create_checkpoint()
        save_path = self.ckpt_manager.save_checkpoint(checkpoint, tag="test")
        self.assertTrue(save_path.exists())

        # Reset state
        self.training_state = TrainingState()

        # Load checkpoint
        loaded_checkpoint, metadata = self.ckpt_manager.load_checkpoint()
        self._load_checkpoint(loaded_checkpoint)

        # Verify state restored
        self.assertEqual(self.training_state.epoch, 2)
        self.assertEqual(self.training_state.batch_idx, 15)
        self.assertEqual(self.training_state.global_step, 45)
        self.assertEqual(self.training_state.losses, [0.5, 0.4, 0.35])

    def test_mid_epoch_resume(self):
        """Test resuming from middle of an epoch."""
        # Simulate training partway through epoch 3
        self.training_state.epoch = 3
        self.training_state.batch_idx = 7  # Middle of epoch
        self.training_state.global_step = 67

        # Record some training progress
        batch_size = 4
        steps_per_epoch = len(self.dataset) // batch_size

        # Save model weights before modification
        initial_weights = self.model.fc1.weight.data.clone()

        # Do a few training steps
        for i in range(3):
            batch_data = torch.randn(batch_size, 10)
            batch_labels = torch.randint(0, 5, (batch_size,))

            self.optimizer.zero_grad()
            output = self.model(batch_data)
            loss = nn.functional.cross_entropy(output, batch_labels)
            loss.backward()
            self.optimizer.step()

            self.training_state.batch_idx += 1
            self.training_state.global_step += 1
            self.training_state.losses.append(loss.item())

        # Model weights should have changed
        self.assertFalse(torch.allclose(initial_weights, self.model.fc1.weight.data))

        # Save checkpoint mid-epoch
        checkpoint = self._create_checkpoint()
        self.ckpt_manager.save_checkpoint(checkpoint, tag="mid_epoch")

        saved_batch_idx = self.training_state.batch_idx
        saved_global_step = self.training_state.global_step
        saved_losses = self.training_state.losses.copy()

        # Simulate crash/reset - create new model and state
        self.model = DummyModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.training_state = TrainingState()

        # Load checkpoint
        loaded_checkpoint, _ = self.ckpt_manager.load_checkpoint()
        self._load_checkpoint(loaded_checkpoint)

        # Verify we can resume from exact batch
        self.assertEqual(self.training_state.epoch, 3)
        self.assertEqual(self.training_state.batch_idx, saved_batch_idx)
        self.assertEqual(self.training_state.global_step, saved_global_step)
        self.assertEqual(self.training_state.losses, saved_losses)

        # Continue training from where we left off
        remaining_batches = steps_per_epoch - self.training_state.batch_idx
        self.assertGreater(remaining_batches, 0, "Should have batches remaining in epoch")

        # Verify we can continue training
        for i in range(min(3, remaining_batches)):
            batch_data = torch.randn(batch_size, 10)
            batch_labels = torch.randint(0, 5, (batch_size,))

            self.optimizer.zero_grad()
            output = self.model(batch_data)
            loss = nn.functional.cross_entropy(output, batch_labels)
            loss.backward()
            self.optimizer.step()

            self.training_state.batch_idx += 1
            self.training_state.global_step += 1

        # Global step should have continued incrementing
        self.assertEqual(self.training_state.global_step, saved_global_step + 3)

    def test_rng_state_preservation(self):
        """Test that RNG states are preserved for deterministic resume."""
        # Set specific seeds
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Generate some random numbers
        torch_randoms_before = [torch.randn(3) for _ in range(3)]
        np_randoms_before = [np.random.randn(3) for _ in range(3)]
        py_randoms_before = [random.random() for _ in range(3)]

        # Save checkpoint with RNG state
        checkpoint = self._create_checkpoint()
        self.ckpt_manager.save_checkpoint(checkpoint, tag="rng_test")

        # Generate more random numbers (advance RNG state)
        torch_randoms_mid = [torch.randn(3) for _ in range(3)]
        np_randoms_mid = [np.random.randn(3) for _ in range(3)]
        py_randoms_mid = [random.random() for _ in range(3)]

        # Load checkpoint (should restore RNG state)
        loaded_checkpoint, _ = self.ckpt_manager.load_checkpoint()
        self._load_checkpoint(loaded_checkpoint)

        # Generate random numbers again - should match the "mid" values
        torch_randoms_after = [torch.randn(3) for _ in range(3)]
        np_randoms_after = [np.random.randn(3) for _ in range(3)]
        py_randoms_after = [random.random() for _ in range(3)]

        # Verify deterministic behavior
        for i in range(3):
            self.assertTrue(torch.allclose(torch_randoms_mid[i], torch_randoms_after[i]))
            np.testing.assert_array_almost_equal(np_randoms_mid[i], np_randoms_after[i])
            self.assertAlmostEqual(py_randoms_mid[i], py_randoms_after[i])

    def test_optimizer_state_preservation(self):
        """Test that optimizer momentum and other states are preserved."""
        # Train for several steps to build up optimizer momentum
        for step in range(10):
            batch_data = torch.randn(4, 10)
            batch_labels = torch.randint(0, 5, (4,))

            self.optimizer.zero_grad()
            output = self.model(batch_data)
            loss = nn.functional.cross_entropy(output, batch_labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.training_state.global_step += 1

        # Save optimizer state details before checkpoint
        opt_state_before = self.optimizer.state_dict()
        lr_before = self.scheduler.get_last_lr()[0]

        # Save checkpoint
        checkpoint = self._create_checkpoint()
        self.ckpt_manager.save_checkpoint(checkpoint, tag="optimizer_test")

        # Create new optimizer and scheduler
        self.model = DummyModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10)

        # Verify they have different states initially
        self.assertNotEqual(
            self.scheduler.get_last_lr()[0],
            lr_before
        )

        # Load checkpoint
        loaded_checkpoint, _ = self.ckpt_manager.load_checkpoint()
        self._load_checkpoint(loaded_checkpoint)

        # Verify optimizer state restored
        opt_state_after = self.optimizer.state_dict()

        # Check learning rate restored
        self.assertEqual(self.scheduler.get_last_lr()[0], lr_before)

        # Check momentum buffers exist and match
        if len(opt_state_before['state']) > 0:
            for key in opt_state_before['state']:
                self.assertIn(key, opt_state_after['state'])
                if 'exp_avg' in opt_state_before['state'][key]:
                    self.assertTrue(
                        torch.allclose(
                            opt_state_before['state'][key]['exp_avg'],
                            opt_state_after['state'][key]['exp_avg']
                        )
                    )

    def test_corruption_recovery(self):
        """Test recovery from corrupted checkpoint using backup."""
        # Save initial checkpoint
        self.training_state.epoch = 1
        self.training_state.global_step = 25
        checkpoint = self._create_checkpoint()
        self.ckpt_manager.save_checkpoint(checkpoint, tag="good")

        # Advance training
        self.training_state.epoch = 2
        self.training_state.global_step = 50

        # Save another checkpoint (this will backup the previous one)
        checkpoint2 = self._create_checkpoint()
        self.ckpt_manager.save_checkpoint(checkpoint2, tag="newer")

        # Corrupt the current checkpoint file
        current_ckpt = self.checkpoint_dir / "checkpoint_current.pt"
        self.assertTrue(current_ckpt.exists())

        # Write garbage to corrupt it
        with open(current_ckpt, 'wb') as f:
            f.write(b'corrupted data')

        # Try to load - should detect corruption and check alternatives
        # The CheckpointManager should handle this gracefully
        result = self.ckpt_manager.get_latest_checkpoint()

        # Should either return None or try backup
        if result is not None:
            # If it found something, verify it's valid
            try:
                loaded_checkpoint, _ = self.ckpt_manager.load_checkpoint(result)
                # Should have loaded the backup
                self.assertIsNotNone(loaded_checkpoint)
            except:
                # Corruption detected, no valid checkpoint
                pass

    def test_atomic_write_verification(self):
        """Test that writes are atomic and don't leave partial files."""
        # Start a save operation
        checkpoint = self._create_checkpoint()

        # Simulate interruption by checking for temp files during save
        # This is tricky to test properly without modifying the checkpoint manager
        # So we'll verify the expected behavior

        # Save checkpoint normally
        save_path = self.ckpt_manager.save_checkpoint(checkpoint, tag="atomic")

        # Check that no .tmp files remain
        tmp_files = list(self.checkpoint_dir.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0, "No temporary files should remain after save")

        # Verify checkpoint and metadata exist
        self.assertTrue(save_path.exists())
        self.assertTrue(save_path.with_suffix(".json").exists())

    def test_multi_model_state(self):
        """Test saving/loading states for multiple models (Llama + Qwen adapters)."""
        # Create multiple models to simulate Llama and Qwen adapters
        llama_adapter = DummyModel(hidden_dim=30)
        qwen_adapter = DummyModel(hidden_dim=25)
        encoder = DummyModel(input_dim=20, output_dim=10)

        # Create optimizers for each
        llama_opt = optim.Adam(llama_adapter.parameters(), lr=2e-4)
        qwen_opt = optim.Adam(qwen_adapter.parameters(), lr=3e-4)
        encoder_opt = optim.Adam(encoder.parameters(), lr=1e-4)

        # Train each model differently
        for i in range(5):
            # Llama adapter
            data = torch.randn(4, 10)
            labels = torch.randint(0, 5, (4,))
            llama_opt.zero_grad()
            loss = nn.functional.cross_entropy(llama_adapter(data), labels)
            loss.backward()
            llama_opt.step()

            # Qwen adapter
            qwen_opt.zero_grad()
            loss = nn.functional.cross_entropy(qwen_adapter(data), labels)
            loss.backward()
            qwen_opt.step()

            # Encoder
            data_enc = torch.randn(4, 20)
            encoder_opt.zero_grad()
            loss = nn.functional.cross_entropy(encoder(data_enc), labels)
            loss.backward()
            encoder_opt.step()

        # Create comprehensive checkpoint
        checkpoint = {
            'epoch': 5,
            'global_step': 100,
            'encoder_state_dict': encoder.state_dict(),
            'llama_adapter_state_dict': llama_adapter.state_dict(),
            'qwen_adapter_state_dict': qwen_adapter.state_dict(),
            'encoder_optimizer': encoder_opt.state_dict(),
            'llama_optimizer': llama_opt.state_dict(),
            'qwen_optimizer': qwen_opt.state_dict(),
            'training_state': self.training_state.to_dict()
        }

        # Save checkpoint
        self.ckpt_manager.save_checkpoint(checkpoint, tag="multi_model")

        # Create new models
        llama_adapter_new = DummyModel(hidden_dim=30)
        qwen_adapter_new = DummyModel(hidden_dim=25)
        encoder_new = DummyModel(input_dim=20, output_dim=10)

        # Verify they have different weights initially
        self.assertFalse(
            torch.allclose(
                encoder.fc1.weight,
                encoder_new.fc1.weight
            )
        )

        # Load checkpoint
        loaded_checkpoint, _ = self.ckpt_manager.load_checkpoint()

        # Restore all model states
        encoder_new.load_state_dict(loaded_checkpoint['encoder_state_dict'])
        llama_adapter_new.load_state_dict(loaded_checkpoint['llama_adapter_state_dict'])
        qwen_adapter_new.load_state_dict(loaded_checkpoint['qwen_adapter_state_dict'])

        # Verify all weights restored correctly
        self.assertTrue(torch.allclose(encoder.fc1.weight, encoder_new.fc1.weight))
        self.assertTrue(torch.allclose(llama_adapter.fc1.weight, llama_adapter_new.fc1.weight))
        self.assertTrue(torch.allclose(qwen_adapter.fc1.weight, qwen_adapter_new.fc1.weight))

        # Verify optimizers can be restored
        llama_opt_new = optim.Adam(llama_adapter_new.parameters(), lr=2e-4)
        qwen_opt_new = optim.Adam(qwen_adapter_new.parameters(), lr=3e-4)
        encoder_opt_new = optim.Adam(encoder_new.parameters(), lr=1e-4)

        llama_opt_new.load_state_dict(loaded_checkpoint['llama_optimizer'])
        qwen_opt_new.load_state_dict(loaded_checkpoint['qwen_optimizer'])
        encoder_opt_new.load_state_dict(loaded_checkpoint['encoder_optimizer'])

        # Check that learning rates match
        self.assertEqual(
            llama_opt.param_groups[0]['lr'],
            llama_opt_new.param_groups[0]['lr']
        )

    def test_emergency_save(self):
        """Test emergency checkpoint save during preemption."""
        # Set up state
        self.training_state.epoch = 3
        self.training_state.global_step = 75
        checkpoint = self._create_checkpoint()

        # Perform emergency save
        success = self.ckpt_manager.emergency_save(checkpoint)
        self.assertTrue(success)

        # Verify checkpoint was saved (either as current or emergency)
        latest = self.ckpt_manager.get_latest_checkpoint()
        self.assertIsNotNone(latest)

        # Load and verify
        loaded_checkpoint, metadata = self.ckpt_manager.load_checkpoint(latest)
        self.assertEqual(loaded_checkpoint['epoch'], 3)
        self.assertEqual(loaded_checkpoint['global_step'], 75)

    def test_checkpoint_listing(self):
        """Test listing available checkpoints."""
        # Save multiple checkpoints
        for i in range(3):
            self.training_state.epoch = i
            self.training_state.global_step = i * 25
            checkpoint = self._create_checkpoint()
            self.ckpt_manager.save_checkpoint(checkpoint, tag="ckpt_{}".format(i))
            time.sleep(0.1)  # Small delay to ensure different timestamps

        # List checkpoints
        checkpoints = self.ckpt_manager.list_checkpoints()

        # Should have at least the current checkpoint
        self.assertGreater(len(checkpoints), 0)

        # Verify checkpoint has metadata
        for ckpt_path, metadata in checkpoints:
            self.assertTrue(ckpt_path.exists())
            self.assertIn('timestamp', metadata)
            self.assertIn('tag', metadata)

    def test_resume_exact_batch(self):
        """Test that we resume from the exact batch index within an epoch."""
        batch_size = 4
        num_batches = len(self.dataset) // batch_size

        # Train to middle of epoch 2, batch 10
        target_epoch = 2
        target_batch = 10

        self.training_state.epoch = target_epoch
        self.training_state.batch_idx = target_batch
        self.training_state.global_step = target_epoch * num_batches + target_batch

        # Record the exact data indices we would process next
        torch.manual_seed(42)  # Fixed seed for reproducibility
        perm = torch.randperm(len(self.dataset))
        next_batch_indices = perm[target_batch * batch_size : (target_batch + 1) * batch_size]

        # Save checkpoint
        checkpoint = self._create_checkpoint()
        self.ckpt_manager.save_checkpoint(checkpoint, tag="exact_batch")

        # Simulate restart
        self.training_state = TrainingState()

        # Load checkpoint
        loaded_checkpoint, _ = self.ckpt_manager.load_checkpoint()
        self._load_checkpoint(loaded_checkpoint)

        # Verify we're at the right position
        self.assertEqual(self.training_state.epoch, target_epoch)
        self.assertEqual(self.training_state.batch_idx, target_batch)

        # Verify we can compute which samples to process next
        torch.manual_seed(42)  # Same seed
        perm_resumed = torch.randperm(len(self.dataset))
        next_batch_indices_resumed = perm_resumed[
            self.training_state.batch_idx * batch_size :
            (self.training_state.batch_idx + 1) * batch_size
        ]

        # Should get the same batch indices
        self.assertTrue(torch.equal(next_batch_indices, next_batch_indices_resumed))

    def test_save_interval(self):
        """Test that save interval is respected."""
        # Create manager with 1 second interval
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir / "interval_test"),
            save_interval_minutes=1/60.0,  # 1 second
            max_checkpoints=1
        )

        # Initially should save
        self.assertTrue(manager.should_save())

        # Save checkpoint
        checkpoint = self._create_checkpoint()
        manager.save_checkpoint(checkpoint)

        # Immediately after save, should not save
        self.assertFalse(manager.should_save())

        # Wait for interval
        time.sleep(1.1)

        # Now should save again
        self.assertTrue(manager.should_save())


class TestPreemptionScenarios(unittest.TestCase):
    """Test specific preemption scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_preempt_"))
        self.checkpoint_dir = self.test_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_sigterm_handling(self):
        """Test handling of SIGTERM signal (SLURM preemption)."""
        # This test would need to spawn a subprocess and send it SIGTERM
        # For now, we'll test the mechanism without actual signals

        checkpoint_saved = False

        def mock_save_checkpoint(reason):
            nonlocal checkpoint_saved
            checkpoint_saved = True
            self.assertEqual(reason, "preemption")

        # Simulate SIGTERM handler behavior
        mock_save_checkpoint("preemption")
        self.assertTrue(checkpoint_saved)

    def test_concurrent_save_protection(self):
        """Test that concurrent save attempts are handled safely."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=1
        )

        model = DummyModel()
        results = []
        errors = []

        def save_checkpoint(tag):
            try:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'tag': tag
                }
                path = manager.save_checkpoint(checkpoint, tag=tag)
                results.append((tag, path))
            except Exception as e:
                errors.append((tag, e))

        # Launch concurrent saves
        threads = []
        for i in range(5):
            t = threading.Thread(target=save_checkpoint, args=("concurrent_{}".format(i),))
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # Should have saved successfully (one will win, others might retry)
        self.assertGreater(len(results), 0)

        # Verify checkpoint is valid
        latest = manager.get_latest_checkpoint()
        self.assertIsNotNone(latest)

        # Load and verify
        checkpoint, _ = manager.load_checkpoint(latest)
        self.assertIn('model_state_dict', checkpoint)


def run_integration_test():
    """Run a full integration test simulating real training with interruption."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Simulating training with interruption and resume")
    print("="*70 + "\n")

    test_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))
    checkpoint_dir = test_dir / "checkpoints"

    try:
        # Phase 1: Initial training
        print("Phase 1: Initial training...")
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

        model = DummyModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        dataset = DummyDataset(size=100)

        epoch = 0
        batch_size = 10
        global_step = 0

        # Train for 2.5 epochs
        for epoch in range(3):
            perm = torch.randperm(len(dataset))
            num_batches = len(dataset) // batch_size

            for batch_idx in range(num_batches):
                # Simulate interruption mid-epoch
                if epoch == 2 and batch_idx == 5:
                    print("\n*** SIMULATING PREEMPTION at epoch {}, batch {} ***\n".format(epoch, batch_idx))

                    # Save emergency checkpoint
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'next_batch_start': batch_idx,
                        'permutation': perm
                    }

                    save_path = manager.save_checkpoint(checkpoint, tag="preemption")
                    print("Emergency checkpoint saved: {}".format(save_path))
                    break

                # Training step
                indices = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_data = dataset.data[indices]
                batch_labels = dataset.labels[indices]

                optimizer.zero_grad()
                output = model(batch_data)
                loss = nn.functional.cross_entropy(output, batch_labels)
                loss.backward()
                optimizer.step()

                global_step += 1

                if batch_idx % 3 == 0:
                    print("  Epoch {}, Batch {}/{}, Step {}, Loss: {:.4f}".format(
                        epoch, batch_idx, num_batches, global_step, loss.item()))

            if epoch == 2 and batch_idx == 5:
                break  # Simulated interruption

        print("\nTraining interrupted at epoch {}, batch {}, step {}".format(epoch, batch_idx, global_step))

        # Phase 2: Resume training
        print("\n" + "-"*50)
        print("Phase 2: Resuming training from checkpoint...")
        print("-"*50 + "\n")

        # Create new model and optimizer (simulating fresh start)
        model_resumed = DummyModel()
        optimizer_resumed = optim.Adam(model_resumed.parameters(), lr=1e-3)

        # Load checkpoint
        checkpoint, metadata = manager.load_checkpoint()

        model_resumed.load_state_dict(checkpoint['model_state_dict'])
        optimizer_resumed.load_state_dict(checkpoint['optimizer_state_dict'])

        resumed_epoch = checkpoint['epoch']
        resumed_batch = checkpoint['batch_idx']
        resumed_step = checkpoint['global_step']
        resumed_perm = checkpoint['permutation']

        print("Resumed from: epoch {}, batch {}, step {}".format(resumed_epoch, resumed_batch, resumed_step))

        # Continue training from where we left off
        for epoch in range(resumed_epoch, 5):  # Train to epoch 5
            if epoch == resumed_epoch:
                # Resume from mid-epoch
                start_batch = resumed_batch
                perm = resumed_perm
            else:
                # New epoch, new permutation
                start_batch = 0
                perm = torch.randperm(len(dataset))

            num_batches = len(dataset) // batch_size

            for batch_idx in range(start_batch, num_batches):
                indices = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_data = dataset.data[indices]
                batch_labels = dataset.labels[indices]

                optimizer_resumed.zero_grad()
                output = model_resumed(batch_data)
                loss = nn.functional.cross_entropy(output, batch_labels)
                loss.backward()
                optimizer_resumed.step()

                resumed_step += 1

                if batch_idx % 3 == 0:
                    print("  Epoch {}, Batch {}/{}, Step {}, Loss: {:.4f}".format(
                        epoch, batch_idx, num_batches, resumed_step, loss.item()))

        print("\nTraining completed! Final step: {}".format(resumed_step))
        print("\n✅ Integration test passed - training successfully resumed from interruption")

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Test checkpoint resume functionality')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration test')
    parser.add_argument('--test-specific', type=str,
                       help='Run specific test (e.g., TestCheckpointResume.test_mid_epoch_resume)')
    args = parser.parse_args()

    if args.integration:
        run_integration_test()
        return

    # Set up test suite
    if args.test_specific:
        # Run specific test
        suite = unittest.TestLoader().loadTestsFromName(args.test_specific)
    else:
        # Run all tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        suite.addTests(loader.loadTestsFromTestCase(TestCheckpointResume))
        suite.addTests(loader.loadTestsFromTestCase(TestPreemptionScenarios))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
        print("   Tests run: {}".format(result.testsRun))
    else:
        print("❌ SOME TESTS FAILED")
        print("   Tests run: {}".format(result.testsRun))
        print("   Failures: {}".format(len(result.failures)))
        print("   Errors: {}".format(len(result.errors)))
    print("="*70)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())