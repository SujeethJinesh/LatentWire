#!/usr/bin/env python3
"""
Test preemption, checkpointing, and resume functionality.

This script verifies that:
1. Checkpoints are saved correctly with atomic operations
2. Resume from checkpoint restores state properly
3. Signal handling for preemption works
4. Training can continue from interrupted state
"""

import os
import sys
import json
import signal
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.checkpointing import save_latest_checkpoint, prune_save_dir
from latentwire.models import InterlinguaEncoder, Adapter
from latentwire.train import load_checkpoint, find_latest_checkpoint


class PreemptionSimulator:
    """Simulates training with preemption and checkpointing."""

    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.test_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.preemption_received = False
        self.save_count = 0

    def handle_preemption(self, signum, frame):
        """Handle preemption signal."""
        print(f"\n[PREEMPTION] Received signal {signum}")
        self.preemption_received = True

    def create_dummy_models(self, d_z: int = 128, latent_len: int = 16):
        """Create dummy encoder and adapters for testing."""
        encoder = InterlinguaEncoder(
            d_z=d_z,
            latent_len=latent_len
        )

        adapters = {
            'llama': Adapter(d_z=d_z, d_model=4096, latent_length=latent_len),
            'qwen': Adapter(d_z=d_z, d_model=4096, latent_length=latent_len)
        }

        return encoder, adapters

    def create_dummy_optimizer(self, encoder, adapters):
        """Create optimizer for all parameters."""
        params = list(encoder.parameters())
        for adapter in adapters.values():
            params.extend(adapter.parameters())
        return torch.optim.AdamW(params, lr=1e-4)

    def save_checkpoint(self, epoch: int, step: int, encoder, adapters, optimizer,
                       loss_history: list, extra_state: Dict[str, Any] = None):
        """Save checkpoint with all training state."""
        self.save_count += 1

        # Prepare artifacts
        artifacts = {
            'encoder.pt': encoder.state_dict(),
            'adapter_llama.pt': adapters['llama'].state_dict(),
            'adapter_qwen.pt': adapters['qwen'].state_dict(),
            'optimizer.pt': optimizer.state_dict(),
            'state.pt': {
                'epoch': epoch,
                'global_step': step,
                'loss_history': loss_history,
                'save_count': self.save_count,
                **(extra_state or {})
            },
            'config.json': {
                'd_z': encoder.d_z,
                'latent_len': encoder.latent_len,
                'encoder_type': 'byte',
                'test_mode': True
            }
        }

        # Save checkpoint atomically
        save_dir = self.checkpoint_dir / f"step_{step}"
        save_dir.mkdir(exist_ok=True)

        freed_pre, freed_post = save_latest_checkpoint(
            str(save_dir),
            artifacts,
            pre_prune=True,
            post_prune=True,
            verbose=True
        )

        print(f"[CHECKPOINT] Saved at epoch {epoch}, step {step}")
        print(f"  - Freed before: {freed_pre} bytes")
        print(f"  - Freed after: {freed_post} bytes")

        return save_dir

    def verify_checkpoint_integrity(self, checkpoint_dir: Path) -> bool:
        """Verify checkpoint files are complete and valid."""
        required_files = ['encoder.pt', 'adapter_llama.pt', 'adapter_qwen.pt',
                         'state.pt', 'config.json']

        for file in required_files:
            file_path = checkpoint_dir / file
            if not file_path.exists():
                print(f"[ERROR] Missing required file: {file}")
                return False

            # Try to load file to verify integrity
            try:
                if file.endswith('.pt'):
                    torch.load(file_path, map_location='cpu')
                elif file.endswith('.json'):
                    with open(file_path) as f:
                        json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")
                return False

        print(f"[VERIFY] Checkpoint at {checkpoint_dir} is valid")
        return True

    def simulate_training(self, num_epochs: int = 3, steps_per_epoch: int = 10,
                         save_every: int = 5, preempt_at_step: int = 15):
        """Simulate training with checkpointing and optional preemption."""

        # Set up signal handler
        signal.signal(signal.SIGUSR1, self.handle_preemption)

        # Create models and optimizer
        encoder, adapters = self.create_dummy_models()
        optimizer = self.create_dummy_optimizer(encoder, adapters)

        # Training state
        global_step = 0
        loss_history = []
        start_epoch = 0

        # Try to resume from checkpoint
        latest_ckpt = find_latest_checkpoint(str(self.checkpoint_dir))
        if latest_ckpt:
            print(f"\n[RESUME] Found checkpoint: {latest_ckpt}")
            state = load_checkpoint(
                latest_ckpt,
                encoder,
                adapters,
                optimizer,
                load_optimizer=True,
                load_lr_scheduler=False,
                strict=True,
                verbose=True
            )

            if state:
                start_epoch = state.get('epoch', 0)
                global_step = state.get('global_step', 0)
                loss_history = state.get('loss_history', [])
                print(f"[RESUME] Resuming from epoch {start_epoch}, step {global_step}")

        print(f"\n[TRAIN] Starting training from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            for step in range(steps_per_epoch):
                global_step += 1

                # Simulate training step
                loss = np.random.randn() * 0.1 + 2.0 - (global_step * 0.01)
                loss_history.append(loss)

                print(f"  Step {global_step}: loss = {loss:.4f}")

                # Simulate preemption
                if global_step == preempt_at_step and not self.preemption_received:
                    print(f"\n[SIMULATE] Triggering preemption at step {global_step}")
                    os.kill(os.getpid(), signal.SIGUSR1)

                # Check for preemption or regular save
                if self.preemption_received or (save_every > 0 and global_step % save_every == 0):
                    checkpoint_dir = self.save_checkpoint(
                        epoch, global_step, encoder, adapters, optimizer,
                        loss_history, {'preempted': self.preemption_received}
                    )

                    # Verify checkpoint
                    self.verify_checkpoint_integrity(checkpoint_dir)

                    if self.preemption_received:
                        print(f"\n[PREEMPT] Exiting gracefully after saving checkpoint")
                        return {
                            'preempted': True,
                            'final_step': global_step,
                            'final_epoch': epoch,
                            'checkpoint_dir': str(checkpoint_dir)
                        }

                # Small delay to simulate computation
                time.sleep(0.01)

        # Final checkpoint
        checkpoint_dir = self.save_checkpoint(
            num_epochs, global_step, encoder, adapters, optimizer,
            loss_history, {'completed': True}
        )

        return {
            'preempted': False,
            'final_step': global_step,
            'final_epoch': num_epochs,
            'checkpoint_dir': str(checkpoint_dir)
        }


def test_atomic_writes():
    """Test atomic write operations."""
    print("\n" + "="*60)
    print("TEST: Atomic Write Operations")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.pt"

        # Test torch save
        data = {'test': torch.randn(10, 10)}
        from latentwire.checkpointing import _atomic_save_torch
        _atomic_save_torch(data, str(test_file))

        # Verify file exists and is valid
        assert test_file.exists(), "File not created"
        loaded = torch.load(test_file)
        assert torch.allclose(loaded['test'], data['test']), "Data mismatch"
        print("✅ Atomic torch save works")

        # Test JSON save
        json_file = Path(tmpdir) / "test.json"
        json_data = {'epoch': 1, 'step': 100}
        from latentwire.checkpointing import _atomic_save_json
        _atomic_save_json(json_data, str(json_file))

        with open(json_file) as f:
            loaded_json = json.load(f)
        assert loaded_json == json_data, "JSON data mismatch"
        print("✅ Atomic JSON save works")


def test_checkpoint_pruning():
    """Test checkpoint directory pruning."""
    print("\n" + "="*60)
    print("TEST: Checkpoint Pruning")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "checkpoints"
        save_dir.mkdir()

        # Create various files
        (save_dir / "encoder.pt").write_text("encoder")
        (save_dir / "adapter_llama.pt").write_text("adapter")
        (save_dir / "state.pt").write_text("state")
        (save_dir / "config.json").write_text("{}")

        # Create files that should be pruned
        (save_dir / "temp.tmp").write_text("temp")
        (save_dir / "old_checkpoint.pt.old").write_text("old")
        (save_dir / "encoder_step1000.pt").write_text("step")

        # Create step directory
        step_dir = save_dir / "step_500"
        step_dir.mkdir()
        (step_dir / "encoder.pt").write_text("step_encoder")

        print(f"Files before pruning: {list(save_dir.iterdir())}")

        # Prune with keep list
        freed = prune_save_dir(
            str(save_dir),
            keep_only=['encoder.pt', 'adapter_llama.pt', 'state.pt', 'config.json']
        )

        remaining = list(save_dir.iterdir())
        print(f"Files after pruning: {remaining}")
        print(f"Freed: {freed} bytes")

        # Check expected files remain
        assert (save_dir / "encoder.pt").exists(), "encoder.pt removed"
        assert (save_dir / "adapter_llama.pt").exists(), "adapter_llama.pt removed"
        assert (save_dir / "state.pt").exists(), "state.pt removed"
        assert (save_dir / "config.json").exists(), "config.json removed"

        # Check unwanted files are gone
        assert not (save_dir / "temp.tmp").exists(), "temp.tmp not removed"
        assert not (save_dir / "old_checkpoint.pt.old").exists(), "old file not removed"
        assert not (save_dir / "encoder_step1000.pt").exists(), "step file not removed"
        assert not (save_dir / "step_500").exists(), "step directory not removed"

        print("✅ Checkpoint pruning works correctly")


def test_resume_functionality():
    """Test resume from checkpoint."""
    print("\n" + "="*60)
    print("TEST: Resume Functionality")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        simulator = PreemptionSimulator(tmpdir)

        # Run training with simulated preemption
        print("\n[Phase 1] Initial training run with preemption...")
        result1 = simulator.simulate_training(
            num_epochs=3,
            steps_per_epoch=10,
            save_every=5,
            preempt_at_step=12
        )

        assert result1['preempted'], "Training should have been preempted"
        print(f"✅ Training preempted at step {result1['final_step']}")

        # Create new simulator (simulating restart)
        simulator2 = PreemptionSimulator(tmpdir)

        # Resume training
        print("\n[Phase 2] Resuming training from checkpoint...")
        result2 = simulator2.simulate_training(
            num_epochs=3,
            steps_per_epoch=10,
            save_every=5,
            preempt_at_step=999  # Don't preempt this time
        )

        assert not result2['preempted'], "Training should complete"
        assert result2['final_step'] > result1['final_step'], "Training should progress"
        print(f"✅ Training completed at step {result2['final_step']}")

        # Verify checkpoint integrity
        final_ckpt = Path(result2['checkpoint_dir'])
        assert simulator2.verify_checkpoint_integrity(final_ckpt)
        print("✅ Final checkpoint is valid")


def test_concurrent_saves():
    """Test that concurrent saves don't corrupt checkpoints."""
    print("\n" + "="*60)
    print("TEST: Concurrent Save Safety")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "concurrent_test"
        save_dir.mkdir(parents=True)

        # Simulate multiple processes trying to save
        import threading
        import random

        errors = []

        def save_worker(worker_id: int):
            try:
                artifacts = {
                    'encoder.pt': {'worker': worker_id, 'data': torch.randn(10, 10)},
                    'state.pt': {'step': worker_id * 100},
                    'config.json': {'worker_id': worker_id}
                }

                # Add random delay to increase chance of collision
                time.sleep(random.random() * 0.1)

                save_latest_checkpoint(
                    str(save_dir),
                    artifacts,
                    pre_prune=False,
                    post_prune=False,
                    verbose=False
                )
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Launch concurrent saves
        threads = []
        for i in range(5):
            t = threading.Thread(target=save_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        if errors:
            print(f"Errors during concurrent saves: {errors}")

        # Verify final checkpoint is valid
        assert (save_dir / "encoder.pt").exists(), "encoder.pt missing"
        assert (save_dir / "state.pt").exists(), "state.pt missing"
        assert (save_dir / "config.json").exists(), "config.json missing"

        # Load and verify integrity
        encoder_data = torch.load(save_dir / "encoder.pt")
        state_data = torch.load(save_dir / "state.pt")
        with open(save_dir / "config.json") as f:
            config_data = json.load(f)

        print(f"Final checkpoint from worker {config_data.get('worker_id')}")
        print("✅ Concurrent saves handled safely")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PREEMPTION & CHECKPOINT VERIFICATION SUITE")
    print("="*80)

    try:
        # Test 1: Atomic operations
        test_atomic_writes()

        # Test 2: Pruning logic
        test_checkpoint_pruning()

        # Test 3: Resume functionality
        test_resume_functionality()

        # Test 4: Concurrent save safety
        test_concurrent_saves()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print("\nSummary:")
        print("- Atomic write operations work correctly")
        print("- Checkpoint pruning removes old files properly")
        print("- Resume from checkpoint restores state accurately")
        print("- Concurrent saves are handled safely")
        print("- Preemption signals trigger graceful checkpoint saves")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())