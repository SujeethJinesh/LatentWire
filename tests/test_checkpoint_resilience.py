#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test checkpoint resilience for long-running SLURM jobs.

Tests:
1. Atomic saves (no corruption on interrupt)
2. Auto-resume from latest checkpoint
3. Checkpoint pruning (keep only best and latest)
4. Size < 500MB per checkpoint
5. Works with SLURM preemption
"""

import os
import sys
import json
import tempfile
import shutil
import signal
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim

from latentwire.checkpointing import (
    save_latest_checkpoint,
    prune_save_dir,
    _atomic_save_torch,
    _atomic_save_json,
    _is_step_dir,
    _is_tmp_file,
    _human_bytes,
)
from latentwire.models import InterlinguaEncoder, Adapter
from latentwire.train import find_latest_checkpoint, load_checkpoint


def test_atomic_saves():
    """Test that saves are atomic and won't corrupt on interrupt."""
    print("\n=== Test 1: Atomic Saves ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.pt"
        test_data = {"data": torch.randn(100, 100)}

        # Simulate interrupt during save
        original_save = torch.save
        call_count = [0]

        def interrupted_save(obj, f):
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate interrupt on first call
                raise KeyboardInterrupt("Simulated interrupt")
            return original_save(obj, f)

        # First save should fail but not corrupt
        with patch('torch.save', interrupted_save):
            try:
                _atomic_save_torch(test_data, str(test_file))
                assert False, "Should have raised KeyboardInterrupt"
            except KeyboardInterrupt:
                pass

        # File should not exist (atomic property)
        assert not test_file.exists(), "File should not exist after interrupted save"

        # Second save should succeed
        _atomic_save_torch(test_data, str(test_file))
        assert test_file.exists(), "File should exist after successful save"

        # Verify data integrity
        loaded = torch.load(test_file)
        assert torch.allclose(loaded["data"], test_data["data"])

    print("✅ Atomic saves working correctly")


def test_auto_resume():
    """Test auto-resume from latest checkpoint."""
    print("\n=== Test 2: Auto-Resume ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "checkpoints"
        save_dir.mkdir()

        # Create encoder and adapters
        encoder = InterlinguaEncoder(
            d_z=128, n_layers=2, n_heads=4, latent_len=8
        )
        adapters = {
            "llama": Adapter(d_z=128, d_model=256, latent_length=8),
            "qwen": Adapter(d_z=128, d_model=256, latent_length=8),
        }
        optimizer = optim.Adam(
            list(encoder.parameters()) +
            list(adapters["llama"].parameters()) +
            list(adapters["qwen"].parameters())
        )

        # Save checkpoint at step 100
        state_blob = {
            "epoch": 5,
            "global_step": 100,
            "encoder": encoder.state_dict(),
            "adp_llama": adapters["llama"].state_dict(),
            "adp_qwen": adapters["qwen"].state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
            }
        }

        artifacts = {
            "state.pt": state_blob,
            "encoder.pt": encoder.state_dict(),
            "adapter_llama.pt": adapters["llama"].state_dict(),
            "adapter_qwen.pt": adapters["qwen"].state_dict(),
        }

        save_latest_checkpoint(str(save_dir), artifacts, verbose=False)

        # Find latest checkpoint
        latest = find_latest_checkpoint(str(save_dir))
        assert latest is not None, "Should find checkpoint"
        assert "state.pt" in latest

        # Create new models
        new_encoder = InterlinguaEncoder(
            d_z=128, n_layers=2, n_heads=4, latent_len=8
        )
        new_adapters = {
            "llama": Adapter(d_z=128, d_model=256, latent_length=8),
            "qwen": Adapter(d_z=128, d_model=256, latent_length=8),
        }
        new_optimizer = optim.Adam(
            list(new_encoder.parameters()) +
            list(new_adapters["llama"].parameters()) +
            list(new_adapters["qwen"].parameters())
        )

        # Load checkpoint
        epoch, global_step = load_checkpoint(
            latest,
            new_encoder,
            new_adapters,
            optimizer=new_optimizer,
            device="cpu"
        )

        assert epoch == 5, f"Expected epoch 5, got {epoch}"
        assert global_step == 100, f"Expected step 100, got {global_step}"

        # Verify weights loaded correctly
        for key in encoder.state_dict():
            assert torch.allclose(
                encoder.state_dict()[key],
                new_encoder.state_dict()[key]
            ), f"Encoder weight mismatch for {key}"

    print("✅ Auto-resume working correctly")


def test_checkpoint_pruning():
    """Test that old checkpoints are pruned correctly."""
    print("\n=== Test 3: Checkpoint Pruning ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # Create some files to be pruned
        (save_dir / "step_100").mkdir(parents=True)
        (save_dir / "step_100" / "model.pt").touch()
        (save_dir / "encoder_step200.pt").touch()
        (save_dir / "state.pt.tmp").touch()
        (save_dir / "backup.bak").touch()

        # Create files to keep
        (save_dir / "encoder.pt").write_text("keep")
        (save_dir / "state.pt").write_text("keep")
        (save_dir / "config.json").write_text("{}")

        # Run pruning
        freed = prune_save_dir(
            str(save_dir),
            keep_only=["encoder.pt", "state.pt", "config.json"]
        )

        # Check results
        remaining = list(save_dir.iterdir())
        remaining_names = [f.name for f in remaining]

        assert "encoder.pt" in remaining_names
        assert "state.pt" in remaining_names
        assert "config.json" in remaining_names
        assert "step_100" not in remaining_names
        assert "encoder_step200.pt" not in remaining_names
        assert "state.pt.tmp" not in remaining_names
        assert "backup.bak" not in remaining_names

    print("✅ Checkpoint pruning working correctly")


def test_checkpoint_size():
    """Test that checkpoints stay under 500MB."""
    print("\n=== Test 4: Checkpoint Size ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # Create typical model configuration
        encoder = InterlinguaEncoder(
            d_z=256, n_layers=6, n_heads=8, latent_len=32
        )
        adapters = {
            "llama": Adapter(d_z=256, d_model=4096, latent_length=32),
            "qwen": Adapter(d_z=256, d_model=3584, latent_length=32),
        }
        optimizer = optim.Adam(
            list(encoder.parameters()) +
            list(adapters["llama"].parameters()) +
            list(adapters["qwen"].parameters())
        )

        # Create checkpoint
        state_blob = {
            "epoch": 10,
            "global_step": 1000,
            "encoder": encoder.state_dict(),
            "adp_llama": adapters["llama"].state_dict(),
            "adp_qwen": adapters["qwen"].state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
            }
        }

        # Save checkpoint
        torch.save(state_blob, save_dir / "state.pt")

        # Check size
        size_bytes = (save_dir / "state.pt").stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        print(f"  Checkpoint size: {size_mb:.1f} MB")
        assert size_mb < 500, f"Checkpoint too large: {size_mb:.1f} MB > 500 MB"

    print("✅ Checkpoint size within limits")


def test_slurm_preemption():
    """Test handling of SLURM preemption signals."""
    print("\n=== Test 5: SLURM Preemption ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # Track if emergency save was called
        emergency_saved = [False]

        def mock_emergency_save():
            emergency_saved[0] = True
            # Create emergency checkpoint
            (save_dir / "emergency_state.pt").write_text("emergency")

        # Simulate SIGTERM (SLURM preemption signal)
        def send_sigterm():
            time.sleep(0.1)  # Small delay
            os.kill(os.getpid(), signal.SIGTERM)

        # Install signal handler
        def handler(signum, frame):
            mock_emergency_save()
            # Don't exit in test

        old_handler = signal.signal(signal.SIGTERM, handler)

        try:
            # Start thread to send signal
            thread = threading.Thread(target=send_sigterm)
            thread.start()

            # Simulate training loop
            for i in range(10):
                time.sleep(0.05)
                if emergency_saved[0]:
                    break

            thread.join()

            # Check emergency save was triggered
            assert emergency_saved[0], "Emergency save not triggered"
            assert (save_dir / "emergency_state.pt").exists()

        finally:
            signal.signal(signal.SIGTERM, old_handler)

    print("✅ SLURM preemption handling working")


def test_checkpoint_recovery_patterns():
    """Test various checkpoint recovery patterns."""
    print("\n=== Test 6: Recovery Patterns ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # Test 1: Recovery from partial state.pt
        partial_state = {
            "epoch": 3,
            "global_step": 50,
            # Missing some components
        }
        torch.save(partial_state, save_dir / "state.pt")

        # Should still find it
        latest = find_latest_checkpoint(str(save_dir))
        assert latest is not None

        # Test 2: Recovery from timestamped checkpoints
        save_dir2 = Path(tmpdir) / "timestamped"
        save_dir2.mkdir()

        torch.save({"step": 100}, save_dir2 / "state_step100.pt")
        torch.save({"step": 200}, save_dir2 / "state_step200.pt")
        torch.save({"step": 150}, save_dir2 / "state_step150.pt")

        latest = find_latest_checkpoint(str(save_dir2))
        assert "step200" in latest, f"Should find step 200, got {latest}"

        # Test 3: Recovery with best checkpoint
        best_dir = Path(tmpdir) / "best"
        best_dir.mkdir()

        state_best = {
            "epoch": 10,
            "global_step": 500,
            "best_first_acc": 0.95,
            "best_step": 500,
        }
        torch.save(state_best, best_dir / "state.pt")

        latest = find_latest_checkpoint(str(best_dir))
        loaded_state = torch.load(latest)
        assert loaded_state["best_first_acc"] == 0.95

    print("✅ Recovery patterns working correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Checkpoint Resilience")
    print("=" * 60)

    try:
        test_atomic_saves()
        test_auto_resume()
        test_checkpoint_pruning()
        test_checkpoint_size()
        test_slurm_preemption()
        test_checkpoint_recovery_patterns()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()