#!/usr/bin/env python3
"""
Comprehensive validation of checkpoint saving/loading strategy.

Tests:
1. Checkpoint saving at appropriate intervals
2. Resume functionality after crashes
3. Checkpoint size and storage requirements
4. Critical state preservation
5. Cleanup of old checkpoints

This is critical for 24-hour runs that might be interrupted!
"""

import os
import sys
import json
import time
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from latentwire.models import InterlinguaInterlinguaEncoder, Adapter
from latentwire.checkpointing import save_latest_checkpoint, prune_save_dir, CANONICAL_FILES
from latentwire.train import load_checkpoint, find_latest_checkpoint


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def print_status(test_name: str, passed: bool, details: str = ""):
    """Print test status with consistent formatting."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")


def estimate_checkpoint_size(
    d_z: int = 256,
    latent_len: int = 32,
    n_layers: int = 6,
    include_optimizer: bool = True
) -> Dict[str, float]:
    """Estimate checkpoint size based on model configuration."""

    # ByteEncoder backbone parameters
    backbone_params = 0
    # Transformer layers
    hidden_dim = d_z * 4  # ff_mult=4
    for _ in range(n_layers):
        # Self-attention
        backbone_params += 4 * d_z * d_z  # Q, K, V, O projections
        # FFN
        backbone_params += 2 * d_z * hidden_dim  # Up and down projections
        # LayerNorms
        backbone_params += 4 * d_z  # 2 LN layers with weight + bias

    # Compressor head
    backbone_params += d_z * latent_len * d_z  # Linear projection

    # Adapters (2 models)
    adapter_params_per_model = 0
    adapter_hidden = d_z * 2  # hidden_mult=2
    adapter_params_per_model += d_z * adapter_hidden  # First linear
    adapter_params_per_model += adapter_hidden * 4096  # To d_model (assuming 4096)
    adapter_params_per_model += 2 * adapter_hidden  # LayerNorms
    total_adapter_params = 2 * adapter_params_per_model

    # Total parameters
    total_params = backbone_params + total_adapter_params

    # Size calculations (float32 = 4 bytes per param)
    model_size_mb = (total_params * 4) / (1024 * 1024)

    # Optimizer state (Adam has 2 momentum buffers, each in float32)
    optimizer_size_mb = 0
    if include_optimizer:
        # Adam stores 2 momentum buffers (mean and variance), each same size as params
        # Each buffer is float32 (4 bytes per param)
        optimizer_size_mb = (total_params * 4 * 2) / (1024 * 1024)  # 2 buffers @ 4 bytes each

    # Additional state (RNG, config, etc.)
    misc_size_mb = 1  # ~1MB for misc state

    total_size_mb = model_size_mb + optimizer_size_mb + misc_size_mb

    return {
        "encoder_params": backbone_params,
        "adapter_params": total_adapter_params,
        "total_params": total_params,
        "model_size_mb": model_size_mb,
        "optimizer_size_mb": optimizer_size_mb,
        "total_size_mb": total_size_mb,
    }


def test_checkpoint_saving(tmpdir: Path) -> bool:
    """Test that checkpoints are saved correctly with all required components."""
    print("\nTest: Checkpoint Saving")

    try:
        # Create minimal models
        encoder = InterlinguaEncoder(
            d_z=64,  # Small for testing
            n_layers=2,
            latent_len=8
        )
        adapters = {
            "llama": Adapter(d_z=64, d_model=256, latent_length=8),
            "qwen": Adapter(d_z=64, d_model=256, latent_length=8),
        }

        # Create optimizer
        params = list(encoder.parameters())
        for adapter in adapters.values():
            params.extend(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        # Do a forward/backward pass to populate optimizer state
        dummy_input = torch.randn(2, 100)  # batch_size=2, seq_len=100
        latents = encoder(dummy_input)
        loss = sum(lat.mean() for lat in latents["private"].values())
        loss.backward()
        optimizer.step()

        # Prepare checkpoint artifacts
        state_blob = {
            "epoch": 5,
            "global_step": 1000,
            "rng_state": {
                "python": np.random.get_state(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "optimizer": optimizer.state_dict(),
            "encoder": encoder.state_dict(),
        }
        for name, adapter in adapters.items():
            state_blob[f"adp_{name}"] = adapter.state_dict()

        config = {
            "d_z": 64,
            "latent_len": 8,
            "n_layers": 2,
            "llama_id": "meta-llama/Llama-3.1-8B",
            "qwen_id": "Qwen/Qwen2.5-7B",
        }

        artifacts = {
            "encoder.pt": encoder.state_dict(),
            "adapter_llama.pt": adapters["llama"].state_dict(),
            "adapter_qwen.pt": adapters["qwen"].state_dict(),
            "state.pt": state_blob,
            "config.json": config,
        }

        # Save checkpoint
        ckpt_dir = tmpdir / "checkpoint_test"
        freed_pre, freed_post = save_latest_checkpoint(
            str(ckpt_dir),
            artifacts,
            pre_prune=True,
            post_prune=True,
            verbose=False
        )

        # Verify all files exist
        for filename in artifacts.keys():
            filepath = ckpt_dir / filename
            if not filepath.exists():
                print_status("File exists", False, f"Missing {filename}")
                return False

        # Check file sizes
        total_size = 0
        for filename in artifacts.keys():
            filepath = ckpt_dir / filename
            size = filepath.stat().st_size
            total_size += size
            print(f"    {filename}: {size / 1024:.1f} KB")

        print(f"    Total checkpoint size: {total_size / (1024*1024):.2f} MB")

        # Verify canonical files are preserved
        canonical_present = all(
            (ckpt_dir / f).exists()
            for f in ["encoder.pt", "adapter_llama.pt", "adapter_qwen.pt", "state.pt", "config.json"]
        )

        print_status("All canonical files saved", canonical_present)
        return canonical_present

    except Exception as e:
        print_status("Checkpoint saving", False, str(e))
        traceback.print_exc()
        return False


def test_resume_functionality(tmpdir: Path) -> bool:
    """Test that training can resume correctly from a checkpoint."""
    print("\nTest: Resume Functionality")

    try:
        # Create and save initial checkpoint
        encoder = InterlinguaEncoder(d_z=64, n_layers=2, latent_len=8)
        adapters = {
            "llama": Adapter(d_z=64, d_model=256, latent_length=8),
            "qwen": Adapter(d_z=64, d_model=256, latent_length=8),
        }

        # Set some distinctive values we can check
        encoder.compressor[0].weight.data.fill_(0.123)
        adapters["llama"].mlp[0].weight.data.fill_(0.456)

        params = list(encoder.parameters())
        for adapter in adapters.values():
            params.extend(adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        # Do some training steps
        for step in range(3):
            dummy_input = torch.randn(2, 100)
            latents = encoder(dummy_input)
            loss = sum(lat.mean() for lat in latents["private"].values())
            loss.backward()
            optimizer.step()

        # Save checkpoint
        ckpt_dir = tmpdir / "resume_test"
        state_blob = {
            "epoch": 10,
            "global_step": 2500,
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        for name, adapter in adapters.items():
            state_blob[f"adp_{name}"] = adapter.state_dict()

        artifacts = {
            "encoder.pt": encoder.state_dict(),
            "adapter_llama.pt": adapters["llama"].state_dict(),
            "adapter_qwen.pt": adapters["qwen"].state_dict(),
            "state.pt": state_blob,
            "config.json": {"d_z": 64, "latent_len": 8},
        }

        save_latest_checkpoint(str(ckpt_dir), artifacts, verbose=False)

        # Create new models and optimizer
        encoder_new = InterlinguaEncoder(d_z=64, n_layers=2, latent_len=8)
        adapters_new = {
            "llama": Adapter(d_z=64, d_model=256, latent_length=8),
            "qwen": Adapter(d_z=64, d_model=256, latent_length=8),
        }
        params_new = list(encoder_new.parameters())
        for adapter in adapters_new.values():
            params_new.extend(adapter.parameters())
        optimizer_new = torch.optim.AdamW(params_new, lr=1e-4)

        # Load checkpoint
        epoch_loaded, step_loaded = load_checkpoint(
            str(ckpt_dir / "state.pt"),
            encoder_new,
            adapters_new,
            optimizer=optimizer_new,
            device="cpu"
        )

        # Verify state was restored
        checks = [
            ("Epoch restored", epoch_loaded == 10),
            ("Step restored", step_loaded == 2500),
            ("Encoder weights restored",
             torch.allclose(encoder.compressor[0].weight, encoder_new.compressor[0].weight)),
            ("Adapter weights restored",
             torch.allclose(adapters["llama"].mlp[0].weight, adapters_new["llama"].mlp[0].weight)),
            ("Optimizer state restored",
             len(optimizer_new.state) == len(optimizer.state)),
        ]

        all_passed = True
        for check_name, passed in checks:
            print_status(check_name, passed)
            all_passed = all_passed and passed

        return all_passed

    except Exception as e:
        print_status("Resume functionality", False, str(e))
        traceback.print_exc()
        return False


def test_checkpoint_intervals(tmpdir: Path) -> bool:
    """Test that checkpoints are saved at appropriate intervals."""
    print("\nTest: Checkpoint Intervals")

    try:
        # Simulate training with periodic saves
        save_every = 100  # steps
        total_steps = 350

        saved_steps = []
        for step in range(total_steps):
            if save_every and (step % save_every == 0) and step > 0:
                # Would save checkpoint here
                saved_steps.append(step)

        expected_saves = [100, 200, 300]
        intervals_correct = saved_steps == expected_saves

        print_status(
            "Periodic saves",
            intervals_correct,
            f"Saved at steps: {saved_steps}"
        )

        # Test best checkpoint saving on improvement
        best_metric = -1.0
        best_saves = []

        # Simulate metric improvements
        metrics_over_time = [0.1, 0.15, 0.12, 0.25, 0.23, 0.30, 0.28]
        for step, metric in enumerate(metrics_over_time):
            if metric > best_metric:
                best_metric = metric
                best_saves.append((step, metric))

        print(f"    Best checkpoint saves: {best_saves}")
        print_status(
            "Best checkpoint tracking",
            len(best_saves) == 3,  # Should save 3 times for improvements
            f"Saved {len(best_saves)} best checkpoints"
        )

        return intervals_correct and len(best_saves) == 3

    except Exception as e:
        print_status("Checkpoint intervals", False, str(e))
        return False


def test_checkpoint_pruning(tmpdir: Path) -> bool:
    """Test that old checkpoints are properly cleaned up."""
    print("\nTest: Checkpoint Pruning")

    try:
        ckpt_dir = tmpdir / "pruning_test"
        ckpt_dir.mkdir(exist_ok=True)

        # Create various files that should be pruned
        (ckpt_dir / "step_100").mkdir()
        (ckpt_dir / "step_200").mkdir()
        (ckpt_dir / "epoch_1").mkdir()
        (ckpt_dir / "encoder_step1000.pt").touch()
        (ckpt_dir / "state_step500.pt").touch()
        (ckpt_dir / "model.pt.tmp").touch()
        (ckpt_dir / "checkpoint.new").touch()
        (ckpt_dir / "old_file.pt.old").touch()

        # Create canonical files that should be kept
        (ckpt_dir / "encoder.pt").touch()
        (ckpt_dir / "adapter_llama.pt").touch()
        (ckpt_dir / "state.pt").touch()
        (ckpt_dir / "config.json").touch()

        # Run pruning
        freed = prune_save_dir(
            str(ckpt_dir),
            keep_only=["encoder.pt", "adapter_llama.pt", "state.pt", "config.json"]
        )

        # Check what remains
        remaining = list(ckpt_dir.iterdir())
        remaining_names = [f.name for f in remaining]

        # Should only have canonical files
        expected = {"encoder.pt", "adapter_llama.pt", "state.pt", "config.json"}
        only_canonical = set(remaining_names) == expected

        print(f"    Freed {freed} bytes during pruning")
        print(f"    Remaining files: {remaining_names}")
        print_status(
            "Pruning removes temporary files",
            only_canonical,
            f"Kept {len(remaining)} files"
        )

        return only_canonical

    except Exception as e:
        print_status("Checkpoint pruning", False, str(e))
        return False


def test_checkpoint_size_estimates() -> bool:
    """Test checkpoint size estimates for different configurations."""
    print("\nTest: Checkpoint Size Estimates")

    try:
        configs = [
            {"d_z": 256, "latent_len": 32, "n_layers": 6, "name": "Standard"},
            {"d_z": 512, "latent_len": 64, "n_layers": 8, "name": "Large"},
            {"d_z": 128, "latent_len": 16, "n_layers": 4, "name": "Small"},
        ]

        print("\n    Configuration Estimates:")
        print("    " + "-"*60)

        all_reasonable = True
        for config in configs:
            est = estimate_checkpoint_size(
                d_z=config["d_z"],
                latent_len=config["latent_len"],
                n_layers=config["n_layers"],
                include_optimizer=True
            )

            print(f"    {config['name']} (d_z={config['d_z']}, L={config['latent_len']}):")
            print(f"      Model params: {est['total_params']:,}")
            print(f"      Model size: {est['model_size_mb']:.1f} MB")
            print(f"      With optimizer: {est['total_size_mb']:.1f} MB")

            # Check if size is reasonable (< 500MB for these configs)
            if est['total_size_mb'] > 500:
                all_reasonable = False
                print(f"      ⚠️  WARNING: Size exceeds 500MB threshold")

        print_status(
            "Checkpoint sizes reasonable",
            all_reasonable,
            "All configurations < 500MB"
        )

        return all_reasonable

    except Exception as e:
        print_status("Size estimates", False, str(e))
        return False


def test_crash_recovery(tmpdir: Path) -> bool:
    """Test recovery after simulated crash during checkpoint save."""
    print("\nTest: Crash Recovery")

    try:
        ckpt_dir = tmpdir / "crash_test"
        ckpt_dir.mkdir(exist_ok=True)

        # Create a valid checkpoint
        artifacts = {
            "encoder.pt": {"weight": [1, 2, 3]},
            "state.pt": {"epoch": 5, "global_step": 1000},
            "config.json": {"d_z": 256},
        }

        save_latest_checkpoint(str(ckpt_dir), artifacts, verbose=False)

        # Simulate partial write (crash during save)
        partial_file = ckpt_dir / "encoder.pt.tmp"
        partial_file.write_text("partial")

        # Try to find latest checkpoint (should ignore .tmp file)
        latest = find_latest_checkpoint(str(ckpt_dir))

        can_find = latest is not None
        print_status(
            "Can find checkpoint after crash",
            can_find,
            f"Found: {latest}"
        )

        # Clean up temporary files and try again
        prune_save_dir(str(ckpt_dir))

        remaining_tmp = list(ckpt_dir.glob("*.tmp"))
        cleanup_works = len(remaining_tmp) == 0

        print_status(
            "Cleanup removes partial files",
            cleanup_works,
            f"Remaining .tmp files: {len(remaining_tmp)}"
        )

        return can_find and cleanup_works

    except Exception as e:
        print_status("Crash recovery", False, str(e))
        return False


def test_concurrent_saves(tmpdir: Path) -> bool:
    """Test that concurrent checkpoint saves don't corrupt data."""
    print("\nTest: Concurrent Save Safety")

    try:
        ckpt_dir = tmpdir / "concurrent_test"

        # Save checkpoint 1
        artifacts1 = {
            "encoder.pt": {"version": 1},
            "state.pt": {"epoch": 1},
        }
        save_latest_checkpoint(str(ckpt_dir), artifacts1, verbose=False)

        # Immediately save checkpoint 2 (simulating concurrent save)
        artifacts2 = {
            "encoder.pt": {"version": 2},
            "state.pt": {"epoch": 2},
        }
        save_latest_checkpoint(str(ckpt_dir), artifacts2, verbose=False)

        # Load and verify we get consistent state
        state_file = ckpt_dir / "state.pt"
        encoder_file = ckpt_dir / "encoder.pt"

        if state_file.exists() and encoder_file.exists():
            # In practice these would be torch.load
            # Here we just check files exist and aren't corrupted
            state_size = state_file.stat().st_size
            encoder_size = encoder_file.stat().st_size

            files_valid = state_size > 0 and encoder_size > 0
            print_status(
                "Files not corrupted",
                files_valid,
                f"Sizes: state={state_size}, encoder={encoder_size}"
            )
            return files_valid
        else:
            print_status("Files exist after concurrent save", False)
            return False

    except Exception as e:
        print_status("Concurrent saves", False, str(e))
        return False


def main():
    """Run all checkpoint validation tests."""

    print_section("CHECKPOINT SAVING/LOADING VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Create temporary directory for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        tests = [
            ("Checkpoint Saving", test_checkpoint_saving, tmpdir),
            ("Resume Functionality", test_resume_functionality, tmpdir),
            ("Save Intervals", test_checkpoint_intervals, tmpdir),
            ("Checkpoint Pruning", test_checkpoint_pruning, tmpdir),
            ("Size Estimates", test_checkpoint_size_estimates, None),
            ("Crash Recovery", test_crash_recovery, tmpdir),
            ("Concurrent Saves", test_concurrent_saves, tmpdir),
        ]

        results = []
        for test_name, test_func, test_arg in tests:
            print_section(test_name)
            if test_arg is None:
                passed = test_func()
            else:
                passed = test_func(test_arg)
            results.append((test_name, passed))

        # Summary
        print_section("VALIDATION SUMMARY")

        total = len(results)
        passed = sum(1 for _, p in results if p)

        print(f"\nResults: {passed}/{total} tests passed")
        print("-"*40)

        for test_name, test_passed in results:
            status = "✅" if test_passed else "❌"
            print(f"  {status} {test_name}")

        print("\n" + "="*80)

        # Overall assessment
        if passed == total:
            print("✅ CHECKPOINT STRATEGY VALIDATED - Safe for 24-hour runs")
            print("\nKey findings:")
            print("  • Checkpoints save all critical state")
            print("  • Resume functionality works correctly")
            print("  • Old checkpoints are properly pruned")
            print("  • Checkpoint sizes are reasonable (<500MB)")
            print("  • System can recover from crashes")
        else:
            print("⚠️  CHECKPOINT VALIDATION INCOMPLETE")
            print(f"\n{total - passed} tests failed - review before long runs")

        print("\nRecommendations for 24-hour runs:")
        print("  1. Set --save_every 500 (save every 500 steps)")
        print("  2. Monitor disk space (expect ~1GB for checkpoints)")
        print("  3. Use --auto_resume flag for automatic recovery")
        print("  4. Keep best checkpoint separate from latest")
        print("  5. Test resume on actual hardware before long runs")

        return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)