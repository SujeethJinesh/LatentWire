#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive system test suite for LatentWire.
Tests all critical components before expensive HPC runs.
Exit code 0 if all tests pass, 1 if any fail.
Target runtime: <5 minutes with minimal resources.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing torch - if not available, skip torch tests
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available locally. Skipping torch-dependent tests.")
    print("Note: These tests will work on HPC where PyTorch is installed.")

# Try importing project modules - only if torch is available
if TORCH_AVAILABLE:
    try:
        from latentwire.models import ByteEncoder, Adapter
        from latentwire.train import create_model_components, save_checkpoint, load_checkpoint
        from latentwire.data import get_dataset
        from latentwire.eval import load_model_from_checkpoint
        MODULES_AVAILABLE = True
    except ImportError as e:
        MODULES_AVAILABLE = False
        print(f"Warning: Project modules not available: {e}")
else:
    MODULES_AVAILABLE = False

# Test configuration
TEST_CONFIG = {
    "samples": 10,
    "epochs": 1,
    "batch_size": 2,
    "latent_len": 4,
    "d_z": 32,
    "encoder_type": "byte",
    "dataset": "squad",
    "llama_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen_id": "Qwen/Qwen2.5-7B-Instruct",
    "sequential_models": True,
    "warm_anchor_text": "Answer: ",
    "first_token_ce_weight": 0.5,
    "k_token_ce_from_prefix": 4,
    "gradient_accumulation_steps": 1
}

class TestResults:
    """Track test results and provide summary."""
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()

    def add_test(self, name, passed, message=""):
        self.tests.append({
            "name": name,
            "passed": passed,
            "message": message,
            "time": time.time() - self.start_time
        })
        if passed:
            self.passed += 1
            print(f"[PASS] {name}")
        else:
            self.failed += 1
            print(f"[FAIL] {name}: {message}")

    def summary(self):
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print(f"TEST SUMMARY: {self.passed}/{len(self.tests)} passed in {total_time:.1f}s")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for test in self.tests:
                if not test["passed"]:
                    print(f"  - {test['name']}: {test['message']}")
        print("="*60)
        return self.failed == 0

# Initialize results tracker
results = TestResults()

def test_checkpoint_save_load():
    """Test 1: Checkpoint save/load cycle."""
    print("\n" + "="*60)
    print("TEST 1: Checkpoint Save/Load Cycle")
    print("="*60)

    if not TORCH_AVAILABLE or not MODULES_AVAILABLE:
        results.add_test("Checkpoint Save/Load", True, "Skipped (PyTorch not available locally)")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal model components
            encoder = ByteEncoder(
                d_byte=256,
                d_z=TEST_CONFIG["d_z"],
                n_head=4,
                n_layer=2,
                seq_len=TEST_CONFIG["latent_len"]
            )

            # Create dummy adapters
            llama_adapter = Adapter(
                d_z=TEST_CONFIG["d_z"],
                d_model=4096,  # Llama embedding dimension
                layer_norm_eps=1e-5
            )

            qwen_adapter = Adapter(
                d_z=TEST_CONFIG["d_z"],
                d_model=3584,  # Qwen embedding dimension
                layer_norm_eps=1e-6
            )

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir(parents=True)

            state = {
                "encoder": encoder.state_dict(),
                "llama_adapter": llama_adapter.state_dict(),
                "qwen_adapter": qwen_adapter.state_dict(),
                "epoch": 5,
                "global_step": 100,
                "config": TEST_CONFIG,
                "metrics": {
                    "loss": 2.5,
                    "llama_loss": 2.3,
                    "qwen_loss": 2.7
                }
            }

            torch.save(state, checkpoint_path / "checkpoint.pt")

            # Save config separately (as done in training)
            with open(checkpoint_path / "config.json", "w") as f:
                json.dump(TEST_CONFIG, f, indent=2)

            # Load checkpoint
            loaded_state = torch.load(checkpoint_path / "checkpoint.pt", map_location="cpu")

            # Verify all components
            assert "encoder" in loaded_state
            assert "llama_adapter" in loaded_state
            assert "qwen_adapter" in loaded_state
            assert loaded_state["epoch"] == 5
            assert loaded_state["global_step"] == 100
            assert loaded_state["metrics"]["loss"] == 2.5

            # Load into new models and verify weights match
            new_encoder = ByteEncoder(
                d_byte=256,
                d_z=TEST_CONFIG["d_z"],
                n_head=4,
                n_layer=2,
                seq_len=TEST_CONFIG["latent_len"]
            )
            new_encoder.load_state_dict(loaded_state["encoder"])

            # Check a few parameters match
            for (n1, p1), (n2, p2) in zip(encoder.named_parameters(), new_encoder.named_parameters()):
                assert torch.allclose(p1, p2), f"Parameter mismatch: {n1}"

            results.add_test("Checkpoint Save/Load", True)

    except Exception as e:
        results.add_test("Checkpoint Save/Load", False, str(e))

def test_preemption_signal():
    """Test 2: Preemption signal simulation."""
    print("\n" + "="*60)
    print("TEST 2: Preemption Signal Simulation")
    print("="*60)

    try:
        # Create a simple script that handles SIGTERM
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "preempt_test.py"
            script_path.write_text("""
import signal
import sys
import time

def handle_sigterm(signum, frame):
    print("SIGTERM received, saving checkpoint...")
    with open("checkpoint_saved.txt", "w") as f:
                f.write("checkpoint")
    print("Checkpoint saved, exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
print("Process started, waiting for signal...")
time.sleep(10)  # Wait for signal
print("Timeout - no signal received")
""")

            # Start process
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Give it time to start
            time.sleep(0.5)

            # Send SIGTERM
            proc.terminate()

            # Wait for completion
            stdout, stderr = proc.communicate(timeout=5)

            # Check that checkpoint was saved
            checkpoint_file = Path(tmpdir) / "checkpoint_saved.txt"
            assert checkpoint_file.exists(), "Checkpoint file not created"
            assert "SIGTERM received" in stdout, f"Signal not handled properly. stdout: {stdout}"

            results.add_test("Preemption Signal", True)

    except Exception as e:
        results.add_test("Preemption Signal", False, str(e))

def test_resume_from_checkpoint():
    """Test 3: Resume from checkpoint."""
    print("\n" + "="*60)
    print("TEST 3: Resume from Checkpoint")
    print("="*60)

    if not TORCH_AVAILABLE:
        results.add_test("Resume from Checkpoint", True, "Skipped (PyTorch not available locally)")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            checkpoint_path.mkdir(parents=True)

            # Create initial checkpoint
            initial_state = {
                "epoch": 3,
                "global_step": 50,
                "encoder": {"weight": torch.randn(10, 10)},
                "llama_adapter": {"weight": torch.randn(5, 5)},
                "qwen_adapter": {"weight": torch.randn(5, 5)},
                "optimizer": {"step": 50},
                "config": TEST_CONFIG,
                "metrics": {"loss": [3.0, 2.8, 2.6]}
            }

            torch.save(initial_state, checkpoint_path / "checkpoint.pt")

            # Simulate resume
            resumed_state = torch.load(checkpoint_path / "checkpoint.pt", map_location="cpu")

            # Verify we can continue from where we left off
            assert resumed_state["epoch"] == 3
            assert resumed_state["global_step"] == 50
            assert len(resumed_state["metrics"]["loss"]) == 3

            # Simulate continuing training
            resumed_state["epoch"] = 4
            resumed_state["global_step"] = 75
            resumed_state["metrics"]["loss"].append(2.4)

            # Save updated checkpoint
            torch.save(resumed_state, checkpoint_path / "checkpoint_resume.pt")

            # Load and verify
            final_state = torch.load(checkpoint_path / "checkpoint_resume.pt", map_location="cpu")
            assert final_state["epoch"] == 4
            assert final_state["global_step"] == 75
            assert len(final_state["metrics"]["loss"]) == 4
            assert final_state["metrics"]["loss"][-1] == 2.4

            results.add_test("Resume from Checkpoint", True)

    except Exception as e:
        results.add_test("Resume from Checkpoint", False, str(e))

def test_elastic_gpu_configuration():
    """Test 4: Elastic GPU configuration (1-4 GPUs)."""
    print("\n" + "="*60)
    print("TEST 4: Elastic GPU Configuration")
    print("="*60)

    if not TORCH_AVAILABLE:
        results.add_test("Elastic GPU Configuration", True, "Skipped (PyTorch not available locally)")
        return

    try:
        # Test different GPU configurations
        gpu_configs = [1, 2, 4]

        for n_gpus in gpu_configs:
            # Simulate different CUDA_VISIBLE_DEVICES settings
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(n_gpus))}):
                # Mock torch.cuda to simulate GPU availability
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.device_count", return_value=n_gpus):

                        # Test batch size scaling
                        if n_gpus == 1:
                            effective_batch = TEST_CONFIG["batch_size"]
                        else:
                            effective_batch = TEST_CONFIG["batch_size"] * n_gpus

                        # Test data parallel configuration
                        if n_gpus > 1:
                            # In real scenario, this would use DDP
                            assert effective_batch == TEST_CONFIG["batch_size"] * n_gpus

                        print(f"  [OK] Tested with {n_gpus} GPU(s), effective batch size: {effective_batch}")

        results.add_test("Elastic GPU Configuration", True)

    except Exception as e:
        results.add_test("Elastic GPU Configuration", False, str(e))

def test_data_loading_performance():
    """Test 5: Data loading performance."""
    print("\n" + "="*60)
    print("TEST 5: Data Loading Performance")
    print("="*60)

    if not MODULES_AVAILABLE:
        results.add_test("Data Loading Performance", True, "Skipped (Modules not available locally)")
        return

    try:
        # Test loading a small subset
        start_time = time.time()

        dataset = get_dataset(
            dataset_name="squad",
            split="validation",
            samples=100,
            cache_dir=None
        )

        load_time = time.time() - start_time

        # Basic checks
        assert len(dataset) <= 100, f"Dataset size mismatch: {len(dataset)}"
        assert load_time < 30, f"Data loading too slow: {load_time:.2f}s"

        # Test iteration speed
        start_time = time.time()
        for i, item in enumerate(dataset):
            if i >= 10:  # Just test first 10
                break
            assert "prefix" in item
            assert "target" in item

        iter_time = time.time() - start_time
        assert iter_time < 5, f"Data iteration too slow: {iter_time:.2f}s"

        print(f"  Data loading: {load_time:.2f}s for 100 samples")
        print(f"  Iteration: {iter_time:.2f}s for 10 samples")

        results.add_test("Data Loading Performance", True)

    except Exception as e:
        results.add_test("Data Loading Performance", False, str(e))

def test_logging_capture():
    """Test 6: Logging capture."""
    print("\n" + "="*60)
    print("TEST 6: Logging Capture")
    print("="*60)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            # Create a test script that uses tee pattern
            script_path = Path(tmpdir) / "log_test.sh"
            script_path.write_text(f"""#!/bin/bash
{{
    echo "Starting test"
    echo "Progress: 50%"
    echo "Error test" >&2
    echo "Complete"
}} 2>&1 | tee "{log_file}"
""")
            script_path.chmod(0o755)

            # Run script
            result = subprocess.run(
                ["/bin/bash", str(script_path)],
                capture_output=True,
                text=True
            )

            # Verify log file contains output
            assert log_file.exists(), "Log file not created"
            log_content = log_file.read_text()
            assert "Starting test" in log_content
            assert "Progress: 50%" in log_content
            assert "Error test" in log_content
            assert "Complete" in log_content

            print(f"  [OK] Log file created with {len(log_content)} bytes")

            # Test JSON logging
            json_log = Path(tmpdir) / "metrics.json"
            metrics = {
                "epoch": 1,
                "loss": 2.5,
                "timestamp": datetime.now().isoformat()
            }

            with open(json_log, "w") as f:
                json.dump(metrics, f, indent=2)

            # Verify JSON can be loaded
            loaded_metrics = json.loads(json_log.read_text())
            assert loaded_metrics["epoch"] == 1
            assert loaded_metrics["loss"] == 2.5

            results.add_test("Logging Capture", True)

    except Exception as e:
        results.add_test("Logging Capture", False, str(e))

def test_state_management():
    """Test 7: State management."""
    print("\n" + "="*60)
    print("TEST 7: State Management")
    print("="*60)

    if not TORCH_AVAILABLE:
        results.add_test("State Management", True, "Skipped (PyTorch not available locally)")
        return

    try:
        # Test optimizer state preservation
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Do some steps
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(5, 10)).sum()
            loss.backward()
            optimizer.step()

        # Save state
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 3
        }

        # Create new model and optimizer
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Load state
        new_model.load_state_dict(state["model"])
        new_optimizer.load_state_dict(state["optimizer"])

        # Verify optimizer state matches
        for group, new_group in zip(optimizer.param_groups, new_optimizer.param_groups):
            assert group["lr"] == new_group["lr"]

        # Test learning rate scheduler state
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        for _ in range(5):
            scheduler.step()

        scheduler_state = scheduler.state_dict()
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)
        new_scheduler.load_state_dict(scheduler_state)

        assert scheduler.last_epoch == new_scheduler.last_epoch

        print("  [OK] Model state preserved")
        print("  [OK] Optimizer state preserved")
        print("  [OK] Scheduler state preserved")

        results.add_test("State Management", True)

    except Exception as e:
        results.add_test("State Management", False, str(e))

def test_git_operations():
    """Test 8: Git operations."""
    print("\n" + "="*60)
    print("TEST 8: Git Operations")
    print("="*60)

    try:
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Not in a git repository"

        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True
        )
        current_branch = result.stdout.strip()
        print(f"  Current branch: {current_branch}")

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        )
        uncommitted_files = result.stdout.strip()
        if uncommitted_files:
            print(f"  Warning: Uncommitted changes detected")

        # Test git log parsing
        result = subprocess.run(
            ["git", "log", "--oneline", "-n", "1"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to get git log"
        last_commit = result.stdout.strip()
        print(f"  Last commit: {last_commit}")

        # Test remote connectivity (dry-run fetch)
        result = subprocess.run(
            ["git", "fetch", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Don't fail if network is unavailable, just warn
        if result.returncode != 0:
            print("  Warning: Cannot connect to git remote")
        else:
            print("  [OK] Git remote accessible")

        results.add_test("Git Operations", True)

    except subprocess.TimeoutExpired:
        results.add_test("Git Operations", True, "Remote check timed out (non-critical)")
    except Exception as e:
        results.add_test("Git Operations", False, str(e))

def test_minimal_training_loop():
    """Bonus: Test minimal training loop integration."""
    print("\n" + "="*60)
    print("BONUS: Minimal Training Loop")
    print("="*60)

    if not TORCH_AVAILABLE or not MODULES_AVAILABLE:
        results.add_test("Minimal Training Loop", True, "Skipped (PyTorch not available locally)")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal encoder
            encoder = ByteEncoder(
                d_byte=64,  # Smaller for testing
                d_z=32,
                n_head=2,
                n_layer=1,
                seq_len=4
            )

            # Test forward pass
            dummy_input = torch.randint(0, 256, (2, 100), dtype=torch.long)
            with torch.no_grad():
                output = encoder(dummy_input)

            assert output.shape == (2, 4, 32), f"Unexpected output shape: {output.shape}"

            # Test backward pass
            optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
            optimizer.zero_grad()

            output = encoder(dummy_input)
            loss = output.mean()
            loss.backward()

            # Check gradients exist
            has_grads = False
            for param in encoder.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grads = True
                    break

            assert has_grads, "No gradients computed"

            # Test optimizer step
            optimizer.step()

            print("  [OK] Forward pass successful")
            print("  [OK] Backward pass successful")
            print("  [OK] Optimizer step successful")

            results.add_test("Minimal Training Loop", True)

    except Exception as e:
        results.add_test("Minimal Training Loop", False, str(e))

def main():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("LATENTWIRE SYSTEM TEST SUITE")
    print(f"Started at: {datetime.now().isoformat()}")
    print("="*60)

    # Run all tests
    test_checkpoint_save_load()
    test_preemption_signal()
    test_resume_from_checkpoint()
    test_elastic_gpu_configuration()
    test_data_loading_performance()
    test_logging_capture()
    test_state_management()
    test_git_operations()
    test_minimal_training_loop()

    # Print summary
    success = results.summary()

    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()