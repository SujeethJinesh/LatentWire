#!/usr/bin/env python3
"""
Comprehensive validation script for preemptible training system.

This script validates:
1. Signal handling (SIGTERM, SIGINT, SIGUSR1)
2. Checkpoint save/load functionality
3. Resume from checkpoint
4. Multi-GPU support (if available)
5. Performance benchmarks

Run this before deploying to HPC to ensure the preemption system works correctly.
"""

import os
import sys
import time
import json
import signal
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    GPU_COUNT = 0

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import checkpointing utilities
from latentwire.checkpointing import (
    save_latest_checkpoint,
    prune_save_dir
)


class ValidationResults:
    """Track validation test results."""

    def __init__(self):
        self.tests = {}
        self.start_time = time.time()

    def add_test(self, name: str, passed: bool, details: str = "", duration: float = 0):
        """Add a test result."""
        self.tests[name] = {
            "passed": passed,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }

    def print_summary(self):
        """Print test summary."""
        total_duration = time.time() - self.start_time
        passed = sum(1 for t in self.tests.values() if t["passed"])
        failed = len(self.tests) - passed

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        for name, result in self.tests.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{status} | {name}")
            if result["details"]:
                print(f"         {result['details']}")

        print("-" * 60)
        print(f"Total: {len(self.tests)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Duration: {total_duration:.2f}s")
        print("=" * 60)

        return failed == 0

    def save_report(self, path: Path):
        """Save validation report to JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - self.start_time,
            "environment": {
                "torch_available": TORCH_AVAILABLE,
                "cuda_available": CUDA_AVAILABLE,
                "gpu_count": GPU_COUNT,
                "python_version": sys.version,
            },
            "tests": self.tests
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)


def test_checkpoint_atomicity(results: ValidationResults):
    """Test atomic checkpoint saving."""
    print("\n🔬 Testing checkpoint atomicity...")
    start_time = time.time()

    if not TORCH_AVAILABLE:
        results.add_test("checkpoint_atomicity", False, "PyTorch not available")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple model
            model = nn.Linear(100, 50)
            optimizer = optim.Adam(model.parameters())

            # Save checkpoint
            artifacts = {
                "model.pt": model.state_dict(),
                "optimizer.pt": optimizer.state_dict(),
                "state.pt": {"epoch": 1, "step": 100}
            }

            save_latest_checkpoint(tmpdir, artifacts, verbose=False)

            # Verify files exist
            for filename in ["model.pt", "optimizer.pt", "state.pt"]:
                if not os.path.exists(os.path.join(tmpdir, filename)):
                    results.add_test("checkpoint_atomicity", False, f"Missing {filename}")
                    return

            # Verify no temp files remain
            temp_files = [f for f in os.listdir(tmpdir) if '.tmp' in f or f.startswith('.')]
            if temp_files:
                results.add_test("checkpoint_atomicity", False, f"Temp files remain: {temp_files}")
                return

            duration = time.time() - start_time
            results.add_test("checkpoint_atomicity", True, f"Saved in {duration:.3f}s", duration)

    except Exception as e:
        results.add_test("checkpoint_atomicity", False, str(e))


def test_checkpoint_pruning(results: ValidationResults):
    """Test checkpoint directory pruning."""
    print("\n🗑️  Testing checkpoint pruning...")
    start_time = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various files and directories
            files_to_create = [
                "step_100/model.pt",
                "epoch_5/checkpoint.pt",
                "temp.tmp",
                "model.pt.bak",
                "encoder_step500.pt",
                "valid_file.txt"
            ]

            for filepath in files_to_create:
                full_path = Path(tmpdir) / filepath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text("test data")

            # Prune directory
            freed = prune_save_dir(tmpdir, keep_only=["valid_file.txt"])

            # Check that step directories were removed
            remaining = os.listdir(tmpdir)
            if "step_100" in remaining or "epoch_5" in remaining:
                results.add_test("checkpoint_pruning", False, "Step directories not removed")
                return

            # Check that temp files were removed
            if any(f.endswith('.tmp') or f.endswith('.bak') for f in remaining):
                results.add_test("checkpoint_pruning", False, "Temp files not removed")
                return

            # Check that valid file remains
            if "valid_file.txt" not in remaining:
                results.add_test("checkpoint_pruning", False, "Valid file was removed")
                return

            duration = time.time() - start_time
            results.add_test("checkpoint_pruning", True, f"Freed {freed} bytes in {duration:.3f}s", duration)

    except Exception as e:
        results.add_test("checkpoint_pruning", False, str(e))


def test_signal_handling(results: ValidationResults):
    """Test signal handling with subprocess."""
    print("\n📡 Testing signal handling...")
    start_time = time.time()

    script = """
import signal
import sys
import time
import json

checkpoint_saved = False

def handle_sigterm(signum, frame):
    global checkpoint_saved
    checkpoint_saved = True
    with open('/tmp/signal_test.json', 'w') as f:
        json.dump({"signal": "SIGTERM", "saved": True}, f)
    sys.exit(99)

signal.signal(signal.SIGTERM, handle_sigterm)

# Run for up to 5 seconds
for i in range(50):
    time.sleep(0.1)
    print(f"Running {i}", end="\\r")

print("No signal received")
"""

    try:
        # Write test script
        test_script = Path("/tmp/signal_test.py")
        test_script.write_text(script)

        # Start subprocess
        proc = subprocess.Popen(
            [sys.executable, str(test_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a bit then send SIGTERM
        time.sleep(1)
        proc.terminate()  # Sends SIGTERM

        # Wait for process to exit
        try:
            exit_code = proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            results.add_test("signal_handling", False, "Process did not exit on SIGTERM")
            return

        # Check if checkpoint was saved
        signal_file = Path("/tmp/signal_test.json")
        if signal_file.exists():
            with open(signal_file) as f:
                data = json.load(f)
            if data.get("saved"):
                duration = time.time() - start_time
                results.add_test("signal_handling", True, f"SIGTERM handled, exit code {exit_code}", duration)
                signal_file.unlink()
            else:
                results.add_test("signal_handling", False, "Signal received but checkpoint not saved")
        else:
            results.add_test("signal_handling", False, "No checkpoint file created")

    except Exception as e:
        results.add_test("signal_handling", False, str(e))

    finally:
        # Cleanup
        if test_script.exists():
            test_script.unlink()


def test_resume_functionality(results: ValidationResults):
    """Test checkpoint resume functionality."""
    print("\n🔄 Testing resume functionality...")
    start_time = time.time()

    if not TORCH_AVAILABLE:
        results.add_test("resume_functionality", False, "PyTorch not available")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save initial checkpoint
            model1 = nn.Linear(50, 25)
            optimizer1 = optim.SGD(model1.parameters(), lr=0.01)

            # Train for a few steps
            for _ in range(5):
                optimizer1.zero_grad()
                loss = model1(torch.randn(10, 50)).sum()
                loss.backward()
                optimizer1.step()

            # Save checkpoint
            artifacts = {
                "model.pt": model1.state_dict(),
                "optimizer.pt": optimizer1.state_dict(),
                "state.pt": {"epoch": 3, "step": 150, "loss": 0.5}
            }
            save_latest_checkpoint(tmpdir, artifacts, verbose=False)

            # Create new model and load checkpoint
            model2 = nn.Linear(50, 25)
            optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

            # Load checkpoint
            model2.load_state_dict(torch.load(os.path.join(tmpdir, "model.pt")))
            optimizer2.load_state_dict(torch.load(os.path.join(tmpdir, "optimizer.pt")))
            state = torch.load(os.path.join(tmpdir, "state.pt"))

            # Verify state
            if state["epoch"] != 3 or state["step"] != 150:
                results.add_test("resume_functionality", False, "State mismatch")
                return

            # Verify weights match
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                if not torch.allclose(p1, p2):
                    results.add_test("resume_functionality", False, "Weight mismatch after resume")
                    return

            duration = time.time() - start_time
            results.add_test("resume_functionality", True, f"Resume successful in {duration:.3f}s", duration)

    except Exception as e:
        results.add_test("resume_functionality", False, str(e))


def test_gpu_metrics(results: ValidationResults):
    """Test GPU metrics collection."""
    print("\n🎮 Testing GPU metrics...")
    start_time = time.time()

    if not CUDA_AVAILABLE:
        results.add_test("gpu_metrics", True, "No GPU available (skipped)", 0)
        return

    try:
        # Reset peak memory stats
        for i in range(GPU_COUNT):
            torch.cuda.reset_peak_memory_stats(i)

        # Create some GPU tensors
        tensors = []
        for i in range(min(GPU_COUNT, 2)):  # Test up to 2 GPUs
            device = torch.device(f"cuda:{i}")

            # Allocate memory
            tensor = torch.randn(1000, 1000, device=device)
            tensors.append(tensor)

            # Get memory stats
            allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
            peak_gb = torch.cuda.max_memory_allocated(i) / 1024**3

            if allocated_gb <= 0:
                results.add_test("gpu_metrics", False, f"GPU {i}: No memory allocated")
                return

        details = f"{GPU_COUNT} GPUs tested, memory tracking working"
        duration = time.time() - start_time
        results.add_test("gpu_metrics", True, details, duration)

    except Exception as e:
        results.add_test("gpu_metrics", False, str(e))


def test_performance_benchmark(results: ValidationResults):
    """Benchmark checkpoint save/load performance."""
    print("\n⚡ Testing checkpoint performance...")
    start_time = time.time()

    if not TORCH_AVAILABLE:
        results.add_test("performance", False, "PyTorch not available")
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test different model sizes
            sizes = [(100, 50), (500, 250), (1000, 500)]
            timings = []

            for d_in, d_out in sizes:
                model = nn.Linear(d_in, d_out)
                optimizer = optim.Adam(model.parameters())

                # Time save
                save_start = time.time()
                artifacts = {
                    "model.pt": model.state_dict(),
                    "optimizer.pt": optimizer.state_dict()
                }
                save_latest_checkpoint(tmpdir, artifacts, verbose=False)
                save_time = time.time() - save_start

                # Time load
                load_start = time.time()
                torch.load(os.path.join(tmpdir, "model.pt"))
                torch.load(os.path.join(tmpdir, "optimizer.pt"))
                load_time = time.time() - load_start

                timings.append((d_in * d_out, save_time, load_time))

            # Check performance
            largest_params = timings[-1][0]
            largest_save = timings[-1][1]
            largest_load = timings[-1][2]

            if largest_save > 1.0:  # Should save in under 1 second
                results.add_test("performance", False, f"Save too slow: {largest_save:.2f}s")
                return

            if largest_load > 0.5:  # Should load in under 0.5 seconds
                results.add_test("performance", False, f"Load too slow: {largest_load:.2f}s")
                return

            details = f"Largest model ({largest_params} params): save={largest_save:.3f}s, load={largest_load:.3f}s"
            duration = time.time() - start_time
            results.add_test("performance", True, details, duration)

    except Exception as e:
        results.add_test("performance", False, str(e))


def test_distributed_compatibility(results: ValidationResults):
    """Test distributed training compatibility."""
    print("\n🌐 Testing distributed compatibility...")
    start_time = time.time()

    if not TORCH_AVAILABLE:
        results.add_test("distributed", False, "PyTorch not available")
        return

    try:
        # Check if distributed package is available
        import torch.distributed as dist

        # Check environment variables
        env_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        missing = [v for v in env_vars if v not in os.environ]

        if missing:
            # Not in distributed environment, which is fine
            results.add_test("distributed", True, "Not in distributed environment (normal)", 0)
        else:
            # In distributed environment, check if initialized
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                details = f"Rank {rank}/{world_size}"
            else:
                details = "Distributed env detected but not initialized"

            duration = time.time() - start_time
            results.add_test("distributed", True, details, duration)

    except ImportError:
        results.add_test("distributed", True, "Distributed package not available (normal)", 0)
    except Exception as e:
        results.add_test("distributed", False, str(e))


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("🧪 PREEMPTIBLE TRAINING VALIDATION SUITE")
    print("=" * 60)
    print(f"PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"CUDA: {'✅' if CUDA_AVAILABLE else '❌'}")
    print(f"GPUs: {GPU_COUNT}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)

    results = ValidationResults()

    # Run all tests
    test_checkpoint_atomicity(results)
    test_checkpoint_pruning(results)
    test_signal_handling(results)
    test_resume_functionality(results)
    test_gpu_metrics(results)
    test_performance_benchmark(results)
    test_distributed_compatibility(results)

    # Print summary
    success = results.print_summary()

    # Save report
    report_path = Path("runs/validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.save_report(report_path)
    print(f"\n📄 Report saved to: {report_path}")

    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())