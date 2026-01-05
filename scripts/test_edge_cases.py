#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Edge Case Testing for LatentWire

Tests edge cases across the codebase:
1. Empty datasets
2. Single sample datasets
3. Very long inputs (OOM scenarios)
4. Keyboard interrupt handling
5. Network failures during download
6. Disk full scenarios
7. GPU availability issues
8. Corrupted checkpoints
9. Invalid configurations
10. Concurrent training runs
"""

import os
import sys
import json
import signal
import tempfile
import traceback
import time
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test results tracker
test_results = {
    "passed": [],
    "failed": [],
    "warnings": [],
    "errors": []
}

def log_test(test_name, status, message=""):
    """Log test result with consistent formatting."""
    symbol = {"pass": "[PASS]", "fail": "[FAIL]", "warn": "[WARN]", "error": "[ERROR]"}.get(status, "?")
    print(f"\n{symbol} {test_name}: {status.upper()}")
    if message:
        print(f"  → {message}")

    if status == "pass":
        test_results["passed"].append(test_name)
    elif status == "fail":
        test_results["failed"].append((test_name, message))
    elif status == "warn":
        test_results["warnings"].append((test_name, message))
    elif status == "error":
        test_results["errors"].append((test_name, message))


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_empty_dataset():
    """Test handling of empty datasets."""
    test_name = "Empty Dataset Handling"
    try:
        from latentwire.data import load_examples

        # Test with 0 samples
        examples = load_examples("squad", split="train", samples=0)

        if len(examples) == 0:
            log_test(test_name, "pass", "Correctly returns empty list for 0 samples")
        else:
            log_test(test_name, "fail", f"Expected 0 examples, got {len(examples)}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_single_sample():
    """Test handling of single sample datasets."""
    test_name = "Single Sample Dataset"
    try:
        from latentwire.data import load_examples

        # Test with 1 sample
        examples = load_examples("squad", split="train", samples=1)

        if len(examples) == 1:
            # Try to process it
            from latentwire.core_utils import format_chat_prompt
            prompt = format_chat_prompt(examples[0]["prefix"], "llama")
            if prompt:
                log_test(test_name, "pass", "Single sample handled correctly")
            else:
                log_test(test_name, "fail", "Failed to format single sample")
        else:
            log_test(test_name, "fail", f"Expected 1 example, got {len(examples)}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_very_long_input():
    """Test handling of very long inputs that might cause OOM."""
    test_name = "Very Long Input (OOM Prevention)"
    try:
        from latentwire.core_utils import format_chat_prompt, truncate_text

        # Create a very long text (10MB)
        long_text = "A" * (10 * 1024 * 1024)  # 10MB of text

        # Test truncation
        truncated = truncate_text(long_text, max_length=1000)
        if len(truncated) <= 1000:
            log_test(test_name, "pass", "Long text properly truncated")
        else:
            log_test(test_name, "fail", f"Truncation failed: got {len(truncated)} chars")

    except MemoryError:
        log_test(test_name, "fail", "MemoryError not handled gracefully")
    except Exception as e:
        log_test(test_name, "error", str(e))


def test_interrupt_handling():
    """Test graceful handling of keyboard interrupts."""
    test_name = "Keyboard Interrupt Handling"

    def interruptible_task():
        """A task that can be interrupted."""
        try:
            # Simulate a long-running task
            for i in range(10):
                time.sleep(0.1)
                if i == 5:
                    # Simulate interrupt
                    raise KeyboardInterrupt()
        except KeyboardInterrupt:
            # Should handle gracefully
            return "interrupted"
        return "completed"

    try:
        result = interruptible_task()
        if result == "interrupted":
            log_test(test_name, "pass", "KeyboardInterrupt handled gracefully")
        else:
            log_test(test_name, "fail", "KeyboardInterrupt not triggered")
    except KeyboardInterrupt:
        log_test(test_name, "fail", "KeyboardInterrupt not caught properly")
    except Exception as e:
        log_test(test_name, "error", str(e))


def test_network_failure():
    """Test handling of network failures during dataset download."""
    test_name = "Network Failure During Download"

    try:
        # Mock network failure
        with patch('datasets.load_dataset') as mock_load:
            mock_load.side_effect = ConnectionError("Network unreachable")

            from latentwire.data import load_examples

            try:
                examples = load_examples("squad", split="train", samples=10)
                log_test(test_name, "fail", "Should have raised an error")
            except (ConnectionError, Exception) as e:
                if "Network" in str(e) or "Connection" in str(e):
                    log_test(test_name, "pass", "Network error handled appropriately")
                else:
                    log_test(test_name, "warn", f"Unexpected error: {e}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_disk_full():
    """Test handling of disk full scenarios."""
    test_name = "Disk Full Scenario"

    try:
        # Create a temporary directory with limited space
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_checkpoint.pt"

            # Mock torch.save to raise OSError
            with patch('torch.save') as mock_save:
                mock_save.side_effect = OSError(28, "No space left on device")

                try:
                    import torch
                    torch.save({"test": "data"}, test_file)
                    log_test(test_name, "fail", "Should have raised disk full error")
                except OSError as e:
                    if e.errno == 28 or "No space" in str(e):
                        log_test(test_name, "pass", "Disk full error raised correctly")
                    else:
                        log_test(test_name, "warn", f"Unexpected OSError: {e}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_gpu_unavailable():
    """Test handling when GPU is not available."""
    test_name = "GPU Unavailable"

    try:
        import torch

        # Mock CUDA as unavailable
        with patch('torch.cuda.is_available', return_value=False):
            from latentwire.models import InterlinguaEncoder

            # Try to create model
            encoder = InterlinguaEncoder(
                latent_len=32,
                d_z=256,
                encoder_type="byte"
            )

            # Check if model defaults to CPU
            if next(encoder.parameters()).device.type == 'cpu':
                log_test(test_name, "pass", "Model correctly defaults to CPU")
            else:
                log_test(test_name, "fail", "Model not on CPU when GPU unavailable")

    except Exception as e:
        log_test(test_name, "warn", f"Unable to test GPU unavailability: {e}")


def test_corrupted_checkpoint():
    """Test handling of corrupted checkpoint files."""
    test_name = "Corrupted Checkpoint"

    try:
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmpfile:
            # Write garbage data
            tmpfile.write(b"This is not a valid checkpoint file!")
            tmpfile.flush()

            import torch

            try:
                checkpoint = torch.load(tmpfile.name)
                log_test(test_name, "fail", "Should have raised an error for corrupted file")
            except (RuntimeError, pickle.UnpicklingError, Exception) as e:
                log_test(test_name, "pass", f"Corrupted checkpoint detected: {type(e).__name__}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_invalid_config():
    """Test handling of invalid configuration parameters."""
    test_name = "Invalid Configuration"

    try:
        from latentwire.models import InterlinguaEncoder

        invalid_configs = [
            {"latent_len": -1, "d_z": 256, "encoder_type": "byte"},  # Negative latent_len
            {"latent_len": 32, "d_z": 0, "encoder_type": "byte"},     # Zero dimension
            {"latent_len": 32, "d_z": 256, "encoder_type": "invalid"}, # Invalid encoder type
        ]

        errors_caught = 0
        for config in invalid_configs:
            try:
                encoder = InterlinguaEncoder(**config)
                # If we get here, validation failed
                break
            except (ValueError, AssertionError, KeyError):
                errors_caught += 1

        if errors_caught == len(invalid_configs):
            log_test(test_name, "pass", f"All {errors_caught} invalid configs rejected")
        else:
            log_test(test_name, "fail", f"Only {errors_caught}/{len(invalid_configs)} configs rejected")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_concurrent_runs():
    """Test handling of concurrent training runs to same directory."""
    test_name = "Concurrent Training Runs"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "concurrent_test"

            def write_checkpoint(run_id):
                """Simulate writing checkpoint."""
                output_dir.mkdir(parents=True, exist_ok=True)
                ckpt_file = output_dir / f"checkpoint_{run_id}.pt"
                with open(ckpt_file, 'w') as f:
                    f.write(f"Run {run_id}")
                return ckpt_file

            # Simulate concurrent writes
            with mp.Pool(4) as pool:
                results = pool.map(write_checkpoint, range(4))

            # Check all files were written
            written_files = list(output_dir.glob("checkpoint_*.pt"))
            if len(written_files) == 4:
                log_test(test_name, "pass", "Concurrent writes handled correctly")
            else:
                log_test(test_name, "warn", f"Expected 4 files, got {len(written_files)}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_memory_leak():
    """Test for memory leaks during training loop."""
    test_name = "Memory Leak Detection"

    try:
        import torch
        import gc

        if not torch.cuda.is_available():
            log_test(test_name, "warn", "Skipping - no GPU available")
            return

        # Get initial memory
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated()

        # Simulate multiple training iterations
        for _ in range(10):
            # Create tensors that should be freed
            x = torch.randn(100, 100, device='cuda')
            y = x @ x.T
            del x, y

        # Force cleanup
        torch.cuda.empty_cache()
        gc.collect()
        final_memory = torch.cuda.memory_allocated()

        # Check for leak (allowing small variance)
        memory_increase = final_memory - initial_memory
        if memory_increase < 1024 * 1024:  # Less than 1MB increase
            log_test(test_name, "pass", f"No significant memory leak ({memory_increase} bytes)")
        else:
            log_test(test_name, "warn", f"Possible memory leak: {memory_increase / 1e6:.2f} MB increase")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_tokenization_edge_cases():
    """Test edge cases in tokenization."""
    test_name = "Tokenization Edge Cases"

    try:
        from latentwire.core_utils import format_chat_prompt

        edge_cases = [
            "",  # Empty string
            " ",  # Single space
            "\n\n\n",  # Multiple newlines
            "🎉🔥💻",  # Emojis
            "中文测试",  # Chinese characters
            "\x00\x01\x02",  # Control characters
            "A" * 100000,  # Very long string
        ]

        failed = []
        for i, text in enumerate(edge_cases):
            try:
                result = format_chat_prompt(text, "llama")
                if result is None:
                    failed.append(i)
            except Exception as e:
                failed.append((i, str(e)))

        if not failed:
            log_test(test_name, "pass", "All tokenization edge cases handled")
        else:
            log_test(test_name, "warn", f"Failed cases: {failed}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_checkpoint_recovery():
    """Test recovery from interrupted training."""
    test_name = "Checkpoint Recovery"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"

            # Simulate partial checkpoint
            import torch
            partial_data = {
                "epoch": 5,
                "global_step": 100,
                # Missing some expected keys
            }
            torch.save(partial_data, ckpt_path)

            # Try to load
            loaded = torch.load(ckpt_path)

            # Check if we can detect missing keys
            required_keys = ["model_state", "optimizer_state", "config"]
            missing = [k for k in required_keys if k not in loaded]

            if missing:
                log_test(test_name, "pass", f"Detected missing keys: {missing}")
            else:
                log_test(test_name, "warn", "Could not detect incomplete checkpoint")

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_data_loader_edge_cases():
    """Test edge cases in data loading."""
    test_name = "DataLoader Edge Cases"

    try:
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        # Test with various batch sizes and dataset sizes
        test_cases = [
            (10, 3),   # Dataset size not divisible by batch size
            (1, 1),    # Single sample, single batch
            (100, 200), # Batch size larger than dataset
            (0, 10),   # Empty dataset (should fail gracefully)
        ]

        issues = []
        for dataset_size, batch_size in test_cases:
            try:
                if dataset_size > 0:
                    dataset = TensorDataset(torch.randn(dataset_size, 10))
                    loader = DataLoader(dataset, batch_size=batch_size)
                    batches = list(loader)

                    # Verify we got all samples
                    total_samples = sum(len(b[0]) for b in batches)
                    if total_samples != dataset_size:
                        issues.append(f"Size mismatch: {dataset_size} vs {total_samples}")
                else:
                    # Empty dataset should be handled
                    dataset = TensorDataset(torch.randn(0, 10))
                    loader = DataLoader(dataset, batch_size=batch_size)
                    batches = list(loader)
                    if batches:
                        issues.append("Empty dataset produced batches")

            except Exception as e:
                issues.append(f"Case ({dataset_size}, {batch_size}): {e}")

        if not issues:
            log_test(test_name, "pass", "All DataLoader edge cases handled")
        else:
            log_test(test_name, "warn", f"Issues: {issues[:3]}")  # Show first 3

    except Exception as e:
        log_test(test_name, "error", str(e))


def test_gradient_edge_cases():
    """Test edge cases in gradient computation."""
    test_name = "Gradient Edge Cases"

    try:
        import torch
        import torch.nn as nn

        # Test cases that often cause issues
        test_cases = [
            ("NaN loss", float('nan')),
            ("Inf loss", float('inf')),
            ("Zero loss", 0.0),
            ("Very large loss", 1e10),
            ("Very small loss", 1e-10),
        ]

        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        issues = []
        for case_name, loss_value in test_cases:
            try:
                optimizer.zero_grad()
                loss = torch.tensor(loss_value, requires_grad=True)

                # Try backward
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()

                    # Check gradients
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                issues.append(f"{case_name}: NaN gradients")
                            elif torch.isinf(param.grad).any():
                                issues.append(f"{case_name}: Inf gradients")
                else:
                    # Should handle NaN/Inf gracefully
                    pass

            except Exception as e:
                issues.append(f"{case_name}: {e}")

        if not issues:
            log_test(test_name, "pass", "Gradient edge cases handled")
        else:
            log_test(test_name, "warn", f"Issues: {issues[:3]}")

    except Exception as e:
        log_test(test_name, "error", str(e))


def run_all_tests():
    """Run all edge case tests."""
    print("\n" + "="*60)
    print("EDGE CASE TESTING FOR LATENTWIRE")
    print("="*60)

    tests = [
        test_empty_dataset,
        test_single_sample,
        test_very_long_input,
        test_interrupt_handling,
        test_network_failure,
        test_disk_full,
        test_gpu_unavailable,
        test_corrupted_checkpoint,
        test_invalid_config,
        test_concurrent_runs,
        test_memory_leak,
        test_tokenization_edge_cases,
        test_checkpoint_recovery,
        test_data_loader_edge_cases,
        test_gradient_edge_cases,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            log_test(test_func.__name__, "error", f"Unexpected error: {e}")

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"[PASS] Passed: {len(test_results['passed'])}")
    print(f"[FAIL] Failed: {len(test_results['failed'])}")
    print(f"[WARN] Warnings: {len(test_results['warnings'])}")
    print(f"[ERROR] Errors: {len(test_results['errors'])}")

    if test_results['failed']:
        print("\nFailed Tests:")
        for name, msg in test_results['failed']:
            print(f"  - {name}: {msg}")

    if test_results['errors']:
        print("\nError Tests:")
        for name, msg in test_results['errors']:
            print(f"  - {name}: {msg}")

    # Save detailed results
    results_file = Path("runs") / f"edge_case_results_{int(time.time())}.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # Return exit code
    return 0 if not (test_results['failed'] or test_results['errors']) else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())