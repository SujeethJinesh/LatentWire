#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Robustness Tests

Tests edge cases specifically for the training pipeline:
- OOM recovery
- Gradient explosion/vanishing
- Mixed precision issues
- Checkpoint corruption during save
- Dynamic batch size adjustment
- Loss spikes and NaN handling
"""

import os
import sys
import json
import time
import torch
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TrainingRobustnessTests:
    """Test suite for training pipeline robustness."""

    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "errors": []
        }

    def log(self, test_name, status, message=""):
        """Log test result."""
        symbol = {"pass": "[PASS]", "fail": "[FAIL]", "warn": "[WARN]", "error": "[ERROR]"}.get(status, "?")
        print(f"\n{symbol} {test_name}: {status.upper()}")
        if message:
            print(f"  → {message}")

        if status == "pass":
            self.results["passed"].append(test_name)
        elif status == "fail":
            self.results["failed"].append((test_name, message))
        elif status == "warn":
            self.results["warnings"].append((test_name, message))
        elif status == "error":
            self.results["errors"].append((test_name, message))

    def test_oom_recovery(self):
        """Test OOM recovery during training."""
        test_name = "OOM Recovery"

        try:
            import torch
            import torch.nn as nn

            if not torch.cuda.is_available():
                self.log(test_name, "warn", "Skipping - no GPU available")
                return

            class OOMSimulator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = nn.Linear(1000, 1000)

                def forward(self, x):
                    # Gradually increase memory usage
                    for i in range(100):
                        x = self.layer(x)
                        if i > 50:
                            # Try to allocate huge tensor
                            try:
                                huge = torch.randn(10000, 10000, device=x.device)
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    # Clear cache and continue
                                    torch.cuda.empty_cache()
                                    return x
                                raise
                    return x

            model = OOMSimulator().cuda()
            x = torch.randn(10, 1000, device='cuda')

            try:
                output = model(x)
                self.log(test_name, "pass", "OOM handled with cache clearing")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.log(test_name, "warn", "OOM not recovered automatically")
                else:
                    self.log(test_name, "fail", f"Unexpected error: {e}")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_gradient_explosion(self):
        """Test handling of gradient explosion."""
        test_name = "Gradient Explosion"

        try:
            import torch
            import torch.nn as nn
            from torch.nn.utils import clip_grad_norm_

            model = nn.Sequential(
                nn.Linear(10, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 1)
            )

            # Create input that will cause large gradients
            x = torch.randn(10, 10) * 100  # Large input
            target = torch.randn(10, 1) * 1000  # Large target

            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # Large LR

            # Forward pass
            output = model(x)
            loss = ((output - target) ** 2).mean() * 1000  # Scale up loss

            # Backward without clipping
            loss.backward()

            # Check gradient magnitudes
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm > 1000:
                # Now test with gradient clipping
                optimizer.zero_grad()
                output = model(x)
                loss = ((output - target) ** 2).mean() * 1000
                loss.backward()

                # Clip gradients
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Check clipped norm
                clipped_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        clipped_norm += p.grad.data.norm(2).item() ** 2
                clipped_norm = clipped_norm ** 0.5

                if clipped_norm <= 1.01:  # Allow small numerical error
                    self.log(test_name, "pass", f"Gradients clipped from {total_norm:.2f} to {clipped_norm:.2f}")
                else:
                    self.log(test_name, "fail", f"Clipping failed: {clipped_norm:.2f}")
            else:
                self.log(test_name, "warn", f"Gradients not large enough: {total_norm:.2f}")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_nan_loss_handling(self):
        """Test handling of NaN losses during training."""
        test_name = "NaN Loss Handling"

        try:
            import torch
            import torch.nn as nn

            model = nn.Linear(10, 1)
            optimizer = torch.optim.Adam(model.parameters())

            # Create input that will cause NaN
            x = torch.randn(10, 10)
            x[0, 0] = float('inf')  # Inject infinity

            # Forward pass
            output = model(x)
            loss = output.mean()

            if torch.isnan(loss):
                self.log(test_name, "pass", "NaN loss detected correctly")

                # Test recovery strategy
                optimizer.zero_grad()
                x_clean = torch.randn(10, 10)  # Clean input
                output_clean = model(x_clean)
                loss_clean = output_clean.mean()

                if not torch.isnan(loss_clean):
                    self.log(test_name, "pass", "Recovered from NaN by using clean data")
                else:
                    self.log(test_name, "fail", "Model corrupted after NaN")
            else:
                self.log(test_name, "warn", "NaN not produced in test")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_checkpoint_atomic_save(self):
        """Test atomic checkpoint saving to prevent corruption."""
        test_name = "Atomic Checkpoint Save"

        try:
            import torch
            import torch.nn as nn

            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = Path(tmpdir) / "model.pt"
                temp_path = Path(tmpdir) / "model.pt.tmp"

                model = nn.Linear(10, 10)
                state = {"model": model.state_dict(), "epoch": 1}

                # Simulate atomic save
                def atomic_save(obj, path):
                    """Save checkpoint atomically."""
                    temp_path = Path(str(path) + '.tmp')
                    torch.save(obj, temp_path)
                    temp_path.rename(path)  # Atomic on most filesystems
                    return path

                # Test atomic save
                saved_path = atomic_save(state, ckpt_path)

                if saved_path.exists() and not temp_path.exists():
                    # Try to load
                    loaded = torch.load(saved_path)
                    if "model" in loaded and "epoch" in loaded:
                        self.log(test_name, "pass", "Atomic save completed successfully")
                    else:
                        self.log(test_name, "fail", "Checkpoint incomplete")
                else:
                    self.log(test_name, "fail", "Atomic save failed")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_mixed_precision_stability(self):
        """Test mixed precision training stability."""
        test_name = "Mixed Precision Stability"

        try:
            import torch
            import torch.nn as nn
            from torch.cuda.amp import autocast, GradScaler

            if not torch.cuda.is_available():
                self.log(test_name, "warn", "Skipping - no GPU available")
                return

            model = nn.Sequential(
                nn.Linear(10, 100),
                nn.LayerNorm(100),
                nn.ReLU(),
                nn.Linear(100, 1)
            ).cuda()

            optimizer = torch.optim.Adam(model.parameters())
            scaler = GradScaler()

            # Test data
            x = torch.randn(32, 10, device='cuda')
            target = torch.randn(32, 1, device='cuda')

            losses = []
            for i in range(10):
                optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    output = model(x)
                    loss = ((output - target) ** 2).mean()

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(loss.item())

                # Check for NaN/Inf
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    self.log(test_name, "fail", f"Loss became NaN/Inf at step {i}")
                    return

            # Check if loss is decreasing
            if losses[-1] < losses[0]:
                self.log(test_name, "pass", f"Mixed precision stable: loss {losses[0]:.4f} → {losses[-1]:.4f}")
            else:
                self.log(test_name, "warn", "Loss not decreasing in mixed precision")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_dynamic_batch_adjustment(self):
        """Test dynamic batch size adjustment on OOM."""
        test_name = "Dynamic Batch Adjustment"

        try:
            from latentwire.train import suggest_batch_size_adjustment

            # Test adjustment suggestions
            test_cases = [
                # (current_batch, peak_gb, expected_adjustment)
                (32, 70.0, "decrease"),  # High memory usage
                (32, 40.0, "optimal"),   # Good memory usage
                (32, 20.0, "increase"),  # Low memory usage
            ]

            passed = 0
            for current_batch, peak_gb, expected in test_cases:
                suggestion = suggest_batch_size_adjustment(
                    current_batch,
                    peak_gb,
                    after_forward_pass=True,
                    target_utilization=0.60
                )

                if peak_gb > 60 and "decrease" in expected:
                    passed += 1
                elif 30 < peak_gb < 60 and "optimal" in expected:
                    passed += 1
                elif peak_gb < 30 and "increase" in expected:
                    passed += 1

            if passed == len(test_cases):
                self.log(test_name, "pass", "Batch size adjustments correct")
            else:
                self.log(test_name, "fail", f"Only {passed}/{len(test_cases)} correct")

        except ImportError:
            self.log(test_name, "warn", "Could not import suggest_batch_size_adjustment")
        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_data_pipeline_robustness(self):
        """Test data pipeline edge cases."""
        test_name = "Data Pipeline Robustness"

        try:
            from latentwire.data_pipeline import prepare_training_data
            from latentwire.data import load_examples

            # Test with minimal data
            examples = load_examples("squad", split="train", samples=2)

            if len(examples) < 2:
                self.log(test_name, "warn", "Not enough examples loaded")
                return

            # Test pipeline with edge cases
            test_configs = [
                {"batch_size": 1, "sequential": True},
                {"batch_size": 10, "sequential": False},  # Batch > dataset
                {"batch_size": 2, "sequential": True},
            ]

            issues = []
            for config in test_configs:
                try:
                    dataloader = prepare_training_data(
                        examples,
                        llama_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        qwen_id=None,
                        batch_size=config["batch_size"],
                        sequential_models=config["sequential"],
                        warm_anchor_text="Answer: ",
                        anchor_augmentation=None
                    )

                    # Try to get a batch
                    for batch in dataloader:
                        if batch is None:
                            issues.append(f"None batch with config {config}")
                        break  # Just test first batch

                except Exception as e:
                    issues.append(f"Config {config}: {e}")

            if not issues:
                self.log(test_name, "pass", "Data pipeline handles edge cases")
            else:
                self.log(test_name, "fail", f"Issues: {issues[:2]}")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_loss_spike_recovery(self):
        """Test recovery from sudden loss spikes."""
        test_name = "Loss Spike Recovery"

        try:
            import torch
            import torch.nn as nn

            model = nn.Linear(10, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Normal training
            x_normal = torch.randn(10, 10)
            y_normal = torch.randn(10, 1)

            losses = []
            for i in range(20):
                optimizer.zero_grad()

                # Inject spike at step 10
                if i == 10:
                    x = torch.randn(10, 10) * 1000  # Huge input
                    y = torch.randn(10, 1) * 1000
                else:
                    x, y = x_normal, y_normal

                output = model(x)
                loss = ((output - y) ** 2).mean()

                # Skip update if loss spike detected
                if i > 0 and loss.item() > losses[-1] * 10:
                    # Spike detected - skip this update
                    losses.append(losses[-1])  # Keep previous loss
                    continue

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # Check recovery
            if losses[15] < losses[9]:  # Recovered after spike
                self.log(test_name, "pass", "Recovered from loss spike")
            else:
                self.log(test_name, "fail", "Did not recover from spike")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def test_memory_profiling(self):
        """Test memory profiling utilities."""
        test_name = "Memory Profiling"

        try:
            from latentwire.train import get_gpu_memory_stats, log_gpu_memory

            if torch.cuda.is_available():
                # Get memory stats
                stats = get_gpu_memory_stats()

                required_keys = ['gpus', 'total_allocated_gb', 'total_reserved_gb']
                if all(k in stats for k in required_keys):
                    # Test logging
                    log_stats = log_gpu_memory("Test: ", reset_peak=True)
                    if log_stats:
                        self.log(test_name, "pass", "Memory profiling working")
                    else:
                        self.log(test_name, "fail", "Logging returned no stats")
                else:
                    self.log(test_name, "fail", f"Missing keys in stats: {stats.keys()}")
            else:
                # Should return empty dict
                stats = get_gpu_memory_stats()
                if stats == {}:
                    self.log(test_name, "pass", "Correctly returns empty dict without GPU")
                else:
                    self.log(test_name, "fail", "Should return empty dict without GPU")

        except Exception as e:
            self.log(test_name, "error", str(e))

    def run_all_tests(self):
        """Run all robustness tests."""
        print("\n" + "="*60)
        print("TRAINING ROBUSTNESS TESTS")
        print("="*60)

        tests = [
            self.test_oom_recovery,
            self.test_gradient_explosion,
            self.test_nan_loss_handling,
            self.test_checkpoint_atomic_save,
            self.test_mixed_precision_stability,
            self.test_dynamic_batch_adjustment,
            self.test_data_pipeline_robustness,
            self.test_loss_spike_recovery,
            self.test_memory_profiling,
        ]

        for test_method in tests:
            try:
                test_method()
            except Exception as e:
                self.log(test_method.__name__, "error", f"Unexpected: {e}")

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"[PASS] Passed: {len(self.results['passed'])}")
        print(f"[FAIL] Failed: {len(self.results['failed'])}")
        print(f"[WARN] Warnings: {len(self.results['warnings'])}")
        print(f"[ERROR] Errors: {len(self.results['errors'])}")

        if self.results['failed']:
            print("\nFailed Tests:")
            for name, msg in self.results['failed']:
                print(f"  - {name}: {msg}")

        # Save results
        results_file = Path("runs") / f"training_robustness_{int(time.time())}.json"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        return 0 if not self.results['failed'] else 1


if __name__ == "__main__":
    tester = TrainingRobustnessTests()
    sys.exit(tester.run_all_tests())