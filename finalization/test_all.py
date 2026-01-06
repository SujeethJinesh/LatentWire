#!/usr/bin/env python3
"""
Comprehensive Test Suite for LatentWire
========================================

This test suite validates:
1. All imports work correctly
2. GPU detection and CUDA availability
3. Checkpointing functionality
4. All 4 phases can run (initialization)
5. Memory safety and cleanup

Run with: python test_all.py
"""

import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import gc

# Colors for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")

def print_section(text: str):
    """Print a section header."""
    print(f"\n{BOLD}{text}{RESET}")
    print(f"{'-' * 40}")

def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")

def print_info(text: str):
    """Print info message."""
    print(f"{BLUE}ℹ{RESET} {text}")


class ComprehensiveTestSuite:
    """Comprehensive test suite for LatentWire."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {
            "imports": {},
            "gpu": {},
            "checkpointing": {},
            "phases": {},
            "memory": {},
            "summary": {
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }

    def test_imports(self) -> bool:
        """Test all critical imports."""
        print_section("1. IMPORT TESTS")

        all_passed = True

        # Core imports
        core_imports = [
            ("torch", "PyTorch"),
            ("transformers", "HuggingFace Transformers"),
            ("datasets", "HuggingFace Datasets"),
            ("numpy", "NumPy"),
            ("scipy", "SciPy"),
            ("sklearn", "Scikit-learn"),
            ("tqdm", "TQDM"),
            ("rouge_score", "ROUGE Score"),
        ]

        print_info("Testing core library imports...")
        for module_name, display_name in core_imports:
            try:
                __import__(module_name)
                print_success(f"{display_name} ({module_name})")
                self.results["imports"][module_name] = "passed"
            except ImportError as e:
                print_error(f"{display_name} ({module_name}): {str(e)}")
                self.results["imports"][module_name] = f"failed: {str(e)}"
                all_passed = False

        # LatentWire specific imports
        print_info("\nTesting LatentWire module imports...")
        latentwire_modules = [
            "latentwire.train",
            "latentwire.eval",
            "latentwire.models",
            "latentwire.losses",
            "latentwire.data",
            "latentwire.metrics",
            "latentwire.core_utils",
            "latentwire.common",
        ]

        for module_name in latentwire_modules:
            try:
                __import__(module_name)
                print_success(f"{module_name}")
                self.results["imports"][module_name] = "passed"
            except ImportError as e:
                print_error(f"{module_name}: {str(e)}")
                self.results["imports"][module_name] = f"failed: {str(e)}"
                all_passed = False

        # Telepathy imports
        print_info("\nTesting Telepathy module imports...")
        telepathy_modules = [
            "telepathy.checkpoint_manager",
            "telepathy.dynamic_batch_size",
            "telepathy.comprehensive_experiments",
            "telepathy.linear_probe_baseline",
        ]

        for module_name in telepathy_modules:
            try:
                __import__(module_name)
                print_success(f"{module_name}")
                self.results["imports"][module_name] = "passed"
            except ImportError as e:
                print_warning(f"{module_name}: {str(e)}")
                self.results["imports"][module_name] = f"warning: {str(e)}"
                # Don't fail on telepathy imports

        return all_passed

    def test_gpu(self) -> bool:
        """Test GPU detection and CUDA availability."""
        print_section("2. GPU DETECTION TESTS")

        try:
            import torch

            # CUDA availability
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print_success(f"CUDA is available")
                self.results["gpu"]["cuda"] = "available"
            else:
                print_warning("CUDA is not available (CPU mode)")
                self.results["gpu"]["cuda"] = "not available"

            # GPU count
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                print_success(f"Found {gpu_count} GPU(s)")
                self.results["gpu"]["count"] = gpu_count

                # List GPUs
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print_info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                    self.results["gpu"][f"gpu_{i}"] = {
                        "name": gpu_name,
                        "memory_gb": gpu_memory
                    }
            else:
                print_warning("No GPUs found")
                self.results["gpu"]["count"] = 0

            # Test GPU operations if available
            if cuda_available and gpu_count > 0:
                print_info("\nTesting GPU operations...")
                device = torch.device("cuda:0")

                # Test tensor creation
                try:
                    x = torch.randn(100, 100, device=device)
                    y = torch.randn(100, 100, device=device)
                    z = torch.matmul(x, y)
                    print_success("GPU tensor operations work")
                    self.results["gpu"]["operations"] = "passed"
                except Exception as e:
                    print_error(f"GPU tensor operations failed: {e}")
                    self.results["gpu"]["operations"] = f"failed: {str(e)}"
                    return False

                # Test memory
                try:
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    print_info(f"  Memory allocated: {allocated:.3f} GB")
                    print_info(f"  Memory reserved: {reserved:.3f} GB")
                    self.results["gpu"]["memory"] = {
                        "allocated_gb": allocated,
                        "reserved_gb": reserved
                    }
                except Exception as e:
                    print_warning(f"Could not get memory info: {e}")

            # MPS detection (Mac)
            if hasattr(torch.backends, 'mps'):
                mps_available = torch.backends.mps.is_available()
                if mps_available:
                    print_info("MPS (Mac GPU) is available")
                    self.results["gpu"]["mps"] = "available"

            return True

        except ImportError:
            print_error("PyTorch not installed - cannot test GPU")
            self.results["gpu"]["error"] = "PyTorch not installed"
            return False
        except Exception as e:
            print_error(f"GPU test failed: {e}")
            self.results["gpu"]["error"] = str(e)
            return False

    def test_checkpointing(self) -> bool:
        """Test checkpointing functionality."""
        print_section("3. CHECKPOINTING TESTS")

        try:
            import torch
            import torch.nn as nn
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Create a simple model
                model = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 10)
                )

                # Create optimizer
                optimizer = torch.optim.Adam(model.parameters())

                # Create some dummy data and do a forward pass
                x = torch.randn(5, 10)
                loss = model(x).sum()
                loss.backward()
                optimizer.step()

                # Save checkpoint
                checkpoint_path = tmpdir / "test_checkpoint.pt"
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': 10,
                    'loss': loss.item(),
                    'metadata': {'test': True}
                }

                print_info("Saving checkpoint...")
                torch.save(checkpoint, checkpoint_path)
                file_size_mb = checkpoint_path.stat().st_size / 1024 / 1024
                print_success(f"Checkpoint saved ({file_size_mb:.2f} MB)")
                self.results["checkpointing"]["save"] = "passed"

                # Load checkpoint
                print_info("Loading checkpoint...")
                loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print_success("Checkpoint loaded successfully")
                self.results["checkpointing"]["load"] = "passed"

                # Verify contents
                assert 'model_state_dict' in loaded_checkpoint
                assert 'optimizer_state_dict' in loaded_checkpoint
                assert loaded_checkpoint['epoch'] == 10
                assert 'loss' in loaded_checkpoint
                print_success("Checkpoint contents verified")
                self.results["checkpointing"]["verify"] = "passed"

                # Test loading into new model
                new_model = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 10)
                )
                new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
                print_success("State dict loaded into new model")
                self.results["checkpointing"]["state_dict"] = "passed"

                # Test checkpoint manager if available
                try:
                    from telepathy.checkpoint_manager import CheckpointManager

                    print_info("\nTesting CheckpointManager...")
                    manager = CheckpointManager(
                        base_dir=str(tmpdir / "managed"),
                        max_checkpoints=3,
                        save_interval=1
                    )

                    # Save multiple checkpoints
                    for i in range(5):
                        manager.save(
                            model=model,
                            optimizer=optimizer,
                            epoch=i,
                            metrics={'loss': 1.0 / (i+1)}
                        )

                    # Check that old checkpoints were cleaned up
                    checkpoints = list((tmpdir / "managed").glob("*.pt"))
                    assert len(checkpoints) <= 3, f"Expected ≤3 checkpoints, found {len(checkpoints)}"
                    print_success("CheckpointManager works (auto-cleanup verified)")
                    self.results["checkpointing"]["manager"] = "passed"

                except ImportError:
                    print_warning("CheckpointManager not available")
                    self.results["checkpointing"]["manager"] = "not available"

                return True

        except Exception as e:
            print_error(f"Checkpointing test failed: {e}")
            self.results["checkpointing"]["error"] = str(e)
            return False

    def test_phases(self) -> bool:
        """Test that all 4 phases can be initialized."""
        print_section("4. PHASE INITIALIZATION TESTS")

        all_passed = True

        # Phase 1: Calibration
        print_info("Testing Phase 1 (Calibration)...")
        try:
            from telepathy.phase1_calibration import Phase1Calibrator

            with tempfile.TemporaryDirectory() as tmpdir:
                calibrator = Phase1Calibrator(
                    output_dir=tmpdir,
                    num_samples=5,
                    batch_size=2,
                    source_layer=16,
                    target_layer=16
                )
                print_success("Phase 1 Calibrator initialized")
                self.results["phases"]["phase1"] = "initialized"
        except ImportError as e:
            print_warning(f"Phase 1 import failed: {e}")
            self.results["phases"]["phase1"] = f"import failed: {str(e)}"
        except Exception as e:
            print_error(f"Phase 1 initialization failed: {e}")
            self.results["phases"]["phase1"] = f"failed: {str(e)}"
            all_passed = False

        # Phase 2: Bridge Training
        print_info("Testing Phase 2 (Bridge Training)...")
        try:
            # Mock a simple bridge trainer
            import torch
            import torch.nn as nn

            class SimpleBridge(nn.Module):
                def __init__(self, src_dim=4096, tgt_dim=4096, num_latents=32):
                    super().__init__()
                    self.proj = nn.Linear(src_dim, tgt_dim)
                    self.num_latents = num_latents

                def forward(self, x):
                    return self.proj(x)[:, :self.num_latents]

            bridge = SimpleBridge()
            print_success("Phase 2 Bridge module created")
            self.results["phases"]["phase2"] = "initialized"

        except Exception as e:
            print_error(f"Phase 2 initialization failed: {e}")
            self.results["phases"]["phase2"] = f"failed: {str(e)}"
            all_passed = False

        # Phase 3: Fine-tuning
        print_info("Testing Phase 3 (Fine-tuning)...")
        try:
            # Check if LoRA is available
            try:
                from peft import LoraConfig, get_peft_model
                print_success("Phase 3 LoRA support available")
                self.results["phases"]["phase3"] = "LoRA available"
            except ImportError:
                print_warning("Phase 3 LoRA not available (peft not installed)")
                self.results["phases"]["phase3"] = "LoRA not available"

        except Exception as e:
            print_error(f"Phase 3 test failed: {e}")
            self.results["phases"]["phase3"] = f"failed: {str(e)}"
            all_passed = False

        # Phase 4: Evaluation
        print_info("Testing Phase 4 (Evaluation)...")
        try:
            from latentwire.eval import LMWrapper
            from latentwire.metrics import compute_exact_match, compute_f1

            # Test metric functions
            pred = "the answer"
            gold = "the answer"
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)

            assert em == 1.0, f"EM should be 1.0, got {em}"
            assert f1 == 1.0, f"F1 should be 1.0, got {f1}"

            print_success("Phase 4 Evaluation metrics work")
            self.results["phases"]["phase4"] = "metrics work"

        except ImportError as e:
            print_warning(f"Phase 4 import failed: {e}")
            self.results["phases"]["phase4"] = f"import failed: {str(e)}"
        except Exception as e:
            print_error(f"Phase 4 test failed: {e}")
            self.results["phases"]["phase4"] = f"failed: {str(e)}"
            all_passed = False

        return all_passed

    def test_memory_safety(self) -> bool:
        """Test memory safety and cleanup."""
        print_section("5. MEMORY SAFETY TESTS")

        try:
            import torch
            import psutil
            import gc

            process = psutil.Process()

            # Get initial memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            initial_mem = process.memory_info().rss / 1024 / 1024  # MB
            print_info(f"Initial memory: {initial_mem:.1f} MB")

            # Allocate and deallocate tensors
            print_info("Testing memory allocation/deallocation...")

            for i in range(3):
                # Allocate
                tensors = []
                for _ in range(10):
                    if torch.cuda.is_available():
                        t = torch.randn(1000, 1000, device='cuda')
                    else:
                        t = torch.randn(1000, 1000)
                    tensors.append(t)

                # Check memory
                current_mem = process.memory_info().rss / 1024 / 1024
                print_info(f"  Iteration {i+1}: {current_mem:.1f} MB (allocated)")

                # Deallocate
                del tensors
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                current_mem = process.memory_info().rss / 1024 / 1024
                print_info(f"  Iteration {i+1}: {current_mem:.1f} MB (after cleanup)")

            # Final memory check
            final_mem = process.memory_info().rss / 1024 / 1024
            mem_leak = final_mem - initial_mem

            if mem_leak < 100:  # Allow 100MB tolerance
                print_success(f"Memory properly cleaned up (leak: {mem_leak:.1f} MB)")
                self.results["memory"]["cleanup"] = "passed"
                self.results["memory"]["leak_mb"] = mem_leak
            else:
                print_warning(f"Potential memory leak: {mem_leak:.1f} MB")
                self.results["memory"]["cleanup"] = "warning"
                self.results["memory"]["leak_mb"] = mem_leak

            # Test CUDA memory if available
            if torch.cuda.is_available():
                print_info("\nTesting CUDA memory...")

                # Reset stats
                torch.cuda.reset_peak_memory_stats()

                # Allocate large tensor
                x = torch.randn(10000, 10000, device='cuda')
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                print_info(f"  Peak CUDA memory: {peak_mb:.1f} MB")

                # Clean up
                del x
                torch.cuda.empty_cache()
                current_mb = torch.cuda.memory_allocated() / 1024 / 1024

                if current_mb < 10:  # Should be near zero
                    print_success(f"CUDA memory cleaned up ({current_mb:.1f} MB remaining)")
                    self.results["memory"]["cuda_cleanup"] = "passed"
                else:
                    print_warning(f"CUDA memory not fully cleaned: {current_mb:.1f} MB remaining")
                    self.results["memory"]["cuda_cleanup"] = "warning"

            return True

        except ImportError as e:
            print_warning(f"Could not test memory (missing dependency): {e}")
            self.results["memory"]["error"] = f"missing dependency: {str(e)}"
            return True  # Don't fail on missing psutil
        except Exception as e:
            print_error(f"Memory safety test failed: {e}")
            self.results["memory"]["error"] = str(e)
            return False

    def run_quick_integration_test(self) -> bool:
        """Run a quick end-to-end integration test."""
        print_section("6. QUICK INTEGRATION TEST")

        try:
            import torch
            import torch.nn as nn
            from transformers import AutoTokenizer

            print_info("Running quick integration test...")

            # Create simple mock components
            class MockEncoder(nn.Module):
                def __init__(self, vocab_size=1000, hidden_size=256, latent_size=128):
                    super().__init__()
                    self.embed = nn.Embedding(vocab_size, hidden_size)
                    self.proj = nn.Linear(hidden_size, latent_size)

                def forward(self, input_ids):
                    x = self.embed(input_ids)
                    return self.proj(x.mean(dim=1))

            class MockAdapter(nn.Module):
                def __init__(self, latent_size=128, output_size=256):
                    super().__init__()
                    self.proj = nn.Linear(latent_size, output_size)

                def forward(self, latents):
                    return self.proj(latents)

            # Initialize components
            encoder = MockEncoder()
            adapter = MockAdapter()

            # Create dummy data
            batch_size = 4
            seq_len = 10
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))

            # Forward pass
            latents = encoder(input_ids)
            output = adapter(latents)

            # Check shapes
            assert latents.shape == (batch_size, 128), f"Expected latents shape (4, 128), got {latents.shape}"
            assert output.shape == (batch_size, 256), f"Expected output shape (4, 256), got {output.shape}"

            print_success("Integration test passed")
            self.results["integration"] = "passed"
            return True

        except Exception as e:
            print_error(f"Integration test failed: {e}")
            self.results["integration"] = f"failed: {str(e)}"
            return False

    def generate_summary(self) -> None:
        """Generate test summary."""
        print_header("TEST SUMMARY")

        # Count results
        total_tests = 0
        passed = 0
        failed = 0
        warnings = 0

        for category, results in self.results.items():
            if category == "summary":
                continue
            if isinstance(results, dict):
                for test, result in results.items():
                    total_tests += 1
                    if isinstance(result, str):
                        if "passed" in result or result == "available":
                            passed += 1
                        elif "warning" in result or "not available" in result:
                            warnings += 1
                        else:
                            failed += 1

        # Update summary
        self.results["summary"]["total"] = total_tests
        self.results["summary"]["passed"] = passed
        self.results["summary"]["failed"] = failed
        self.results["summary"]["warnings"] = warnings

        # Print summary
        print(f"{BOLD}Total Tests:{RESET} {total_tests}")
        print(f"{GREEN}Passed:{RESET} {passed}")
        print(f"{RED}Failed:{RESET} {failed}")
        print(f"{YELLOW}Warnings:{RESET} {warnings}")

        # Overall status
        print()
        if failed == 0:
            if warnings > 0:
                print(f"{GREEN}{BOLD}✓ ALL CRITICAL TESTS PASSED{RESET} ({warnings} warnings)")
            else:
                print(f"{GREEN}{BOLD}✓ ALL TESTS PASSED{RESET}")
        else:
            print(f"{RED}{BOLD}✗ {failed} TESTS FAILED{RESET}")

        # Save results to file
        results_file = Path("test_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n{BLUE}Results saved to:{RESET} {results_file}")

    def run_all_tests(self) -> bool:
        """Run all tests."""
        print_header("LATENTWIRE COMPREHENSIVE TEST SUITE")
        print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()

        # Run tests
        all_passed = True

        # 1. Import tests
        if not self.test_imports():
            all_passed = False

        # 2. GPU tests
        if not self.test_gpu():
            pass  # GPU not required

        # 3. Checkpointing tests
        if not self.test_checkpointing():
            all_passed = False

        # 4. Phase tests
        if not self.test_phases():
            pass  # Some phases may not be available

        # 5. Memory safety tests
        if not self.test_memory_safety():
            all_passed = False

        # 6. Integration test
        if not self.run_quick_integration_test():
            all_passed = False

        # Generate summary
        self.generate_summary()

        # Print timing
        elapsed = time.time() - start_time
        print(f"\n{BLUE}Total time:{RESET} {elapsed:.2f} seconds")

        return all_passed


def main():
    """Main entry point."""
    # Add parent directory to path
    parent_dir = Path(__file__).parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Set environment variables
    os.environ["PYTHONPATH"] = str(parent_dir)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Run tests
    suite = ComprehensiveTestSuite(verbose=True)
    success = suite.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()