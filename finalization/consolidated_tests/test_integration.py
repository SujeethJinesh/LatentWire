#!/usr/bin/env python3
"""
Full pipeline integration test for LatentWire.

This test validates the complete training and evaluation pipeline:
1. Training for a few steps with checkpoint saving
2. Checkpoint resumption and continued training
3. Evaluation on the checkpoint
4. Validates all components work together end-to-end

Usage:
    python test_integration.py
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import numpy as np
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch not available, some tests will be skipped")


class IntegrationTest:
    """Full pipeline integration test suite."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_dir = None
        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": [],
            "timings": {}
        }

    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        if self.verbose:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def setup(self):
        """Set up test environment."""
        self.log("Setting up test environment...")

        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp(prefix="latentwire_test_"))
        self.log(f"Test directory: {self.test_dir}")

        # Set environment variables
        os.environ["PYTHONPATH"] = str(Path(__file__).parent)
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        return True

    def cleanup(self):
        """Clean up test environment."""
        self.log("Cleaning up test environment...")

        if self.test_dir and self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
                self.log(f"Removed test directory: {self.test_dir}")
            except Exception as e:
                self.log(f"Warning: Could not remove test directory: {e}", "WARN")

    def run_command(self, cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
        """Run a command and capture output."""
        self.log(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout} seconds", "ERROR")
            return -1, "", "Command timed out"
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            return -1, "", str(e)

    def test_initial_training(self) -> Dict[str, Any]:
        """Test initial training with checkpoint saving."""
        self.log("=" * 60)
        self.log("TEST 1: Initial Training with Checkpointing")
        self.log("=" * 60)

        start_time = time.time()

        # Create output directory
        train_dir = self.test_dir / "train_initial"
        train_dir.mkdir(parents=True)

        # Training command for 10 steps
        cmd = [
            sys.executable,
            "latentwire/train.py",
            "--llama_id", "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "--qwen_id", "Qwen/Qwen2.5-7B-Instruct",
            "--samples", "50",  # Small sample size for quick test
            "--epochs", "1",
            "--batch_size", "4",
            "--latent_len", "8",
            "--d_z", "128",
            "--encoder_type", "byte",
            "--dataset", "squad",
            "--save_dir", str(train_dir),
            "--save_every", "5",  # Save checkpoint every 5 samples
            "--save_training_stats",
            "--sequential_models",  # Process models sequentially to save memory
            "--warm_anchor_text", "Answer: ",
            "--first_token_ce_weight", "0.5"
        ]

        returncode, stdout, stderr = self.run_command(cmd, timeout=600)

        elapsed = time.time() - start_time
        self.timings["initial_training"] = elapsed

        # Check if training succeeded
        success = returncode == 0

        # Verify checkpoint was created
        checkpoint_exists = False
        checkpoint_path = None

        if train_dir.exists():
            checkpoints = list(train_dir.glob("checkpoint_*"))
            if checkpoints:
                checkpoint_exists = True
                checkpoint_path = checkpoints[-1]  # Get latest checkpoint
                self.log(f"Found checkpoint: {checkpoint_path}")

        result = {
            "success": success and checkpoint_exists,
            "returncode": returncode,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "elapsed_time": elapsed,
            "train_dir": str(train_dir),
            "error": stderr if not success else None
        }

        if result["success"]:
            self.log(f"✓ Initial training completed in {elapsed:.2f}s", "SUCCESS")
            self.results["tests_passed"] += 1
        else:
            self.log(f"✗ Initial training failed: {stderr[:500]}", "ERROR")
            self.results["tests_failed"] += 1
            self.results["failures"].append("initial_training")

        return result

    def test_checkpoint_resume(self, checkpoint_path: str) -> Dict[str, Any]:
        """Test resuming from checkpoint."""
        self.log("=" * 60)
        self.log("TEST 2: Checkpoint Resume")
        self.log("=" * 60)

        if not checkpoint_path:
            self.log("Skipping resume test - no checkpoint available", "WARN")
            return {"success": False, "skipped": True}

        start_time = time.time()

        # Create output directory for resumed training
        resume_dir = self.test_dir / "train_resume"
        resume_dir.mkdir(parents=True)

        # Resume training command
        cmd = [
            sys.executable,
            "latentwire/train.py",
            "--llama_id", "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "--qwen_id", "Qwen/Qwen2.5-7B-Instruct",
            "--samples", "100",  # Train for more samples
            "--epochs", "1",
            "--batch_size", "4",
            "--latent_len", "8",
            "--d_z", "128",
            "--encoder_type", "byte",
            "--dataset", "squad",
            "--save_dir", str(resume_dir),
            "--save_every", "10",
            "--save_training_stats",
            "--resume_from", checkpoint_path,  # Resume from checkpoint
            "--sequential_models",
            "--warm_anchor_text", "Answer: ",
            "--first_token_ce_weight", "0.5"
        ]

        returncode, stdout, stderr = self.run_command(cmd, timeout=600)

        elapsed = time.time() - start_time
        self.timings["checkpoint_resume"] = elapsed

        # Check if resume succeeded
        success = returncode == 0

        # Verify new checkpoints were created
        new_checkpoints = False
        if resume_dir.exists():
            checkpoints = list(resume_dir.glob("checkpoint_*"))
            new_checkpoints = len(checkpoints) > 0
            if new_checkpoints:
                self.log(f"Created {len(checkpoints)} new checkpoints during resume")

        # Check if training stats show continuation
        stats_continued = False
        stats_file = resume_dir / "training_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    stats = json.load(f)
                    # Check if we have entries showing resume worked
                    if "sample_stats" in stats and len(stats["sample_stats"]) > 0:
                        # Check if first entry shows resumed state
                        first_stat = stats["sample_stats"][0]
                        if first_stat.get("samples_seen", 0) > 0:
                            stats_continued = True
                            self.log(f"Resume detected: started from sample {first_stat['samples_seen']}")
            except Exception as e:
                self.log(f"Could not parse training stats: {e}", "WARN")

        result = {
            "success": success and new_checkpoints,
            "returncode": returncode,
            "elapsed_time": elapsed,
            "resume_dir": str(resume_dir),
            "stats_continued": stats_continued,
            "error": stderr if not success else None
        }

        if result["success"]:
            self.log(f"✓ Checkpoint resume completed in {elapsed:.2f}s", "SUCCESS")
            self.results["tests_passed"] += 1
        else:
            self.log(f"✗ Checkpoint resume failed: {stderr[:500]}", "ERROR")
            self.results["tests_failed"] += 1
            self.results["failures"].append("checkpoint_resume")

        return result

    def test_evaluation(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Test evaluation on saved checkpoint."""
        self.log("=" * 60)
        self.log("TEST 3: Evaluation on Checkpoint")
        self.log("=" * 60)

        if not checkpoint_dir or not Path(checkpoint_dir).exists():
            self.log("Skipping evaluation test - no checkpoint directory", "WARN")
            return {"success": False, "skipped": True}

        start_time = time.time()

        # Find the latest checkpoint
        checkpoint_dir_path = Path(checkpoint_dir)
        checkpoints = list(checkpoint_dir_path.glob("checkpoint_*"))

        if not checkpoints:
            self.log("No checkpoints found for evaluation", "WARN")
            return {"success": False, "skipped": True}

        latest_checkpoint = sorted(checkpoints)[-1]
        self.log(f"Using checkpoint: {latest_checkpoint}")

        # Create output directory for evaluation
        eval_dir = self.test_dir / "eval_results"
        eval_dir.mkdir(parents=True)

        # Evaluation command
        cmd = [
            sys.executable,
            "latentwire/eval.py",
            "--ckpt", str(latest_checkpoint),
            "--samples", "10",  # Small sample for quick test
            "--max_new_tokens", "12",
            "--dataset", "squad",
            "--sequential_eval",
            "--fresh_eval",
            "--calibration", "embed_rms",
            "--latent_anchor_mode", "text",
            "--latent_anchor_text", "Answer: ",
            "--append_bos_after_prefix", "yes",
            "--output_dir", str(eval_dir)
        ]

        returncode, stdout, stderr = self.run_command(cmd, timeout=300)

        elapsed = time.time() - start_time
        self.timings["evaluation"] = elapsed

        # Check if evaluation succeeded
        success = returncode == 0

        # Check for evaluation results
        results_found = False
        metrics = {}

        # Look for results in output
        if success and stdout:
            # Parse evaluation metrics from stdout
            for line in stdout.split('\n'):
                if 'F1' in line or 'EM' in line or 'NLL' in line:
                    self.log(f"Eval metric: {line.strip()}")
                    if 'latent' in line.lower():
                        results_found = True

        # Check for saved results files
        if eval_dir.exists():
            results_files = list(eval_dir.glob("*.json"))
            if results_files:
                results_found = True
                self.log(f"Found {len(results_files)} result files")

                # Try to load and display metrics
                for rf in results_files[:1]:  # Just check first file
                    try:
                        with open(rf) as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                metrics = {k: v for k, v in data.items()
                                         if k in ['f1', 'em', 'nll_per_token']}
                                if metrics:
                                    self.log(f"Metrics: {metrics}")
                    except Exception as e:
                        self.log(f"Could not parse results file: {e}", "WARN")

        result = {
            "success": success and results_found,
            "returncode": returncode,
            "elapsed_time": elapsed,
            "eval_dir": str(eval_dir),
            "metrics": metrics,
            "error": stderr if not success else None
        }

        if result["success"]:
            self.log(f"✓ Evaluation completed in {elapsed:.2f}s", "SUCCESS")
            self.results["tests_passed"] += 1
        else:
            self.log(f"✗ Evaluation failed: {stderr[:500] if stderr else 'No results found'}", "ERROR")
            self.results["tests_failed"] += 1
            self.results["failures"].append("evaluation")

        return result

    def test_import_integrity(self) -> bool:
        """Test that all required modules can be imported."""
        self.log("=" * 60)
        self.log("TEST 4: Import Integrity Check")
        self.log("=" * 60)

        modules_to_test = [
            "latentwire.train",
            "latentwire.eval",
            "latentwire.models",
            "latentwire.losses",
            "latentwire.data",
            "latentwire.core_utils",
            "latentwire.checkpointing",
            "latentwire.feature_registry"
        ]

        all_imports_ok = True

        for module in modules_to_test:
            try:
                __import__(module)
                self.log(f"✓ Successfully imported {module}")
            except ImportError as e:
                self.log(f"✗ Failed to import {module}: {e}", "ERROR")
                all_imports_ok = False
                self.results["failures"].append(f"import_{module}")

        if all_imports_ok:
            self.log("✓ All imports successful", "SUCCESS")
            self.results["tests_passed"] += 1
        else:
            self.results["tests_failed"] += 1

        return all_imports_ok

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        self.log("=" * 60)
        self.log("LATENTWIRE FULL PIPELINE INTEGRATION TEST")
        self.log("=" * 60)

        overall_start = time.time()

        try:
            # Setup
            if not self.setup():
                self.log("Setup failed, aborting tests", "ERROR")
                return self.results

            # Test 1: Import integrity
            self.test_import_integrity()

            # Test 2: Initial training
            train_result = self.test_initial_training()

            # Test 3: Resume from checkpoint (if initial training succeeded)
            resume_result = {"skipped": True}
            if train_result.get("success") and train_result.get("checkpoint_path"):
                resume_result = self.test_checkpoint_resume(train_result["checkpoint_path"])

            # Test 4: Evaluation (on resume directory if available, else initial)
            eval_checkpoint_dir = resume_result.get("resume_dir") if resume_result.get("success") else train_result.get("train_dir")
            if eval_checkpoint_dir:
                self.test_evaluation(eval_checkpoint_dir)

        except Exception as e:
            self.log(f"Unexpected error during tests: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            self.results["tests_failed"] += 1
            self.results["failures"].append(f"unexpected_error: {str(e)}")

        finally:
            # Cleanup
            self.cleanup()

        # Calculate total time
        total_time = time.time() - overall_start
        self.results["total_time"] = total_time
        self.results["timings"] = self.timings

        # Generate summary
        self.log("=" * 60)
        self.log("TEST SUMMARY")
        self.log("=" * 60)
        self.log(f"Tests Passed: {self.results['tests_passed']}")
        self.log(f"Tests Failed: {self.results['tests_failed']}")

        if self.results["failures"]:
            self.log(f"Failed Tests: {', '.join(self.results['failures'])}")

        self.log(f"Total Time: {total_time:.2f}s")

        # Timing breakdown
        if self.timings:
            self.log("\nTiming Breakdown:")
            for name, duration in self.timings.items():
                self.log(f"  {name}: {duration:.2f}s")

        # Overall status
        if self.results["tests_failed"] == 0:
            self.log("\n✓ ALL TESTS PASSED - Pipeline is working correctly!", "SUCCESS")
        else:
            self.log(f"\n✗ {self.results['tests_failed']} TESTS FAILED - Pipeline needs attention", "ERROR")

        return self.results


def main():
    """Main entry point for integration test."""

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="LatentWire Full Pipeline Integration Test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", type=str, help="Output file for results JSON")
    args = parser.parse_args()

    # Run tests
    tester = IntegrationTest(verbose=True)  # Always verbose for integration tests
    results = tester.run_all_tests()

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Exit with appropriate code
    exit_code = 0 if results["tests_failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()