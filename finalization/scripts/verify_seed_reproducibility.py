#!/usr/bin/env python3
"""
Verify seed reproducibility across LatentWire experiments.

This script performs comprehensive checks to ensure:
1. Seeds are properly set for all random operations
2. PyTorch operations are deterministic
3. Dataset shuffling is controlled by seed
4. Multi-seed aggregation works correctly
5. Same seed gives identical results
6. Different seeds give different results

Usage:
    python scripts/verify_seed_reproducibility.py
"""

import os
import sys
import json
import random
import hashlib
import numpy as np
import torch
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import LatentWire modules
try:
    from latentwire.data import (
        load_squad_subset,
        load_hotpot_subset,
        load_gsm8k_subset,
        load_trec_subset
    )
    ALL_DATASETS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some datasets not available: {e}")
    ALL_DATASETS_AVAILABLE = False

# Expected seeds for multi-seed experiments
EXPECTED_SEEDS = [42, 123, 456]


class SeedVerifier:
    """Comprehensive seed reproducibility verification."""

    def __init__(self):
        self.results = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "checks": {},
            "warnings": []
        }

    def set_seed(self, seed: int):
        """Set all random seeds consistently."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def check_torch_determinism(self) -> Dict[str, bool]:
        """Check PyTorch deterministic settings."""
        checks = {}

        # Check if deterministic algorithms can be enabled
        try:
            # Save current state
            old_deterministic = torch.are_deterministic_algorithms_enabled()

            # Try to enable deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=True)
            checks["deterministic_algorithms_available"] = True

            # Restore old state
            torch.use_deterministic_algorithms(old_deterministic)
        except:
            checks["deterministic_algorithms_available"] = False
            self.results["warnings"].append(
                "Cannot enable fully deterministic algorithms. Some operations may be non-deterministic."
            )

        # Check CUDNN settings
        if torch.cuda.is_available():
            checks["cudnn_deterministic"] = torch.backends.cudnn.deterministic
            checks["cudnn_benchmark"] = torch.backends.cudnn.benchmark

            if torch.backends.cudnn.benchmark:
                self.results["warnings"].append(
                    "cudnn.benchmark=True may cause non-deterministic behavior. "
                    "Set to False for full reproducibility."
                )

        return checks

    def check_random_state(self, seed: int) -> Dict[str, bool]:
        """Verify random number generation is deterministic."""
        results = {}

        # Test Python random
        self.set_seed(seed)
        py_samples1 = [random.random() for _ in range(10)]
        self.set_seed(seed)
        py_samples2 = [random.random() for _ in range(10)]
        results["python_random_deterministic"] = (py_samples1 == py_samples2)

        # Test NumPy random
        self.set_seed(seed)
        np_samples1 = np.random.randn(10)
        self.set_seed(seed)
        np_samples2 = np.random.randn(10)
        results["numpy_random_deterministic"] = np.allclose(np_samples1, np_samples2)

        # Test PyTorch random
        self.set_seed(seed)
        torch_samples1 = torch.randn(10)
        self.set_seed(seed)
        torch_samples2 = torch.randn(10)
        results["torch_random_deterministic"] = torch.allclose(torch_samples1, torch_samples2)

        # Test CUDA random if available
        if torch.cuda.is_available():
            self.set_seed(seed)
            cuda_samples1 = torch.randn(10, device='cuda')
            self.set_seed(seed)
            cuda_samples2 = torch.randn(10, device='cuda')
            results["cuda_random_deterministic"] = torch.allclose(cuda_samples1, cuda_samples2)

        return results

    def check_dataset_shuffling(self) -> Dict[str, bool]:
        """Verify dataset loading is deterministic with same seed."""
        results = {}

        datasets = {}
        if ALL_DATASETS_AVAILABLE:
            datasets.update({
                "squad": lambda seed: load_squad_subset(split="train", samples=10, seed=seed),
                "hotpot": lambda seed: load_hotpot_subset(split="train", samples=10, seed=seed),
                "gsm8k": lambda seed: load_gsm8k_subset(split="train", samples=10, seed=seed),
                "trec": lambda seed: load_trec_subset(split="test", samples=10, seed=seed),
            })
        else:
            self.results["warnings"].append("Some datasets not available for testing")

        for dataset_name, load_func in datasets.items():
            try:
                # Load with same seed twice
                data1 = load_func(42)
                data2 = load_func(42)

                # Check if data is identical
                same_seed_identical = self._compare_datasets(data1, data2)
                results[f"{dataset_name}_same_seed_identical"] = same_seed_identical

                # Load with different seed
                data3 = load_func(123)

                # Check if data is different
                diff_seed_different = not self._compare_datasets(data1, data3)
                results[f"{dataset_name}_diff_seed_different"] = diff_seed_different

                if not same_seed_identical:
                    self.results["warnings"].append(
                        f"{dataset_name}: Same seed produces different data! Non-deterministic loading."
                    )
                if not diff_seed_different:
                    self.results["warnings"].append(
                        f"{dataset_name}: Different seeds produce same data! Seed not affecting shuffle."
                    )

            except Exception as e:
                results[f"{dataset_name}_error"] = str(e)
                self.results["warnings"].append(f"Error loading {dataset_name}: {e}")

        return results

    def _compare_datasets(self, data1: List[Dict], data2: List[Dict]) -> bool:
        """Compare two dataset samples for equality."""
        if len(data1) != len(data2):
            return False

        for d1, d2 in zip(data1, data2):
            if set(d1.keys()) != set(d2.keys()):
                return False
            for key in d1.keys():
                if d1[key] != d2[key]:
                    return False
        return True

    def check_model_initialization(self) -> Dict[str, bool]:
        """Verify model initialization is deterministic."""
        results = {}

        try:
            from latentwire.models import InterlinguaInterlinguaEncoder, Adapter

            # Test encoder initialization
            self.set_seed(42)
            enc1 = InterlinguaEncoder(
                d_z=128,
                latent_len=32,
                encoder_type="byte",
                vocab_size=256
            )
            params1 = self._get_param_hash(enc1)

            self.set_seed(42)
            enc2 = InterlinguaEncoder(
                d_z=128,
                latent_len=32,
                encoder_type="byte",
                vocab_size=256
            )
            params2 = self._get_param_hash(enc2)

            results["encoder_init_deterministic"] = (params1 == params2)

            # Test with different seed
            self.set_seed(123)
            enc3 = InterlinguaEncoder(
                d_z=128,
                latent_len=32,
                encoder_type="byte",
                vocab_size=256
            )
            params3 = self._get_param_hash(enc3)

            results["encoder_init_seed_sensitive"] = (params1 != params3)

        except Exception as e:
            results["model_init_error"] = str(e)
            self.results["warnings"].append(f"Error testing model initialization: {e}")

        return results

    def _get_param_hash(self, model: torch.nn.Module) -> str:
        """Get a hash of model parameters for comparison."""
        param_bytes = b""
        for param in model.parameters():
            param_bytes += param.data.cpu().numpy().tobytes()
        return hashlib.md5(param_bytes).hexdigest()

    def check_multi_seed_consistency(self) -> Dict[str, bool]:
        """Verify multi-seed experiments use expected seeds."""
        results = {}

        # Check if statistical testing expects correct seeds
        try:
            from scripts.statistical_testing import aggregate_multiseed_results

            # Create mock results with expected seeds
            mock_scores = {seed: 0.8 + seed/1000 for seed in EXPECTED_SEEDS}
            stats = aggregate_multiseed_results(mock_scores)

            results["statistical_testing_handles_seeds"] = (
                stats["n"] == len(EXPECTED_SEEDS) and
                not np.isnan(stats["std"])
            )

        except Exception as e:
            results["statistical_testing_error"] = str(e)
            self.results["warnings"].append(f"Error importing statistical testing: {e}")

        # Check experiment scripts for seed configuration
        script_paths = [
            "telepathy/run_unified_comparison.py",
            "telepathy/run_paper_experiments.sh",
            "telepathy/aggregate_results.py"
        ]

        for script_path in script_paths:
            script_file = Path(script_path)
            if script_file.exists():
                content = script_file.read_text()
                # Check for expected seed values
                has_seeds = all(str(seed) in content for seed in EXPECTED_SEEDS)
                results[f"{script_file.stem}_has_expected_seeds"] = has_seeds

                if not has_seeds:
                    self.results["warnings"].append(
                        f"{script_path} may not use standard seeds {EXPECTED_SEEDS}"
                    )

        return results

    def check_transformers_seed(self) -> Dict[str, bool]:
        """Check if transformers library seed is properly set."""
        results = {}

        try:
            from transformers import set_seed as transformers_set_seed

            # Test if transformers set_seed works
            transformers_set_seed(42)
            results["transformers_set_seed_available"] = True

            # Check if it's being used in training
            train_file = Path("latentwire/train.py")
            if train_file.exists():
                content = train_file.read_text()
                uses_transformers_seed = "transformers.set_seed" in content or "from transformers import set_seed" in content
                results["train_uses_transformers_seed"] = uses_transformers_seed

                if not uses_transformers_seed:
                    self.results["warnings"].append(
                        "latentwire/train.py does not use transformers.set_seed(). "
                        "Some HuggingFace operations may be non-deterministic."
                    )

        except ImportError:
            results["transformers_not_available"] = True

        return results

    def run_all_checks(self) -> Dict:
        """Run all reproducibility checks."""
        print("=" * 80)
        print("SEED REPRODUCIBILITY VERIFICATION")
        print("=" * 80)
        print()

        # Check torch determinism settings
        print("Checking PyTorch determinism settings...")
        self.results["checks"]["torch_determinism"] = self.check_torch_determinism()

        # Check random state reproducibility
        print("Checking random number generation...")
        self.results["checks"]["random_state"] = self.check_random_state(seed=42)

        # Check dataset shuffling
        print("Checking dataset loading reproducibility...")
        self.results["checks"]["dataset_shuffling"] = self.check_dataset_shuffling()

        # Check model initialization
        print("Checking model initialization...")
        self.results["checks"]["model_initialization"] = self.check_model_initialization()

        # Check multi-seed configuration
        print("Checking multi-seed experiment configuration...")
        self.results["checks"]["multi_seed"] = self.check_multi_seed_consistency()

        # Check transformers seed
        print("Checking transformers library seed handling...")
        self.results["checks"]["transformers_seed"] = self.check_transformers_seed()

        return self.results

    def print_summary(self):
        """Print a summary of verification results."""
        print()
        print("=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print()

        # Count successes and failures
        total_checks = 0
        passed_checks = 0
        failed_checks = []

        for category, checks in self.results["checks"].items():
            for check_name, result in checks.items():
                if isinstance(result, bool):
                    total_checks += 1
                    if result:
                        passed_checks += 1
                    else:
                        failed_checks.append(f"{category}.{check_name}")

        # Print statistics
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}/{total_checks} ({100*passed_checks/total_checks:.1f}%)")
        print()

        # Print failures
        if failed_checks:
            print("FAILED CHECKS:")
            for check in failed_checks:
                print(f"  ❌ {check}")
            print()

        # Print warnings
        if self.results["warnings"]:
            print("WARNINGS:")
            for warning in self.results["warnings"]:
                print(f"  ⚠️  {warning}")
            print()

        # Overall status
        if passed_checks == total_checks and not self.results["warnings"]:
            print("✅ ALL CHECKS PASSED - Full reproducibility verified!")
        elif passed_checks == total_checks:
            print("✅ All checks passed, but there are warnings to address.")
        else:
            print("❌ Some checks failed. Review above for details.")

        print()

        # Recommendations
        print("RECOMMENDATIONS FOR FULL REPRODUCIBILITY:")
        print("-" * 40)

        recommendations = []

        # Check for CUDNN benchmark
        if self.results["checks"].get("torch_determinism", {}).get("cudnn_benchmark"):
            recommendations.append(
                "Set torch.backends.cudnn.benchmark = False for deterministic GPU operations"
            )

        # Check for deterministic algorithms
        if not self.results["checks"].get("torch_determinism", {}).get("deterministic_algorithms_available"):
            recommendations.append(
                "Add torch.use_deterministic_algorithms(True) to enable deterministic operations"
            )

        # Check for transformers seed
        if not self.results["checks"].get("transformers_seed", {}).get("train_uses_transformers_seed"):
            recommendations.append(
                "Add 'from transformers import set_seed; set_seed(args.seed)' in train.py"
            )

        if not recommendations:
            print("  No additional changes needed for reproducibility.")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print()
        print("=" * 80)

        # Save results to file
        output_file = Path("runs/seed_verification_results.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Full results saved to: {output_file}")


def main():
    """Run seed reproducibility verification."""
    verifier = SeedVerifier()

    # Run all checks
    results = verifier.run_all_checks()

    # Print summary
    verifier.print_summary()

    # Return exit code based on results
    all_passed = all(
        result for checks in results["checks"].values()
        for result in checks.values()
        if isinstance(result, bool)
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()