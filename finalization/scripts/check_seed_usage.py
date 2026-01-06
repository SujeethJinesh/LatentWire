#!/usr/bin/env python3
"""
Check seed usage patterns across the LatentWire codebase.

This script analyzes code to verify:
1. Proper seed setting patterns are used
2. Seeds 42, 123, 456 are consistently used
3. All random operations are seeded
4. Dataset shuffling is controlled

This runs without requiring PyTorch or other dependencies.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Expected seeds for multi-seed experiments
EXPECTED_SEEDS = [42, 123, 456]


class SeedUsageChecker:
    """Check seed usage patterns in code."""

    def __init__(self):
        self.results = {
            "files_checked": 0,
            "issues": [],
            "warnings": [],
            "good_practices": [],
            "seed_patterns": {},
            "summary": {}
        }

    def check_file_for_seeds(self, filepath: Path) -> Dict:
        """Check a single file for seed usage patterns."""
        content = filepath.read_text()
        file_results = {
            "has_torch_manual_seed": False,
            "has_numpy_seed": False,
            "has_random_seed": False,
            "has_cuda_seed": False,
            "has_transformers_seed": False,
            "has_deterministic_settings": False,
            "uses_expected_seeds": False,
            "seed_values": set()
        }

        # Check for seed setting patterns
        patterns = {
            "torch_manual_seed": r'torch\.manual_seed\s*\(',
            "numpy_seed": r'np\.random\.seed\s*\(|numpy\.random\.seed\s*\(',
            "random_seed": r'random\.seed\s*\(',
            "cuda_seed": r'torch\.cuda\.manual_seed|cuda\.manual_seed_all',
            "transformers_seed": r'transformers\.set_seed|from transformers import set_seed',
            "deterministic": r'use_deterministic_algorithms|cudnn\.deterministic|cudnn\.benchmark'
        }

        for name, pattern in patterns.items():
            if re.search(pattern, content):
                if "torch_manual" in name:
                    file_results["has_torch_manual_seed"] = True
                elif "numpy" in name:
                    file_results["has_numpy_seed"] = True
                elif "random_seed" in name:
                    file_results["has_random_seed"] = True
                elif "cuda" in name:
                    file_results["has_cuda_seed"] = True
                elif "transformers" in name:
                    file_results["has_transformers_seed"] = True
                elif "deterministic" in name:
                    file_results["has_deterministic_settings"] = True

        # Extract seed values
        seed_pattern = r'seed[=\s]*[:=]\s*(\d+)'
        seed_matches = re.findall(seed_pattern, content, re.IGNORECASE)
        for match in seed_matches:
            try:
                seed_val = int(match)
                file_results["seed_values"].add(seed_val)
            except:
                pass

        # Check if expected seeds are used
        for seed in EXPECTED_SEEDS:
            if str(seed) in content:
                file_results["uses_expected_seeds"] = True
                break

        return file_results

    def check_training_scripts(self) -> Dict:
        """Check main training scripts for proper seed usage."""
        training_files = [
            "latentwire/train.py",
            "latentwire/eval.py",
            "telepathy/train_telepathy_sst2.py",
            "telepathy/train_telepathy_agnews.py",
            "telepathy/train_telepathy_trec.py",
        ]

        results = {}
        for filepath in training_files:
            path = Path(filepath)
            if path.exists():
                file_results = self.check_file_for_seeds(path)
                results[filepath] = file_results

                # Check for issues
                if path.name == "train.py":
                    if not file_results["has_torch_manual_seed"]:
                        self.results["issues"].append(
                            f"{filepath}: Missing torch.manual_seed() call"
                        )
                    if not file_results["has_numpy_seed"]:
                        self.results["warnings"].append(
                            f"{filepath}: Missing numpy.random.seed() call"
                        )
                    if not file_results["has_random_seed"]:
                        self.results["warnings"].append(
                            f"{filepath}: Missing random.seed() call"
                        )
                    if not file_results["has_transformers_seed"]:
                        self.results["warnings"].append(
                            f"{filepath}: Not using transformers.set_seed() - some HF operations may be non-deterministic"
                        )
                    else:
                        self.results["good_practices"].append(
                            f"{filepath}: Properly sets all random seeds"
                        )

        return results

    def check_experiment_scripts(self) -> Dict:
        """Check experiment scripts for multi-seed configuration."""
        experiment_files = [
            "telepathy/run_unified_comparison.py",
            "telepathy/aggregate_results.py",
            "scripts/statistical_testing.py",
            "telepathy/run_paper_experiments.sh",
            "telepathy/run_enhanced_arxiv_suite.sh",
        ]

        results = {}
        for filepath in experiment_files:
            path = Path(filepath)
            if path.exists():
                content = path.read_text()

                # Check for expected seed values
                has_all_seeds = all(str(seed) in content for seed in EXPECTED_SEEDS)
                has_seed_list = "[42, 123, 456]" in content or "42 123 456" in content

                results[filepath] = {
                    "has_all_expected_seeds": has_all_seeds,
                    "has_seed_list": has_seed_list
                }

                if has_all_seeds:
                    self.results["good_practices"].append(
                        f"{filepath}: Uses standard seeds {EXPECTED_SEEDS}"
                    )
                else:
                    seeds_found = [s for s in EXPECTED_SEEDS if str(s) in content]
                    if seeds_found:
                        self.results["warnings"].append(
                            f"{filepath}: Only uses seeds {seeds_found}, missing some of {EXPECTED_SEEDS}"
                        )
                    else:
                        self.results["issues"].append(
                            f"{filepath}: Does not use standard seeds {EXPECTED_SEEDS}"
                        )

        return results

    def check_data_loading(self) -> Dict:
        """Check data loading functions for seed usage."""
        data_file = Path("latentwire/data.py")
        results = {}

        if data_file.exists():
            content = data_file.read_text()

            # Check each dataset loader
            loaders = [
                "load_squad_subset",
                "load_hotpot_subset",
                "load_gsm8k_subset",
                "load_trec_subset",
            ]

            for loader in loaders:
                # Check if function exists and uses seed parameter
                func_pattern = rf'def {loader}.*seed.*:'
                has_seed_param = bool(re.search(func_pattern, content, re.MULTILINE))

                # Check if it uses random.Random(seed) for shuffling
                if has_seed_param:
                    # Look for the function body
                    func_start = content.find(f"def {loader}")
                    if func_start != -1:
                        # Get next 500 chars after function def
                        func_snippet = content[func_start:func_start+500]
                        uses_random_with_seed = "Random(seed)" in func_snippet or "random.Random(seed)" in func_snippet
                        uses_shuffle = "shuffle" in func_snippet

                        results[loader] = {
                            "has_seed_param": has_seed_param,
                            "uses_random_with_seed": uses_random_with_seed,
                            "uses_shuffle": uses_shuffle
                        }

                        if uses_random_with_seed and uses_shuffle:
                            self.results["good_practices"].append(
                                f"data.py:{loader}: Properly uses seed for deterministic shuffling"
                            )
                        elif has_seed_param and not uses_random_with_seed:
                            self.results["issues"].append(
                                f"data.py:{loader}: Has seed parameter but doesn't use it properly for shuffling"
                            )

        return results

    def check_deterministic_settings(self) -> List[Tuple[str, str]]:
        """Find all files that configure deterministic settings."""
        deterministic_files = []

        patterns_to_check = [
            ("use_deterministic_algorithms", "Full determinism enabled"),
            ("cudnn.deterministic = True", "CUDNN deterministic mode"),
            ("cudnn.benchmark = False", "CUDNN benchmark disabled"),
            ("cudnn.benchmark = True", "CUDNN benchmark enabled (non-deterministic!)"),
        ]

        # Search Python files
        for root, dirs, files in os.walk("."):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'runs']]

            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    try:
                        content = filepath.read_text()
                        for pattern, description in patterns_to_check:
                            if pattern in content:
                                deterministic_files.append((str(filepath), description))
                                if "benchmark = True" in pattern:
                                    self.results["warnings"].append(
                                        f"{filepath}: Uses cudnn.benchmark=True which can cause non-deterministic behavior"
                                    )
                    except:
                        pass

        return deterministic_files

    def generate_summary(self):
        """Generate summary statistics."""
        self.results["summary"] = {
            "total_issues": len(self.results["issues"]),
            "total_warnings": len(self.results["warnings"]),
            "total_good_practices": len(self.results["good_practices"]),
            "uses_expected_seeds": any(
                str(seed) in str(self.results["seed_patterns"])
                for seed in EXPECTED_SEEDS
            )
        }

    def run_all_checks(self):
        """Run all seed usage checks."""
        print("=" * 80)
        print("SEED USAGE VERIFICATION")
        print("=" * 80)
        print()

        print("Checking training scripts...")
        training_results = self.check_training_scripts()
        self.results["training_scripts"] = training_results

        print("Checking experiment scripts...")
        experiment_results = self.check_experiment_scripts()
        self.results["experiment_scripts"] = experiment_results

        print("Checking data loading...")
        data_results = self.check_data_loading()
        self.results["data_loading"] = data_results

        print("Checking deterministic settings...")
        deterministic_files = self.check_deterministic_settings()
        self.results["deterministic_files"] = deterministic_files

        self.generate_summary()

    def print_report(self):
        """Print a detailed report."""
        print()
        print("=" * 80)
        print("VERIFICATION REPORT")
        print("=" * 80)
        print()

        # Good practices
        if self.results["good_practices"]:
            print("✅ GOOD PRACTICES FOUND:")
            for practice in self.results["good_practices"]:
                print(f"  • {practice}")
            print()

        # Issues
        if self.results["issues"]:
            print("❌ ISSUES FOUND:")
            for issue in self.results["issues"]:
                print(f"  • {issue}")
            print()

        # Warnings
        if self.results["warnings"]:
            print("⚠️  WARNINGS:")
            for warning in self.results["warnings"]:
                print(f"  • {warning}")
            print()

        # Deterministic settings
        if self.results.get("deterministic_files"):
            print("🔧 DETERMINISTIC SETTINGS FOUND IN:")
            for filepath, description in self.results["deterministic_files"]:
                print(f"  • {filepath}: {description}")
            print()

        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        summary = self.results["summary"]
        print(f"Issues found: {summary['total_issues']}")
        print(f"Warnings: {summary['total_warnings']}")
        print(f"Good practices: {summary['total_good_practices']}")
        print(f"Uses standard seeds (42, 123, 456): {'Yes' if summary['uses_expected_seeds'] else 'No'}")
        print()

        # Recommendations
        print("RECOMMENDATIONS FOR FULL REPRODUCIBILITY:")
        print("-" * 40)

        recommendations = []

        # Check for missing transformers seed
        if any("transformers.set_seed" in w for w in self.results["warnings"]):
            recommendations.append(
                "Add 'from transformers import set_seed; set_seed(args.seed)' to training scripts"
            )

        # Check for cudnn.benchmark
        if any("cudnn.benchmark=True" in w for w in self.results["warnings"]):
            recommendations.append(
                "Set torch.backends.cudnn.benchmark = False for full determinism"
            )

        # Check for missing standard seeds
        if not summary["uses_expected_seeds"]:
            recommendations.append(
                f"Use standard seeds {EXPECTED_SEEDS} consistently across all experiments"
            )

        if not recommendations:
            recommendations.append("Continue following current good practices!")

        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

        print()
        print("=" * 80)

        # Save results
        output_file = Path("runs/seed_usage_report.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            # Convert sets to lists for JSON serialization
            json_results = json.loads(json.dumps(self.results, default=str))
            json.dump(json_results, f, indent=2)
        print(f"Full report saved to: {output_file}")


def main():
    """Run seed usage verification."""
    checker = SeedUsageChecker()
    checker.run_all_checks()
    checker.print_report()


if __name__ == "__main__":
    main()