#!/usr/bin/env python3
"""
Results Validation Script for Telepathy Experiments

This script validates that all experimental results are present, complete,
and valid for inclusion in the paper. It checks:
1. File existence and structure
2. Required fields in results
3. Metric validity (ranges, types)
4. Statistical test completion
5. No NaN/inf values

Usage:
    python telepathy/validate_results.py [--results-dir RESULTS_DIR]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime


class ResultsValidator:
    """Validates experimental results for completeness and correctness."""

    # Expected experiments based on REPORT.md phases
    EXPECTED_EXPERIMENTS = {
        "phase1_baseline": [
            "baseline_llama_8b",
            "baseline_llama_70b",
            "baseline_gpt4",
            "baseline_claude"
        ],
        "phase2_telepathy": [
            "telepathy_llama_8b_layer16_probe",
            "telepathy_llama_8b_layer24_probe",
            "telepathy_llama_8b_layer16_mlp",
            "telepathy_llama_8b_layer24_mlp"
        ],
        "phase3_cross_model": [
            "cross_model_llama_to_gpt2",
            "cross_model_llama_to_mistral",
            "cross_model_bidirectional"
        ],
        "phase4_analysis": [
            "attention_patterns",
            "probe_weights_analysis",
            "representation_similarity"
        ]
    }

    # Required fields for different result types
    REQUIRED_FIELDS = {
        "classification": ["accuracy", "precision", "recall", "f1", "confusion_matrix"],
        "probe": ["train_accuracy", "val_accuracy", "test_accuracy", "weights_shape"],
        "cross_model": ["source_model", "target_model", "transfer_accuracy", "baseline_accuracy"],
        "statistical": ["mean", "std", "confidence_interval", "p_value", "effect_size"],
        "timing": ["total_time", "samples_per_second", "gpu_memory_mb"]
    }

    # Valid ranges for metrics
    METRIC_RANGES = {
        "accuracy": (0.0, 1.0),
        "precision": (0.0, 1.0),
        "recall": (0.0, 1.0),
        "f1": (0.0, 1.0),
        "train_accuracy": (0.0, 1.0),
        "val_accuracy": (0.0, 1.0),
        "test_accuracy": (0.0, 1.0),
        "transfer_accuracy": (0.0, 1.0),
        "baseline_accuracy": (0.0, 1.0),
        "p_value": (0.0, 1.0),
        "effect_size": (-10.0, 10.0),  # Cohen's d can be large
        "confidence_interval": (0.0, 1.0),
        "samples_per_second": (0.0, float('inf')),
        "gpu_memory_mb": (0.0, 100000)  # Up to 100GB
    }

    def __init__(self, results_dir: Path):
        """Initialize validator with results directory."""
        self.results_dir = results_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.valid_results: Dict[str, Any] = {}

    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all validation checks.

        Returns:
            Tuple of (success, summary_dict)
        """
        print(f"{'='*60}")
        print(f"Telepathy Results Validation")
        print(f"{'='*60}")
        print(f"Results directory: {self.results_dir}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()

        # Check directory exists
        if not self.results_dir.exists():
            self.errors.append(f"Results directory does not exist: {self.results_dir}")
            return False, self._generate_summary()

        # Validate each phase
        for phase, experiments in self.EXPECTED_EXPERIMENTS.items():
            print(f"\nValidating {phase}...")
            print("-" * 40)

            for exp_name in experiments:
                self._validate_experiment(phase, exp_name)

        # Check for statistical test results
        print(f"\nValidating statistical tests...")
        print("-" * 40)
        self._validate_statistical_tests()

        # Check for visualization outputs
        print(f"\nValidating visualization outputs...")
        print("-" * 40)
        self._validate_visualizations()

        # Generate summary
        summary = self._generate_summary()

        # Print results
        self._print_validation_results(summary)

        # Return success status
        success = len(self.errors) == 0
        return success, summary

    def _validate_experiment(self, phase: str, exp_name: str):
        """Validate a single experiment's results."""
        # Look for result files
        exp_pattern = f"{exp_name}*.json"
        result_files = list(self.results_dir.glob(exp_pattern))

        if not result_files:
            self.errors.append(f"[{phase}] No results found for {exp_name} (pattern: {exp_pattern})")
            return

        # Validate each result file
        for result_file in result_files:
            self._validate_result_file(phase, exp_name, result_file)

    def _validate_result_file(self, phase: str, exp_name: str, filepath: Path):
        """Validate a single result file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"[{phase}/{exp_name}] Invalid JSON in {filepath.name}: {e}")
            return
        except Exception as e:
            self.errors.append(f"[{phase}/{exp_name}] Cannot read {filepath.name}: {e}")
            return

        # Determine result type and check required fields
        result_type = self._determine_result_type(exp_name)
        required_fields = self.REQUIRED_FIELDS.get(result_type, [])

        # Check for required fields
        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)

        if missing_fields:
            self.warnings.append(
                f"[{phase}/{exp_name}] Missing fields in {filepath.name}: {missing_fields}"
            )

        # Validate metric values
        for metric, value in data.items():
            if metric in self.METRIC_RANGES:
                min_val, max_val = self.METRIC_RANGES[metric]

                # Check for NaN or inf
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        self.errors.append(
                            f"[{phase}/{exp_name}] NaN value for {metric} in {filepath.name}"
                        )
                    elif np.isinf(value):
                        self.errors.append(
                            f"[{phase}/{exp_name}] Inf value for {metric} in {filepath.name}"
                        )
                    elif not (min_val <= value <= max_val):
                        self.errors.append(
                            f"[{phase}/{exp_name}] {metric}={value} outside valid range "
                            f"[{min_val}, {max_val}] in {filepath.name}"
                        )

                # Handle lists (e.g., confidence intervals)
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, (int, float)):
                            if np.isnan(v) or np.isinf(v):
                                self.errors.append(
                                    f"[{phase}/{exp_name}] NaN/Inf in {metric}[{i}] in {filepath.name}"
                                )

        # Store valid results
        if exp_name not in self.valid_results:
            self.valid_results[exp_name] = []
        self.valid_results[exp_name].append({
            'file': filepath.name,
            'data': data,
            'phase': phase
        })

        print(f"  ✓ {filepath.name}")

    def _determine_result_type(self, exp_name: str) -> str:
        """Determine the type of result based on experiment name."""
        if "probe" in exp_name:
            return "probe"
        elif "cross_model" in exp_name:
            return "cross_model"
        elif "baseline" in exp_name:
            return "classification"
        elif "statistical" in exp_name or "test" in exp_name:
            return "statistical"
        else:
            return "classification"

    def _validate_statistical_tests(self):
        """Validate that statistical tests have been run."""
        stat_files = list(self.results_dir.glob("statistical_*.json"))

        if not stat_files:
            self.errors.append("No statistical test results found")
            return

        expected_tests = [
            "probe_vs_baseline",
            "cross_model_transfer",
            "layer_comparison"
        ]

        found_tests = set()
        for filepath in stat_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Check for test metadata
                if 'test_name' in data:
                    found_tests.add(data['test_name'])

                # Validate statistical fields
                if 'p_value' not in data:
                    self.warnings.append(f"No p_value in {filepath.name}")
                elif data['p_value'] < 0 or data['p_value'] > 1:
                    self.errors.append(f"Invalid p_value in {filepath.name}: {data['p_value']}")

                if 'effect_size' in data and np.isnan(data['effect_size']):
                    self.errors.append(f"NaN effect_size in {filepath.name}")

                print(f"  ✓ {filepath.name}")

            except Exception as e:
                self.errors.append(f"Error reading statistical test {filepath.name}: {e}")

    def _validate_visualizations(self):
        """Check that visualization outputs exist."""
        figures_dir = self.results_dir.parent / 'figures'

        if not figures_dir.exists():
            self.warnings.append("No figures directory found")
            return

        expected_figures = [
            "attention_heatmap*.png",
            "probe_weights*.png",
            "accuracy_comparison*.png",
            "cross_model_transfer*.png"
        ]

        for pattern in expected_figures:
            files = list(figures_dir.glob(pattern))
            if not files:
                self.warnings.append(f"No visualization found matching: {pattern}")
            else:
                for f in files:
                    print(f"  ✓ {f.name}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_expected = sum(len(exps) for exps in self.EXPECTED_EXPERIMENTS.values())
        total_found = len(self.valid_results)

        # Calculate coverage by phase
        phase_coverage = {}
        for phase, experiments in self.EXPECTED_EXPERIMENTS.items():
            found = sum(1 for exp in experiments if exp in self.valid_results)
            phase_coverage[phase] = {
                'expected': len(experiments),
                'found': found,
                'percentage': (found / len(experiments) * 100) if experiments else 0
            }

        # Extract key metrics
        key_metrics = {}
        for exp_name, results in self.valid_results.items():
            if results:
                latest = results[-1]['data']

                # Extract primary metric
                if 'test_accuracy' in latest:
                    key_metrics[exp_name] = latest['test_accuracy']
                elif 'accuracy' in latest:
                    key_metrics[exp_name] = latest['accuracy']
                elif 'transfer_accuracy' in latest:
                    key_metrics[exp_name] = latest['transfer_accuracy']

        return {
            'timestamp': datetime.now().isoformat(),
            'results_dir': str(self.results_dir),
            'total_expected': total_expected,
            'total_found': total_found,
            'coverage_percentage': (total_found / total_expected * 100) if total_expected else 0,
            'phase_coverage': phase_coverage,
            'key_metrics': key_metrics,
            'num_errors': len(self.errors),
            'num_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'valid_results': list(self.valid_results.keys())
        }

    def _print_validation_results(self, summary: Dict[str, Any]):
        """Print validation results in a clear format."""
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")

        # Overall coverage
        print(f"\nOverall Coverage:")
        print(f"  Expected experiments: {summary['total_expected']}")
        print(f"  Found experiments: {summary['total_found']}")
        print(f"  Coverage: {summary['coverage_percentage']:.1f}%")

        # Phase-by-phase coverage
        print(f"\nPhase Coverage:")
        for phase, coverage in summary['phase_coverage'].items():
            status = "✓" if coverage['percentage'] == 100 else "⚠"
            print(f"  {status} {phase}: {coverage['found']}/{coverage['expected']} "
                  f"({coverage['percentage']:.0f}%)")

        # Key metrics
        if summary['key_metrics']:
            print(f"\nKey Metrics Summary:")
            for exp, metric in sorted(summary['key_metrics'].items()):
                print(f"  {exp}: {metric:.4f}")

        # Errors and warnings
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"  • {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  • {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")

        # Final status
        print(f"\n{'='*60}")
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                print("✅ VALIDATION PASSED - All results valid and complete!")
            else:
                print("✅ VALIDATION PASSED WITH WARNINGS - Review warnings above")
        else:
            print("❌ VALIDATION FAILED - Fix errors before paper submission!")
        print(f"{'='*60}\n")

    def save_summary(self, output_path: Optional[Path] = None):
        """Save validation summary to JSON file."""
        if output_path is None:
            output_path = self.results_dir / f"validation_summary_{datetime.now():%Y%m%d_%H%M%S}.json"

        summary = self._generate_summary()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Validation summary saved to: {output_path}")
        return output_path


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description="Validate Telepathy experimental results for completeness and correctness"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs/telepathy_results"),
        help="Directory containing experimental results (default: runs/telepathy_results)"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save validation summary to JSON file"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (strict mode)"
    )

    args = parser.parse_args()

    # Create validator
    validator = ResultsValidator(args.results_dir)

    # Run validation
    success, summary = validator.validate_all()

    # Save summary if requested
    if args.save_summary:
        validator.save_summary()

    # In strict mode, warnings are errors
    if args.strict and summary['num_warnings'] > 0:
        success = False
        print("\n❌ Strict mode: Treating warnings as errors")

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()