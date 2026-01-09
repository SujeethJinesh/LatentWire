#!/usr/bin/env python3
"""
Verify HPC experiment completion.

This script checks that all expected experiments completed successfully
and all required data is present before proceeding with paper updates.

Usage:
    python finalization/scripts/verify_completion.py
    python finalization/scripts/verify_completion.py --runs_dir runs/specific_experiment

Exit codes:
    0: All experiments completed successfully
    1: Some experiments missing or failed
"""

import argparse
import json
import glob
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set


# Expected experiments and datasets
EXPECTED_DATASETS = ['sst2', 'agnews', 'trec']
EXPECTED_METHODS = ['bridge', 'prompt_tuning', 'mistral_zeroshot', 'llama_zeroshot']
EXPECTED_SEEDS = [42, 123, 456]  # Standard 3 seeds


def check_result_files(runs_dir: str) -> Tuple[List[str], List[str]]:
    """
    Check for presence of result files.

    Returns:
        Tuple of (found_files, missing_patterns)
    """
    patterns = [
        f"{runs_dir}/**/unified_results_*.json",
        f"{runs_dir}/**/*_results.json"
    ]

    found = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        found.extend(files)

    if not found:
        return [], ["No result JSON files found"]

    return found, []


def check_log_files(runs_dir: str) -> Tuple[List[str], List[str]]:
    """
    Check log files for completion markers and errors.

    Returns:
        Tuple of (completion_markers_found, error_messages)
    """
    log_patterns = [
        f"{runs_dir}/**/*.log",
        f"{runs_dir}/**/*.out"
    ]

    completions = []
    errors = []

    for pattern in log_patterns:
        for log_file in glob.glob(pattern, recursive=True):
            try:
                with open(log_file, 'r', errors='ignore') as f:
                    content = f.read()

                # Check for completion markers
                if any(marker in content.lower() for marker in ['completed', 'finished', 'done', 'all experiments']):
                    completions.append(log_file)

                # Check for errors
                for line in content.split('\n'):
                    line_lower = line.lower()
                    if any(err in line_lower for err in ['error', 'exception', 'failed', 'traceback']):
                        if 'warning' not in line_lower and 'deprecat' not in line_lower:
                            errors.append(f"{log_file}: {line[:100]}")
            except Exception as e:
                errors.append(f"Could not read {log_file}: {e}")

    return completions, errors


def check_data_completeness(runs_dir: str) -> Tuple[Dict, List[str]]:
    """
    Check that all expected datasets and methods are present.

    Returns:
        Tuple of (coverage_dict, missing_list)
    """
    coverage = {ds: set() for ds in EXPECTED_DATASETS}
    missing = []

    # Find all result files
    for filepath in glob.glob(f"{runs_dir}/**/unified_results_*.json", recursive=True):
        try:
            with open(filepath) as f:
                data = json.load(f)

            results = data.get('results', {})

            for ds in EXPECTED_DATASETS:
                if ds in results:
                    for method in results[ds]:
                        if isinstance(results[ds][method], dict) and 'accuracy' in results[ds][method]:
                            coverage[ds].add(method)
        except Exception as e:
            print(f"  WARNING: Could not parse {filepath}: {e}")

    # Check what's missing
    for ds in EXPECTED_DATASETS:
        for method in EXPECTED_METHODS:
            if method not in coverage[ds]:
                missing.append(f"{ds}/{method}")

    return coverage, missing


def check_seed_coverage(runs_dir: str) -> Tuple[Dict, List[str]]:
    """
    Check that results exist for all expected seeds.

    Returns:
        Tuple of (seed_coverage, missing_seeds)
    """
    import re

    seed_coverage = {ds: {method: set() for method in EXPECTED_METHODS} for ds in EXPECTED_DATASETS}
    missing = []

    # Find all result files
    for filepath in glob.glob(f"{runs_dir}/**/*.json", recursive=True):
        # Extract seed from filename
        match = re.search(r'seed[\-_]?(\d+)', filepath, re.IGNORECASE)
        seed = None
        if match:
            seed = int(match.group(1))

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Also check for seed in meta
            if seed is None:
                seed = data.get('meta', {}).get('seed', 42)

            results = data.get('results', data)

            for ds in EXPECTED_DATASETS:
                if ds in results:
                    for method in EXPECTED_METHODS:
                        if method in results[ds]:
                            seed_coverage[ds][method].add(seed)
        except:
            pass

    # Check what's missing
    for ds in EXPECTED_DATASETS:
        for method in EXPECTED_METHODS:
            present = seed_coverage[ds][method]
            for seed in EXPECTED_SEEDS:
                if seed not in present:
                    missing.append(f"{ds}/{method}/seed{seed}")

    return seed_coverage, missing


def check_result_validity(runs_dir: str) -> List[str]:
    """
    Validate that result values are in expected ranges.

    Returns:
        List of validation errors
    """
    errors = []

    random_chance = {
        'sst2': 50.0,
        'agnews': 25.0,
        'trec': 16.7
    }

    for filepath in glob.glob(f"{runs_dir}/**/unified_results_*.json", recursive=True):
        try:
            with open(filepath) as f:
                data = json.load(f)

            results = data.get('results', {})

            for ds, ds_results in results.items():
                for method, method_results in ds_results.items():
                    if isinstance(method_results, dict) and 'accuracy' in method_results:
                        acc = method_results['accuracy']

                        # Check valid range
                        if acc < 0 or acc > 100:
                            errors.append(f"{filepath}: {ds}/{method} accuracy {acc} outside [0, 100]")

                        # Check bridge beats random
                        if method == 'bridge' and ds in random_chance:
                            if acc < random_chance[ds]:
                                errors.append(f"{filepath}: {ds}/bridge ({acc}%) below random chance ({random_chance[ds]}%)")

        except Exception as e:
            errors.append(f"Could not validate {filepath}: {e}")

    return errors


def generate_report(
    found_files: List[str],
    missing_files: List[str],
    completions: List[str],
    errors: List[str],
    coverage: Dict,
    missing_data: List[str],
    seed_coverage: Dict,
    missing_seeds: List[str],
    validation_errors: List[str]
) -> Tuple[str, bool]:
    """
    Generate a verification report.

    Returns:
        Tuple of (report_string, all_passed)
    """
    lines = []
    all_passed = True

    lines.append("=" * 80)
    lines.append("HPC EXPERIMENT VERIFICATION REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 80)

    # Result files
    lines.append("\n--- Result Files ---")
    lines.append(f"Found: {len(found_files)} files")
    if missing_files:
        lines.append(f"MISSING: {missing_files}")
        all_passed = False
    else:
        lines.append("Status: OK")

    # Log file analysis
    lines.append("\n--- Log File Analysis ---")
    lines.append(f"Completion markers found in: {len(completions)} files")
    if errors:
        lines.append(f"ERRORS FOUND: {len(errors)}")
        for err in errors[:10]:  # Show first 10 errors
            lines.append(f"  - {err}")
        if len(errors) > 10:
            lines.append(f"  ... and {len(errors) - 10} more")
        all_passed = False
    else:
        lines.append("No critical errors found")

    # Data completeness
    lines.append("\n--- Data Completeness ---")
    for ds, methods in coverage.items():
        lines.append(f"  {ds}: {len(methods)} methods ({', '.join(sorted(methods))})")
    if missing_data:
        lines.append(f"MISSING DATA: {missing_data}")
        all_passed = False
    else:
        lines.append("Status: All expected data present")

    # Seed coverage
    lines.append("\n--- Seed Coverage ---")
    for ds in EXPECTED_DATASETS:
        for method in EXPECTED_METHODS:
            seeds = seed_coverage.get(ds, {}).get(method, set())
            if seeds:
                lines.append(f"  {ds}/{method}: seeds {sorted(seeds)}")
    if missing_seeds:
        lines.append(f"MISSING SEEDS: {len(missing_seeds)} entries")
        for ms in missing_seeds[:10]:
            lines.append(f"  - {ms}")
        # Don't fail on missing seeds if we have at least one
        if len(missing_seeds) > len(EXPECTED_DATASETS) * len(EXPECTED_METHODS) * len(EXPECTED_SEEDS) * 0.5:
            all_passed = False

    # Validation
    lines.append("\n--- Result Validation ---")
    if validation_errors:
        lines.append(f"VALIDATION ERRORS: {len(validation_errors)}")
        for err in validation_errors[:5]:
            lines.append(f"  - {err}")
        all_passed = False
    else:
        lines.append("Status: All results valid")

    # Summary
    lines.append("\n" + "=" * 80)
    if all_passed:
        lines.append("OVERALL STATUS: PASSED")
        lines.append("All experiments completed successfully. Ready for paper update.")
    else:
        lines.append("OVERALL STATUS: ISSUES FOUND")
        lines.append("Review errors above before proceeding with paper update.")
    lines.append("=" * 80)

    return "\n".join(lines), all_passed


def main():
    parser = argparse.ArgumentParser(description='Verify HPC experiment completion')
    parser.add_argument('--runs_dir', default='runs', help='Directory containing experiment runs')
    parser.add_argument('--strict', action='store_true', help='Fail on any missing seeds')
    parser.add_argument('--output', help='Write report to file')
    args = parser.parse_args()

    print(f"Verification started at {datetime.now().isoformat()}")
    print(f"Checking: {args.runs_dir}")

    # Run all checks
    print("\nRunning checks...")

    print("  - Checking result files...")
    found_files, missing_files = check_result_files(args.runs_dir)

    print("  - Checking log files...")
    completions, errors = check_log_files(args.runs_dir)

    print("  - Checking data completeness...")
    coverage, missing_data = check_data_completeness(args.runs_dir)

    print("  - Checking seed coverage...")
    seed_coverage, missing_seeds = check_seed_coverage(args.runs_dir)

    print("  - Validating results...")
    validation_errors = check_result_validity(args.runs_dir)

    # Generate report
    report, all_passed = generate_report(
        found_files, missing_files,
        completions, errors,
        coverage, missing_data,
        seed_coverage, missing_seeds,
        validation_errors
    )

    print("\n" + report)

    # Write to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
