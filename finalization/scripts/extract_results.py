#!/usr/bin/env python3
"""
Data extraction script for HPC experimental results.

This script loads results from HPC runs and extracts key metrics
for paper tables. It handles both single-seed and multi-seed experiments.

Usage:
    python finalization/scripts/extract_results.py
    python finalization/scripts/extract_results.py --runs_dir runs/specific_experiment

Output:
    - Prints extracted metrics to stdout
    - Optionally saves to JSON for further processing
"""

import json
import numpy as np
import glob
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scripts.statistical_testing import (
        bootstrap_ci,
        paired_ttest,
        cohens_d_paired,
        p_value_to_stars,
        aggregate_multiseed_results
    )
    STATS_AVAILABLE = True
except ImportError:
    print("WARNING: Statistical testing module not available. Some features disabled.")
    STATS_AVAILABLE = False


def load_latest_results(runs_dir: str = "runs") -> Tuple[Dict, str]:
    """
    Load the most recent unified results file.

    Args:
        runs_dir: Directory containing experiment runs

    Returns:
        Tuple of (data dict, filepath string)
    """
    patterns = [
        f"{runs_dir}/**/unified_results_*.json",
        f"{runs_dir}/**/results.json",
        f"{runs_dir}/**/*_results.json"
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    if not all_files:
        raise FileNotFoundError(f"No results found in {runs_dir}")

    # Sort by modification time
    all_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    latest = all_files[0]

    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f), latest


def load_multiseed_results(runs_dir: str = "runs") -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Load results from multiple random seeds.

    Returns:
        Nested dict: {dataset: {method: {seed: accuracy}}}
    """
    results = {}

    # Find all result files
    patterns = [
        f"{runs_dir}/**/seed*/results.json",
        f"{runs_dir}/**/*_seed*_results.json",
        f"{runs_dir}/**/unified_results_*.json"
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    print(f"Found {len(all_files)} result files")

    for filepath in all_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  WARNING: Could not parse {filepath}")
            continue

        # Extract seed from filename or metadata
        seed = data.get('meta', {}).get('seed', 42)

        # Try to extract seed from filename
        match = re.search(r'seed[\-_]?(\d+)', filepath, re.IGNORECASE)
        if match:
            seed = int(match.group(1))

        # Extract dataset results
        result_data = data.get('results', data)  # Handle both formats

        for dataset, ds_results in result_data.items():
            if dataset in ['meta', 'comparison_table', 'config']:
                continue

            if dataset not in results:
                results[dataset] = {}

            for method, method_results in ds_results.items():
                if isinstance(method_results, dict) and 'accuracy' in method_results:
                    if method not in results[dataset]:
                        results[dataset][method] = {}
                    results[dataset][method][seed] = method_results['accuracy']
                elif isinstance(method_results, (int, float)):
                    # Handle direct accuracy values (like random_chance)
                    if method not in results[dataset]:
                        results[dataset][method] = {}
                    results[dataset][method][seed] = float(method_results)

    return results


def extract_key_metrics(data: Dict) -> Dict:
    """
    Extract key metrics for paper tables from a single results file.

    Args:
        data: Loaded JSON data

    Returns:
        Dict with extracted metrics per dataset and method
    """
    metrics = {}

    result_data = data.get('results', data)

    for dataset, ds_results in result_data.items():
        if dataset in ['meta', 'comparison_table', 'config']:
            continue

        metrics[dataset] = {}

        for method, method_results in ds_results.items():
            if isinstance(method_results, dict):
                metrics[dataset][method] = {
                    'accuracy': method_results.get('accuracy', 0),
                    'latency_ms': method_results.get('latency_ms'),
                    'correct': method_results.get('correct', 0),
                    'total': method_results.get('total', 0),
                    'f1': method_results.get('f1'),
                    'train_loss': method_results.get('train_info', {}).get('final_loss'),
                }
            elif isinstance(method_results, (int, float)):
                metrics[dataset][method] = {'accuracy': float(method_results)}

    return metrics


def aggregate_across_seeds(
    multiseed_data: Dict,
    n_bootstrap: int = 10000
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute mean, std, and 95% CI across random seeds.

    Args:
        multiseed_data: Output from load_multiseed_results()
        n_bootstrap: Number of bootstrap resamples for CI

    Returns:
        Aggregated stats: {dataset: {method: {mean, std, ci_lower, ci_upper, n, seeds}}}
    """
    aggregated = {}

    for dataset, ds_results in multiseed_data.items():
        aggregated[dataset] = {}

        for method, seed_scores in ds_results.items():
            scores = np.array(list(seed_scores.values()))
            n = len(scores)

            if n < 2:
                # Cannot compute std with < 2 seeds
                aggregated[dataset][method] = {
                    'mean': float(np.mean(scores)),
                    'std': 0.0,
                    'ci_lower': float(np.mean(scores)),
                    'ci_upper': float(np.mean(scores)),
                    'n': n,
                    'seeds': list(seed_scores.keys()),
                    'values': scores.tolist()
                }
                continue

            mean_val = float(np.mean(scores))
            std_val = float(np.std(scores, ddof=1))

            # Bootstrap CI if available
            if STATS_AVAILABLE and n >= 3:
                try:
                    _, (ci_lower, ci_upper) = bootstrap_ci(
                        scores,
                        confidence_level=0.95,
                        n_resamples=n_bootstrap,
                        method='percentile',  # Use percentile for small n
                        random_state=42
                    )
                except Exception as e:
                    print(f"  WARNING: Bootstrap failed for {dataset}/{method}: {e}")
                    ci_lower = mean_val - 1.96 * std_val / np.sqrt(n)
                    ci_upper = mean_val + 1.96 * std_val / np.sqrt(n)
            else:
                # Use t-distribution approximation
                from scipy import stats
                t_val = stats.t.ppf(0.975, df=n-1)
                se = std_val / np.sqrt(n)
                ci_lower = mean_val - t_val * se
                ci_upper = mean_val + t_val * se

            aggregated[dataset][method] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n': n,
                'seeds': list(seed_scores.keys()),
                'values': scores.tolist()
            }

    return aggregated


def calculate_significance(
    multiseed_data: Dict,
    baseline_method: str = 'mistral_zeroshot',
    alpha: float = 0.05
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Perform statistical significance tests comparing all methods to baseline.

    Args:
        multiseed_data: Output from load_multiseed_results()
        baseline_method: Method to use as baseline for comparisons
        alpha: Significance threshold

    Returns:
        Significance results: {dataset: {method: {p_value, significant, effect_size, stars}}}
    """
    if not STATS_AVAILABLE:
        print("WARNING: Statistical testing not available")
        return {}

    significance = {}

    for dataset, ds_results in multiseed_data.items():
        significance[dataset] = {}

        if baseline_method not in ds_results:
            print(f"WARNING: Baseline {baseline_method} not found in {dataset}")
            continue

        baseline_seeds = ds_results[baseline_method]

        for method, seed_scores in ds_results.items():
            if method == baseline_method:
                continue

            # Find common seeds for paired test
            common_seeds = set(baseline_seeds.keys()) & set(seed_scores.keys())

            if len(common_seeds) >= 2:
                baseline_paired = np.array([baseline_seeds[s] for s in sorted(common_seeds)])
                method_paired = np.array([seed_scores[s] for s in sorted(common_seeds)])

                diff, p_val, stats = paired_ttest(method_paired, baseline_paired)
                d = cohens_d_paired(method_paired, baseline_paired)

                significance[dataset][method] = {
                    'mean_diff': float(diff),
                    'p_value': float(p_val),
                    'significant': p_val < alpha,
                    'cohens_d': float(d),
                    'stars': p_value_to_stars(p_val),
                    'n_pairs': len(common_seeds),
                    't_statistic': float(stats['t_statistic']),
                    'df': int(stats['degrees_of_freedom'])
                }
            else:
                # Not enough paired data
                baseline_scores = np.array(list(baseline_seeds.values()))
                method_scores = np.array(list(seed_scores.values()))

                significance[dataset][method] = {
                    'mean_diff': float(np.mean(method_scores) - np.mean(baseline_scores)),
                    'p_value': None,
                    'significant': None,
                    'cohens_d': None,
                    'stars': '',
                    'n_pairs': 0,
                    'note': 'Insufficient paired data for significance test'
                }

    return significance


def format_for_latex(
    aggregated: Dict,
    significance: Dict,
    decimal_places: int = 1
) -> Dict[str, str]:
    """
    Generate LaTeX-ready formatted strings.

    Args:
        aggregated: Output from aggregate_across_seeds()
        significance: Output from calculate_significance()
        decimal_places: Number of decimal places for formatting

    Returns:
        Dict mapping placeholder names to formatted values
    """
    latex_values = {}

    for dataset, ds_agg in aggregated.items():
        ds_upper = dataset.upper().replace('-', '').replace('_', '')

        for method, stats in ds_agg.items():
            method_upper = method.upper().replace('_', '').replace('-', '')
            prefix = f"{ds_upper}_{method_upper}"

            mean = stats['mean']
            std = stats['std']
            n = stats['n']

            # Individual values
            latex_values[f'{prefix}_ACC'] = f"{mean:.{decimal_places}f}"
            latex_values[f'{prefix}_STD'] = f"{std:.{decimal_places}f}"
            latex_values[f'{prefix}_N'] = str(n)

            # Get significance info
            sig = significance.get(dataset, {}).get(method, {})
            stars = sig.get('stars', '')

            # Full formatted value
            if n > 1 and std > 0:
                if stars:
                    latex_values[f'{prefix}_FULL'] = f"{mean:.{decimal_places}f} $\\pm$ {std:.{decimal_places}f}$^{{{stars}}}$"
                else:
                    latex_values[f'{prefix}_FULL'] = f"{mean:.{decimal_places}f} $\\pm$ {std:.{decimal_places}f}"
            else:
                if stars:
                    latex_values[f'{prefix}_FULL'] = f"{mean:.{decimal_places}f}$^{{{stars}}}$"
                else:
                    latex_values[f'{prefix}_FULL'] = f"{mean:.{decimal_places}f}"

            # 95% CI
            latex_values[f'{prefix}_CI'] = f"[{stats['ci_lower']:.{decimal_places}f}, {stats['ci_upper']:.{decimal_places}f}]"

    return latex_values


def sanity_check_results(aggregated: Dict) -> List[str]:
    """
    Perform sanity checks on extracted results.

    Args:
        aggregated: Output from aggregate_across_seeds()

    Returns:
        List of warning messages
    """
    warnings = []

    # Define expected random chance values
    random_chance = {
        'sst2': 50.0,
        'agnews': 25.0,
        'trec': 16.7,
        'banking77': 1.3,  # 1/77 classes
    }

    for dataset, methods in aggregated.items():
        for method, stats in methods.items():
            acc = stats.get('mean', 0)
            std = stats.get('std', 0)
            n = stats.get('n', 1)
            ci_low = stats.get('ci_lower', acc)
            ci_high = stats.get('ci_upper', acc)

            # Check 1: Valid accuracy range
            if acc < 0 or acc > 100:
                warnings.append(f"INVALID: {dataset}/{method} accuracy = {acc:.1f} (outside 0-100)")

            # Check 2: Bridge should beat random chance
            if method == 'bridge' and dataset in random_chance:
                chance = random_chance[dataset]
                if acc < chance * 1.1:  # At least 10% above random
                    warnings.append(f"WARNING: {dataset}/bridge ({acc:.1f}%) barely beats random ({chance}%)")

            # Check 3: Reasonable std
            if n > 1 and std == 0:
                warnings.append(f"WARNING: {dataset}/{method} has 0 std with {n} seeds")
            if std > 20:
                warnings.append(f"WARNING: {dataset}/{method} has very high std = {std:.1f}")

            # Check 4: CI contains mean
            if acc < ci_low or acc > ci_high:
                warnings.append(f"ERROR: {dataset}/{method} mean ({acc:.1f}) not in CI [{ci_low:.1f}, {ci_high:.1f}]")

    return warnings


def print_summary(aggregated: Dict, significance: Dict):
    """Print a formatted summary of results."""

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for dataset in sorted(aggregated.keys()):
        print(f"\n{dataset.upper()}")
        print("-" * 40)

        ds_agg = aggregated[dataset]
        ds_sig = significance.get(dataset, {})

        # Sort methods for consistent output
        methods = sorted(ds_agg.keys())

        for method in methods:
            stats = ds_agg[method]
            sig = ds_sig.get(method, {})

            mean = stats['mean']
            std = stats['std']
            n = stats['n']
            stars = sig.get('stars', '')

            if n > 1:
                print(f"  {method:20s}: {mean:5.1f}% +/- {std:4.1f}% (n={n}) {stars}")
            else:
                print(f"  {method:20s}: {mean:5.1f}% (n={n}) {stars}")


def main():
    parser = argparse.ArgumentParser(description='Extract results from HPC experiments')
    parser.add_argument('--runs_dir', default='runs', help='Directory containing experiment runs')
    parser.add_argument('--baseline', default='mistral_zeroshot', help='Baseline method for significance tests')
    parser.add_argument('--output', help='Output JSON file (optional)')
    parser.add_argument('--latex', action='store_true', help='Print LaTeX placeholder values')
    args = parser.parse_args()

    print(f"Extraction started at {datetime.now().isoformat()}")
    print(f"Runs directory: {args.runs_dir}")

    # Load multi-seed results
    print("\n--- Loading multi-seed results ---")
    try:
        multiseed = load_multiseed_results(args.runs_dir)
        print(f"Loaded data for {len(multiseed)} datasets")
        for ds, methods in multiseed.items():
            print(f"  {ds}: {len(methods)} methods, seeds vary by method")
    except Exception as e:
        print(f"ERROR loading multi-seed results: {e}")

        # Fallback to single result file
        print("\n--- Falling back to latest single result file ---")
        data, filepath = load_latest_results(args.runs_dir)
        metrics = extract_key_metrics(data)

        print("\nExtracted metrics:")
        for ds, ds_metrics in metrics.items():
            print(f"\n{ds.upper()}:")
            for method, m in ds_metrics.items():
                print(f"  {method}: {m.get('accuracy', 0):.1f}%")
        return

    # Aggregate across seeds
    print("\n--- Aggregating across seeds ---")
    aggregated = aggregate_across_seeds(multiseed)

    # Calculate significance
    print("\n--- Calculating statistical significance ---")
    significance = calculate_significance(multiseed, baseline_method=args.baseline)

    # Run sanity checks
    print("\n--- Running sanity checks ---")
    warnings = sanity_check_results(aggregated)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("All sanity checks passed!")

    # Print summary
    print_summary(aggregated, significance)

    # Print LaTeX values if requested
    if args.latex:
        print("\n" + "=" * 80)
        print("LATEX PLACEHOLDER VALUES")
        print("=" * 80)
        latex_values = format_for_latex(aggregated, significance)
        for key in sorted(latex_values.keys()):
            print(f"  {key}: {latex_values[key]}")

    # Save to JSON if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'runs_dir': args.runs_dir,
            'baseline': args.baseline,
            'aggregated': aggregated,
            'significance': significance,
            'warnings': warnings
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
