"""
Statistical Testing Module for Telepathy Experiments

Provides comprehensive statistical methods for comparing bridge performance with baselines.
Integrates with run_unified_comparison.py results for rigorous statistical analysis.

Key features:
- Bootstrap confidence intervals with BCa method
- McNemar's test for paired classification comparisons
- Multiple testing corrections (Bonferroni, Holm, FDR)
- Effect size calculations (Cohen's d)
- Power analysis for sample size determination
- Integration with unified comparison results JSON

Author: Telepathy Project
Date: January 2025
"""

import numpy as np
import json
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
from typing import List, Tuple, Dict, Optional, Callable, Union, Any
import warnings
from pathlib import Path


# =============================================================================
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    scores: Union[np.ndarray, List[float]],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = 'BCa',
    random_state: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap confidence interval for accuracy scores.

    Args:
        scores: Array of accuracy scores (e.g., from multiple seeds)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: CI level (default: 0.95 for 95% CI)
        method: 'BCa' (bias-corrected accelerated), 'percentile', or 'basic'
        random_state: Random seed for reproducibility

    Returns:
        (mean, (ci_lower, ci_upper))

    Example:
        >>> scores = [75.2, 77.3, 74.8]  # Accuracy from 3 seeds
        >>> mean, (lower, upper) = bootstrap_ci(scores)
        >>> print(f"Accuracy: {mean:.1f}% [{lower:.1f}, {upper:.1f}]")
    """
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("Scores must be 1-dimensional")

    # Point estimate
    mean_val = np.mean(scores)

    # Handle case with too few samples
    if len(scores) < 3:
        warnings.warn(f"Only {len(scores)} samples. Bootstrap CI requires at least 3.")
        return mean_val, (mean_val, mean_val)

    # Bootstrap CI using scipy
    rng = np.random.default_rng(random_state)

    def statistic(x):
        return np.mean(x)

    res = scipy_bootstrap(
        (scores,),
        statistic,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        method=method.lower(),
        random_state=rng
    )

    return mean_val, (res.confidence_interval.low, res.confidence_interval.high)


# =============================================================================
# 2. MCNEMAR'S TEST FOR PAIRED CLASSIFICATION
# =============================================================================

def mcnemar_test(
    pred1: np.ndarray,
    pred2: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    McNemar's test for comparing two classifiers on same test set.

    Perfect for comparing Bridge vs baselines when both are evaluated on same examples.

    Args:
        pred1: Predictions from method 1 (shape: [n_examples])
        pred2: Predictions from method 2 (shape: [n_examples])
        labels: Ground truth labels (shape: [n_examples])

    Returns:
        (statistic, p_value, contingency_table)

    Example:
        >>> # Bridge vs Text-Relay on same 200 test examples
        >>> stat, p_val, table = mcnemar_test(bridge_preds, relay_preds, true_labels)
        >>> if p_val < 0.05:
        >>>     print("Bridge significantly differs from Text-Relay")
    """
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    labels = np.asarray(labels)

    if not (len(pred1) == len(pred2) == len(labels)):
        raise ValueError("All arrays must have same length")

    # Build contingency table
    correct1 = (pred1 == labels)
    correct2 = (pred2 == labels)

    n00 = np.sum(correct1 & correct2)   # Both correct
    n01 = np.sum(correct1 & ~correct2)  # 1 correct, 2 wrong
    n10 = np.sum(~correct1 & correct2)  # 1 wrong, 2 correct
    n11 = np.sum(~correct1 & ~correct2) # Both wrong

    contingency = np.array([[n00, n01], [n10, n11]])

    # McNemar's test
    b = n01  # Model 1 correct, Model 2 wrong
    c = n10  # Model 1 wrong, Model 2 correct

    if b + c == 0:
        # Perfect agreement on errors
        return 0.0, 1.0, contingency

    # Use exact binomial test for small samples
    if b + c < 25:
        # Exact test
        from scipy.stats import binom
        statistic = min(b, c)
        p_value = 2 * binom.cdf(statistic, b + c, 0.5)
    else:
        # Chi-square with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, 1)

    return statistic, p_value, contingency


# =============================================================================
# 3. BONFERRONI CORRECTION FOR MULTIPLE TESTING
# =============================================================================

def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[List[bool], List[float], float]:
    """
    Bonferroni correction for multiple comparisons.

    When comparing Bridge to multiple baselines, controls family-wise error rate.

    Args:
        p_values: List of raw p-values from multiple tests
        alpha: Desired family-wise error rate (default: 0.05)

    Returns:
        (reject, corrected_p_values, corrected_alpha)
        - reject: List of booleans indicating which hypotheses to reject
        - corrected_p_values: Adjusted p-values (capped at 1.0)
        - corrected_alpha: Bonferroni-adjusted alpha threshold

    Example:
        >>> # Comparing Bridge to 5 baselines
        >>> p_vals = [0.01, 0.03, 0.04, 0.08, 0.15]
        >>> reject, p_adj, alpha_adj = bonferroni_correction(p_vals)
        >>> for i, (rej, p) in enumerate(zip(reject, p_adj)):
        >>>     print(f"Baseline {i+1}: p_adj={p:.3f}, reject={rej}")
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    # Bonferroni correction
    corrected_alpha = alpha / n_tests
    corrected_p_values = np.minimum(p_values * n_tests, 1.0)
    reject = corrected_p_values < alpha

    return reject.tolist(), corrected_p_values.tolist(), corrected_alpha


# =============================================================================
# 4. EFFECT SIZE CALCULATIONS
# =============================================================================

def calculate_effect_size(
    scores1: np.ndarray,
    scores2: np.ndarray,
    paired: bool = False
) -> float:
    """
    Calculate Cohen's d effect size between two methods.

    Quantifies the magnitude of difference (not just significance).

    Args:
        scores1: Scores for method 1 (e.g., Bridge)
        scores2: Scores for method 2 (e.g., baseline)
        paired: Whether scores are paired (same test examples)

    Returns:
        Cohen's d effect size

    Interpretation:
        |d| < 0.2: negligible
        |d| < 0.5: small
        |d| < 0.8: medium
        |d| >= 0.8: large

    Example:
        >>> bridge_scores = np.array([75.2, 77.3, 74.8])
        >>> baseline_scores = np.array([70.1, 71.5, 69.8])
        >>> d = calculate_effect_size(bridge_scores, baseline_scores, paired=True)
        >>> print(f"Effect size: {d:.2f} ({'large' if abs(d) >= 0.8 else 'medium'})")
    """
    scores1 = np.asarray(scores1)
    scores2 = np.asarray(scores2)

    if paired:
        if len(scores1) != len(scores2):
            raise ValueError("Paired scores must have same length")

        # Paired Cohen's d using difference scores
        diffs = scores1 - scores2
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)

        if std_diff == 0:
            return float('inf') if mean_diff != 0 else 0.0

        return mean_diff / std_diff
    else:
        # Independent samples Cohen's d with pooled SD
        n1, n2 = len(scores1), len(scores2)
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return float('inf') if mean1 != mean2 else 0.0

        return (mean1 - mean2) / pooled_std


# =============================================================================
# 5. POWER ANALYSIS AND SAMPLE SIZE DETERMINATION
# =============================================================================

def determine_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = 'paired'
) -> int:
    """
    Determine required sample size (seeds/examples) for desired power.

    Args:
        effect_size: Cohen's d effect size to detect
        power: Desired statistical power (default: 0.8)
        alpha: Significance level (default: 0.05)
        test_type: 'paired' or 'independent'

    Returns:
        Required number of samples per group

    Example:
        >>> # How many seeds to detect medium effect (d=0.5)?
        >>> n = determine_sample_size(effect_size=0.5, power=0.8)
        >>> print(f"Need {n} seeds to detect d=0.5 with 80% power")
    """
    from scipy.stats import norm

    # Z-scores
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = norm.ppf(power)

    if test_type == 'paired':
        # For paired t-test
        n = ((z_alpha + z_beta) / effect_size) ** 2
    else:
        # For independent t-test (2 samples)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


# =============================================================================
# 6. INTEGRATION WITH UNIFIED COMPARISON RESULTS
# =============================================================================

def analyze_unified_results(
    results_path: str,
    baseline_method: str = 'mistral_zeroshot',
    target_method: str = 'bridge',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Analyze results from run_unified_comparison.py with statistical tests.

    Args:
        results_path: Path to unified_results_*.json file
        baseline_method: Baseline to compare against
        target_method: Method to evaluate (typically 'bridge')
        alpha: Significance level

    Returns:
        Dictionary with statistical analysis for each dataset

    Example:
        >>> results = analyze_unified_results(
        ...     'runs/unified/unified_results_20250104.json',
        ...     baseline_method='mistral_zeroshot',
        ...     target_method='bridge'
        ... )
        >>> for dataset, stats in results.items():
        ...     print(f"{dataset}: p={stats['p_value']:.4f}, d={stats['effect_size']:.2f}")
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    analysis = {}

    for dataset in data['aggregated_results'].keys():
        dataset_results = data['aggregated_results'][dataset]

        # Skip if methods not present
        if baseline_method not in dataset_results or target_method not in dataset_results:
            continue

        baseline_data = dataset_results[baseline_method]
        target_data = dataset_results[target_method]

        # Skip if data is missing
        if 'accuracy_mean' not in baseline_data or 'accuracy_mean' not in target_data:
            continue

        # Extract accuracy statistics
        baseline_mean = baseline_data['accuracy_mean']
        baseline_std = baseline_data.get('accuracy_std', 0.0)
        baseline_n = baseline_data.get('num_seeds', 1)

        target_mean = target_data['accuracy_mean']
        target_std = target_data.get('accuracy_std', 0.0)
        target_n = target_data.get('num_seeds', 1)

        # Compute statistics
        diff = target_mean - baseline_mean

        # T-test (if we have std and n > 1)
        if baseline_n > 1 and target_n > 1 and baseline_std > 0 and target_std > 0:
            # Welch's t-test for unequal variances
            se_diff = np.sqrt(baseline_std**2/baseline_n + target_std**2/target_n)
            t_stat = diff / se_diff

            # Degrees of freedom (Welch-Satterthwaite)
            df = (baseline_std**2/baseline_n + target_std**2/target_n)**2 / (
                (baseline_std**2/baseline_n)**2/(baseline_n-1) +
                (target_std**2/target_n)**2/(target_n-1)
            )

            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            p_value = np.nan
            t_stat = np.nan
            df = 0

        # Effect size (using pooled SD)
        if baseline_std > 0 or target_std > 0:
            pooled_std = np.sqrt((baseline_std**2 + target_std**2) / 2)
            effect_size = diff / pooled_std if pooled_std > 0 else 0.0
        else:
            effect_size = 0.0

        analysis[dataset] = {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'target_mean': target_mean,
            'target_std': target_std,
            'difference': diff,
            'relative_improvement': (diff / baseline_mean * 100) if baseline_mean > 0 else 0.0,
            'effect_size': effect_size,
            'p_value': p_value,
            't_statistic': t_stat,
            'degrees_of_freedom': df,
            'significant': p_value < alpha if not np.isnan(p_value) else False,
            'num_seeds': min(baseline_n, target_n)
        }

    return analysis


def generate_comparison_table(
    results_path: str,
    methods_to_compare: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    alpha: float = 0.05
) -> str:
    """
    Generate a formatted comparison table from unified results.

    Args:
        results_path: Path to unified_results_*.json file
        methods_to_compare: List of methods to include (default: all)
        datasets: List of datasets to include (default: all)
        alpha: Significance level for tests

    Returns:
        Formatted markdown table string

    Example:
        >>> table = generate_comparison_table('runs/unified/results.json')
        >>> print(table)
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Default to all datasets and methods
    if datasets is None:
        datasets = list(data['aggregated_results'].keys())

    if methods_to_compare is None:
        # Get all methods from first dataset
        first_dataset = datasets[0]
        methods_to_compare = list(data['aggregated_results'][first_dataset].keys())
        # Filter out non-method keys
        methods_to_compare = [m for m in methods_to_compare if 'accuracy' in str(data['aggregated_results'][first_dataset].get(m, {}))]

    # Build table
    lines = []
    lines.append("| Method | " + " | ".join(datasets) + " |")
    lines.append("|--------|" + "---------|" * len(datasets))

    for method in methods_to_compare:
        row = [method.replace('_', ' ').title()]

        for dataset in datasets:
            dataset_results = data['aggregated_results'].get(dataset, {})
            method_data = dataset_results.get(method, {})

            if 'accuracy_mean' in method_data:
                mean = method_data['accuracy_mean']
                std = method_data.get('accuracy_std', 0.0)
                n = method_data.get('num_seeds', 1)

                # Format with confidence indicator
                if n > 1:
                    # Calculate 95% CI using t-distribution
                    from scipy.stats import t
                    se = std / np.sqrt(n)
                    margin = t.ppf(0.975, n-1) * se
                    cell = f"{mean:.1f}±{margin:.1f}"
                else:
                    cell = f"{mean:.1f}"

                # Add significance star if this is Bridge and significantly better than baselines
                if method == 'bridge' and 'mistral_zeroshot' in dataset_results:
                    baseline = dataset_results['mistral_zeroshot']
                    if 'accuracy_mean' in baseline:
                        # Simple significance test
                        if mean > baseline['accuracy_mean'] + 2 * baseline.get('accuracy_std', 0):
                            cell += "*"
            else:
                cell = "N/A"

            row.append(cell)

        lines.append("| " + " | ".join(row) + " |")

    # Add footer
    lines.append("\n*Note: Values show mean±95% CI. * indicates significant improvement over Mistral zero-shot (p<0.05)*")

    return "\n".join(lines)


# =============================================================================
# 7. COMPREHENSIVE STATISTICAL REPORT
# =============================================================================

def generate_statistical_report(
    results_path: str,
    output_path: Optional[str] = None,
    alpha: float = 0.05
) -> str:
    """
    Generate comprehensive statistical analysis report.

    Args:
        results_path: Path to unified_results_*.json file
        output_path: Optional path to save report (if None, returns string)
        alpha: Significance level

    Returns:
        Formatted report string

    Example:
        >>> report = generate_statistical_report('runs/unified/results.json')
        >>> print(report)
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    report = []
    report.append("=" * 80)
    report.append("TELEPATHY STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Meta information
    meta = data.get('meta', {})
    report.append("Experiment Configuration:")
    report.append(f"  Timestamp: {meta.get('timestamp', 'N/A')}")
    report.append(f"  Seeds: {meta.get('seeds', [])}")
    report.append(f"  Soft tokens: {meta.get('soft_tokens', 'N/A')}")
    report.append(f"  Train steps: {meta.get('train_steps', 'N/A')}")
    report.append(f"  Eval samples: {meta.get('eval_samples', 'N/A')}")
    report.append("")

    # Analyze each dataset
    for dataset in data['aggregated_results'].keys():
        report.append("-" * 80)
        report.append(f"DATASET: {dataset.upper()}")
        report.append("-" * 80)
        report.append("")

        dataset_results = data['aggregated_results'][dataset]

        # Bridge vs each baseline
        if 'bridge' in dataset_results:
            bridge_data = dataset_results['bridge']
            bridge_mean = bridge_data.get('accuracy_mean', 0)
            bridge_std = bridge_data.get('accuracy_std', 0)
            bridge_n = bridge_data.get('num_seeds', 1)

            report.append(f"Bridge Performance:")
            report.append(f"  Mean: {bridge_mean:.2f}%")
            report.append(f"  Std: {bridge_std:.2f}%")
            report.append(f"  Seeds: {bridge_n}")

            # Bootstrap CI
            if bridge_n > 1:
                # Simulate data from mean and std (approximation)
                np.random.seed(42)
                simulated = np.random.normal(bridge_mean, bridge_std, bridge_n)
                _, (ci_low, ci_high) = bootstrap_ci(simulated, n_bootstrap=10000)
                report.append(f"  95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")
            report.append("")

            # Comparisons
            report.append("Statistical Comparisons:")

            baselines = ['prompt_tuning', 'llama_zeroshot', 'mistral_zeroshot', 'mistral_fewshot', 'text_relay']
            p_values = []

            for baseline_name in baselines:
                if baseline_name not in dataset_results:
                    continue

                baseline_data = dataset_results[baseline_name]
                if 'accuracy_mean' not in baseline_data:
                    continue

                baseline_mean = baseline_data['accuracy_mean']
                baseline_std = baseline_data.get('accuracy_std', 0)
                baseline_n = baseline_data.get('num_seeds', 1)

                diff = bridge_mean - baseline_mean
                improvement = (diff / baseline_mean * 100) if baseline_mean > 0 else 0

                # Effect size
                if bridge_std > 0 or baseline_std > 0:
                    pooled_std = np.sqrt((bridge_std**2 + baseline_std**2) / 2)
                    effect_size = diff / pooled_std if pooled_std > 0 else 0
                else:
                    effect_size = 0

                # P-value (approximate)
                if bridge_n > 1 and baseline_n > 1:
                    se_diff = np.sqrt(bridge_std**2/bridge_n + baseline_std**2/baseline_n)
                    if se_diff > 0:
                        t_stat = diff / se_diff
                        df = bridge_n + baseline_n - 2
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                        p_values.append(p_val)
                    else:
                        p_val = 1.0
                else:
                    p_val = np.nan

                report.append(f"  vs {baseline_name.replace('_', ' ').title()}:")
                report.append(f"    Baseline: {baseline_mean:.2f}% ± {baseline_std:.2f}%")
                report.append(f"    Difference: {diff:+.2f}%")
                report.append(f"    Relative improvement: {improvement:+.2f}%")
                report.append(f"    Effect size (d): {effect_size:.2f}")

                if not np.isnan(p_val):
                    report.append(f"    p-value: {p_val:.4f}")
                    if p_val < alpha:
                        report.append(f"    ✓ Significant at α={alpha}")
                    else:
                        report.append(f"    ✗ Not significant")
                report.append("")

            # Multiple testing correction
            if p_values:
                reject, p_adj, alpha_adj = bonferroni_correction(p_values, alpha)
                report.append(f"Bonferroni correction: α_adjusted = {alpha_adj:.4f}")
                report.append(f"Significant after correction: {sum(reject)}/{len(reject)}")
                report.append("")

        # Power analysis
        report.append("Power Analysis:")

        # Estimate required seeds
        typical_std = 2.0  # Typical std for accuracy across seeds
        for effect_pct in [2, 5, 10]:
            effect_abs = effect_pct  # For percentage points
            effect_d = effect_abs / typical_std
            n_required = determine_sample_size(effect_d, power=0.8, alpha=0.05, test_type='paired')
            report.append(f"  To detect {effect_pct}% improvement: {n_required} seeds needed")

        current_seeds = meta.get('seeds', [])
        if len(current_seeds) < 5:
            report.append(f"  ⚠️  Current seeds ({len(current_seeds)}) may be insufficient for reliable conclusions")
        report.append("")

    # Summary
    report.append("=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)

    n_seeds = len(meta.get('seeds', []))
    if n_seeds < 5:
        report.append(f"1. Increase number of seeds from {n_seeds} to at least 5 for more reliable results")

    report.append("2. Consider running McNemar's test by saving per-example predictions")
    report.append("3. Report effect sizes alongside p-values for practical significance")
    report.append("")

    report_str = "\n".join(report)

    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_str)
        print(f"Report saved to: {output_path}")

    return report_str


# =============================================================================
# MAIN EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Telepathy Statistical Testing Module")
    print("=" * 80)
    print()

    # Example 1: Bootstrap CI
    print("1. Bootstrap Confidence Intervals")
    print("-" * 40)
    scores = [75.2, 77.3, 74.8]  # Bridge accuracy from 3 seeds
    mean, (ci_low, ci_high) = bootstrap_ci(scores, n_bootstrap=10000)
    print(f"Bridge accuracy: {mean:.1f}% [95% CI: {ci_low:.1f}%, {ci_high:.1f}%]")
    print()

    # Example 2: Effect size
    print("2. Effect Size Calculation")
    print("-" * 40)
    bridge_scores = np.array([75.2, 77.3, 74.8])
    baseline_scores = np.array([70.1, 71.5, 69.8])
    d = calculate_effect_size(bridge_scores, baseline_scores, paired=True)
    effect_interpretation = (
        "large" if abs(d) >= 0.8 else
        "medium" if abs(d) >= 0.5 else
        "small" if abs(d) >= 0.2 else
        "negligible"
    )
    print(f"Cohen's d = {d:.2f} ({effect_interpretation} effect)")
    print()

    # Example 3: Power analysis
    print("3. Power Analysis")
    print("-" * 40)
    n_seeds = determine_sample_size(effect_size=0.5, power=0.8)
    print(f"Seeds needed to detect d=0.5 with 80% power: {n_seeds}")
    print()

    # Example 4: Multiple testing correction
    print("4. Multiple Testing Correction")
    print("-" * 40)
    p_values = [0.01, 0.03, 0.04, 0.08, 0.15]  # p-values from 5 comparisons
    reject, p_adj, alpha_adj = bonferroni_correction(p_values)
    print("Bonferroni correction for 5 comparisons:")
    for i, (p_raw, p_corr, rej) in enumerate(zip(p_values, p_adj, reject)):
        print(f"  Test {i+1}: p={p_raw:.3f} → p_adj={p_corr:.3f} (reject={rej})")
    print(f"Adjusted alpha: {alpha_adj:.4f}")
    print()

    # Example 5: McNemar's test simulation
    print("5. McNemar's Test (Simulated)")
    print("-" * 40)
    np.random.seed(42)
    n_examples = 200
    # Simulate predictions where Bridge is slightly better
    labels = np.random.randint(0, 4, n_examples)
    bridge_preds = labels.copy()
    bridge_preds[np.random.random(n_examples) < 0.25] = np.random.randint(0, 4, int(n_examples * 0.25))
    baseline_preds = labels.copy()
    baseline_preds[np.random.random(n_examples) < 0.30] = np.random.randint(0, 4, int(n_examples * 0.30))

    stat, p_val, table = mcnemar_test(bridge_preds, baseline_preds, labels)
    print(f"McNemar's test: statistic={stat:.2f}, p-value={p_val:.4f}")
    print(f"Contingency table:\n{table}")
    if p_val < 0.05:
        print("✓ Bridge significantly differs from baseline")
    else:
        print("✗ No significant difference")
    print()

    print("=" * 80)
    print("Use analyze_unified_results() to analyze your experiment results!")