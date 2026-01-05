"""
Statistical Testing Utilities for Machine Learning Experiments

Implements rigorous statistical methods for comparing ML models:
- Bootstrap confidence intervals (95% CI)
- Paired bootstrap test for comparing methods
- McNemar's test for classification comparison
- Multiple comparison corrections (Bonferroni, Benjamini-Hochberg FDR)
- Power analysis for determining required sample sizes

References:
- Dietterich (1998): "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
- Colas et al. (2018): "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments"
- Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
"""

import numpy as np
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Dict, Optional, Callable
import warnings


# =============================================================================
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    method: str = 'BCa',
    random_state: Optional[int] = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        statistic: Function to compute (default: mean). Can be mean, median, std, etc.
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        n_resamples: Number of bootstrap resamples (default: 10000)
        method: 'percentile', 'basic', or 'BCa' (bias-corrected and accelerated, recommended)
        random_state: Random seed for reproducibility

    Returns:
        point_estimate: The statistic computed on original data
        ci: Tuple of (lower_bound, upper_bound)

    Example:
        >>> scores = np.array([0.7, 0.72, 0.68, 0.75, 0.71])
        >>> mean_val, (lower, upper) = bootstrap_ci(scores, statistic=np.mean)
        >>> print(f"Mean: {mean_val:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Data must be 1-dimensional")

    # Compute point estimate
    point_estimate = statistic(data)

    # Use scipy.stats.bootstrap for CI computation
    rng = np.random.default_rng(random_state)

    # Wrap data for scipy.stats.bootstrap
    res = scipy_bootstrap(
        (data,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method.lower(),
        random_state=rng
    )

    ci = (res.confidence_interval.low, res.confidence_interval.high)

    return point_estimate, ci


def bootstrap_ci_multiple_metrics(
    results: Dict[str, np.ndarray],
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    method: str = 'BCa',
    random_state: Optional[int] = None
) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    """
    Compute bootstrap CIs for multiple metrics at once.

    Args:
        results: Dictionary mapping metric names to 1D arrays of scores
        confidence_level: Confidence level (default: 0.95)
        n_resamples: Number of bootstrap resamples
        method: Bootstrap method ('BCa' recommended)
        random_state: Random seed

    Returns:
        Dictionary mapping metric names to (point_estimate, (lower, upper))

    Example:
        >>> results = {
        ...     'f1': np.array([0.7, 0.72, 0.68, 0.75, 0.71]),
        ...     'em': np.array([0.5, 0.52, 0.48, 0.55, 0.51])
        ... }
        >>> cis = bootstrap_ci_multiple_metrics(results)
        >>> for metric, (val, (lo, hi)) in cis.items():
        ...     print(f"{metric}: {val:.3f} [{lo:.3f}, {hi:.3f}]")
    """
    output = {}
    for metric_name, scores in results.items():
        output[metric_name] = bootstrap_ci(
            scores,
            statistic=np.mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method=method,
            random_state=random_state
        )
    return output


# =============================================================================
# 2. PAIRED BOOTSTRAP TEST
# =============================================================================

def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10000,
    random_state: Optional[int] = None,
    alternative: str = 'two-sided'
) -> Tuple[float, float, Dict[str, float]]:
    """
    Paired bootstrap test for comparing two methods on the same test set.

    Tests whether method A significantly outperforms method B by resampling
    test examples with replacement and computing differences in means.

    Args:
        scores_a: Array of scores for method A (shape: [n_examples])
        scores_b: Array of scores for method B (shape: [n_examples])
        n_resamples: Number of bootstrap resamples (default: 10000)
        random_state: Random seed
        alternative: 'two-sided', 'greater' (A > B), or 'less' (A < B)

    Returns:
        observed_diff: Mean difference (mean_a - mean_b)
        p_value: Two-sided p-value
        stats: Dictionary with additional statistics

    Notes:
        - Null hypothesis: The two methods have the same expected performance
        - p-value < 0.05 indicates significant difference at α=0.05
        - Requires at least N=20 samples for reliable results

    Example:
        >>> method_a_scores = np.array([0.7, 0.72, 0.68, 0.75, 0.71] * 5)  # 25 samples
        >>> method_b_scores = np.array([0.65, 0.67, 0.63, 0.70, 0.66] * 5)
        >>> diff, p_val, stats = paired_bootstrap_test(method_a_scores, method_b_scores)
        >>> print(f"Difference: {diff:.3f}, p-value: {p_val:.4f}")
        >>> if p_val < 0.05:
        ...     print("Statistically significant difference!")
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length")

    if len(scores_a) < 20:
        warnings.warn(
            f"Only {len(scores_a)} samples provided. "
            "Bootstrap test requires at least N=20 samples for reliable results."
        )

    n = len(scores_a)

    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Bootstrap resampling
    rng = np.random.default_rng(random_state)
    bootstrap_diffs = np.zeros(n_resamples)

    for i in range(n_resamples):
        # Resample indices with replacement
        indices = rng.choice(n, size=n, replace=True)

        # Compute difference in means for this resample
        bootstrap_diffs[i] = np.mean(scores_a[indices]) - np.mean(scores_b[indices])

    # Compute p-value based on alternative hypothesis
    if alternative == 'two-sided':
        # Two-sided: proportion of |bootstrap_diff| >= |observed_diff|
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    elif alternative == 'greater':
        # One-sided: A > B
        p_value = np.mean(bootstrap_diffs <= observed_diff)
    elif alternative == 'less':
        # One-sided: A < B
        p_value = np.mean(bootstrap_diffs >= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Additional statistics
    stats_dict = {
        'mean_a': np.mean(scores_a),
        'mean_b': np.mean(scores_b),
        'std_a': np.std(scores_a, ddof=1),
        'std_b': np.std(scores_b, ddof=1),
        'bootstrap_std': np.std(bootstrap_diffs, ddof=1),
        'bootstrap_ci_95': (
            np.percentile(bootstrap_diffs, 2.5),
            np.percentile(bootstrap_diffs, 97.5)
        )
    }

    return observed_diff, p_value, stats_dict


# =============================================================================
# 3. MCNEMAR'S TEST
# =============================================================================

def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    ground_truth: np.ndarray,
    exact: Optional[bool] = None,
    correction: bool = True
) -> Tuple[float, float, np.ndarray]:
    """
    McNemar's test for comparing two classifiers on the same test set.

    McNemar's test is appropriate when comparing two models that make binary
    predictions (correct/incorrect) on the same set of examples. It's especially
    useful for expensive models where only one test set evaluation is feasible.

    Args:
        predictions_a: Predictions from model A (shape: [n_examples])
        predictions_b: Predictions from model B (shape: [n_examples])
        ground_truth: True labels (shape: [n_examples])
        exact: If None, use exact binomial test when b+c<25, else chi-square
        correction: Apply continuity correction for chi-square (default: True)

    Returns:
        statistic: Chi-square statistic or exact binomial statistic
        p_value: Two-sided p-value
        contingency_table: 2×2 contingency table

    Notes:
        - Null hypothesis: Both models have the same error rate
        - Recommended by Dietterich (1998) for single test set comparisons
        - Use exact test when b+c < 25 (small sample size)

    Example:
        >>> # Suppose we have 100 test examples
        >>> preds_a = np.array([...])  # Model A predictions
        >>> preds_b = np.array([...])  # Model B predictions
        >>> labels = np.array([...])   # Ground truth
        >>> stat, p_val, table = mcnemar_test(preds_a, preds_b, labels)
        >>> print(f"McNemar statistic: {stat:.3f}, p-value: {p_val:.4f}")
        >>> print(f"Contingency table:\n{table}")
    """
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    ground_truth = np.asarray(ground_truth)

    if not (len(predictions_a) == len(predictions_b) == len(ground_truth)):
        raise ValueError("All arrays must have the same length")

    # Create correctness arrays (1 = correct, 0 = incorrect)
    correct_a = (predictions_a == ground_truth).astype(int)
    correct_b = (predictions_b == ground_truth).astype(int)

    # Build 2×2 contingency table:
    #           Model B Correct | Model B Wrong
    # Model A Correct:    n00   |      n01
    # Model A Wrong:      n10   |      n11
    n00 = np.sum((correct_a == 1) & (correct_b == 1))  # Both correct
    n01 = np.sum((correct_a == 1) & (correct_b == 0))  # A correct, B wrong
    n10 = np.sum((correct_a == 0) & (correct_b == 1))  # A wrong, B correct
    n11 = np.sum((correct_a == 0) & (correct_b == 0))  # Both wrong

    contingency_table = np.array([[n00, n01], [n10, n11]])

    # Decide whether to use exact test
    if exact is None:
        # Use exact test when b + c < 25 (small sample size)
        exact = (n01 + n10) < 25

    # Run McNemar's test
    result = mcnemar(contingency_table, exact=exact, correction=correction)

    return result.statistic, result.pvalue, contingency_table


# =============================================================================
# 4. MULTIPLE COMPARISON CORRECTIONS
# =============================================================================

def multiple_comparison_correction(
    p_values: List[float],
    alpha: float = 0.05,
    method: str = 'bonferroni'
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Correct p-values for multiple comparisons.

    Args:
        p_values: List of raw p-values from multiple tests
        alpha: Family-wise error rate (default: 0.05)
        method: Correction method (see options below)

    Available methods:
        - 'bonferroni': Bonferroni correction (most conservative, controls FWER)
        - 'holm': Holm-Bonferroni (less conservative, controls FWER)
        - 'sidak': Sidak correction (controls FWER)
        - 'fdr_bh': Benjamini-Hochberg (controls FDR, more powerful)
        - 'fdr_by': Benjamini-Yekutieli (controls FDR, more conservative)

    Returns:
        reject: Boolean array indicating which hypotheses to reject
        corrected_p_values: Adjusted p-values
        alphac: Corrected alpha (for Bonferroni)
        alpha_sidak: Corrected alpha (for Sidak)

    Notes:
        - FWER (Family-Wise Error Rate): Probability of making ≥1 false positive
        - FDR (False Discovery Rate): Expected proportion of false positives among rejections
        - Use Bonferroni/Holm when false positives are very costly
        - Use FDR methods (fdr_bh) when you have many tests and want more power

    Example:
        >>> # Compare 5 models to a baseline
        >>> p_vals = [0.001, 0.04, 0.06, 0.15, 0.30]
        >>> reject, p_adj, _, _ = multiple_comparison_correction(p_vals, method='bonferroni')
        >>> for i, (p_raw, p_corr, rej) in enumerate(zip(p_vals, p_adj, reject)):
        ...     print(f"Test {i+1}: p={p_raw:.3f} → {p_corr:.3f} (reject={rej})")
    """
    p_values = np.asarray(p_values)

    if method not in ['bonferroni', 'holm', 'sidak', 'fdr_bh', 'fdr_by', 'holm-sidak', 'simes-hochberg']:
        raise ValueError(f"Unknown method: {method}")

    # Use statsmodels multipletests
    reject, corrected_p_values, alphac, alpha_sidak = multipletests(
        p_values,
        alpha=alpha,
        method=method
    )

    return reject, corrected_p_values, alphac, alpha_sidak


def compare_multiple_methods_to_baseline(
    baseline_scores: np.ndarray,
    method_scores: Dict[str, np.ndarray],
    correction: str = 'bonferroni',
    alpha: float = 0.05,
    n_resamples: int = 10000,
    random_state: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Compare multiple methods to a baseline with proper multiple testing correction.

    Args:
        baseline_scores: Scores for baseline method (shape: [n_examples])
        method_scores: Dictionary mapping method names to score arrays
        correction: Multiple testing correction method
        alpha: Family-wise error rate
        n_resamples: Bootstrap resamples for paired tests
        random_state: Random seed

    Returns:
        Dictionary with results for each method (p-values, corrected p-values, significance)

    Example:
        >>> baseline = np.array([0.70, 0.72, 0.68, 0.75, 0.71] * 5)
        >>> methods = {
        ...     'method1': np.array([0.75, 0.77, 0.73, 0.80, 0.76] * 5),
        ...     'method2': np.array([0.72, 0.74, 0.70, 0.77, 0.73] * 5),
        ...     'method3': np.array([0.68, 0.70, 0.66, 0.73, 0.69] * 5)
        ... }
        >>> results = compare_multiple_methods_to_baseline(baseline, methods, correction='fdr_bh')
        >>> for name, res in results.items():
        ...     print(f"{name}: p={res['p_value']:.4f}, corrected_p={res['corrected_p_value']:.4f}, significant={res['significant']}")
    """
    results = {}
    p_values = []
    method_names = []

    # Run paired bootstrap tests for each method
    for method_name, scores in method_scores.items():
        diff, p_val, stats = paired_bootstrap_test(
            scores,
            baseline_scores,
            n_resamples=n_resamples,
            random_state=random_state,
            alternative='two-sided'
        )

        results[method_name] = {
            'difference': diff,
            'p_value': p_val,
            'stats': stats
        }

        p_values.append(p_val)
        method_names.append(method_name)

    # Apply multiple testing correction
    reject, corrected_p_values, _, _ = multiple_comparison_correction(
        p_values,
        alpha=alpha,
        method=correction
    )

    # Add corrected results
    for i, method_name in enumerate(method_names):
        results[method_name]['corrected_p_value'] = corrected_p_values[i]
        results[method_name]['significant'] = reject[i]
        results[method_name]['alpha'] = alpha

    return results


# =============================================================================
# 5. POWER ANALYSIS & SAMPLE SIZE CALCULATION
# =============================================================================

def estimate_required_samples(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    std_dev: float = 0.1
) -> int:
    """
    Estimate required number of samples for detecting an effect size.

    Uses power analysis to determine how many test examples (or random seeds)
    are needed to reliably detect a performance difference between methods.

    Args:
        effect_size: Expected difference in means (e.g., 0.05 for 5% improvement)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate), typically 0.80
        std_dev: Expected standard deviation of scores

    Returns:
        Estimated number of samples needed per group

    Notes:
        - Power = 0.80 means 80% chance of detecting a true effect
        - For comparing methods with bootstrap, this is the test set size
        - For comparing across random seeds, this is number of seeds needed
        - Based on two-sample t-test approximation

    Reference:
        Colas et al. (2018): "How Many Random Seeds? Statistical Power Analysis"

    Example:
        >>> # How many test examples to detect 5% F1 improvement?
        >>> n = estimate_required_samples(effect_size=0.05, std_dev=0.15)
        >>> print(f"Need {n} test examples to detect 5% improvement with 80% power")
    """
    from scipy.stats import norm

    # Z-scores for alpha and power
    z_alpha = norm.ppf(1 - alpha/2)  # Two-sided test
    z_beta = norm.ppf(power)

    # Cohen's d (standardized effect size)
    d = effect_size / std_dev

    # Sample size formula for two independent samples
    # For paired comparisons, we can use the same formula with paired std dev
    n = 2 * ((z_alpha + z_beta) / d) ** 2

    return int(np.ceil(n))


def estimate_required_seeds_from_data(
    scores_pilot: np.ndarray,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Estimate required number of random seeds based on pilot data.

    Args:
        scores_pilot: Pilot scores from a small number of seeds
        effect_size: Minimum effect size you want to detect
        alpha: Significance level
        power: Desired statistical power

    Returns:
        Estimated number of random seeds needed

    Example:
        >>> # Ran 5 pilot experiments with different seeds
        >>> pilot_scores = np.array([0.70, 0.72, 0.68, 0.75, 0.71])
        >>> n_seeds = estimate_required_seeds_from_data(pilot_scores, effect_size=0.05)
        >>> print(f"Recommend running {n_seeds} random seeds for robust comparison")
    """
    std_dev = np.std(scores_pilot, ddof=1)
    return estimate_required_samples(effect_size, alpha, power, std_dev)


# =============================================================================
# 6. COMPREHENSIVE COMPARISON REPORT
# =============================================================================

def comprehensive_comparison_report(
    baseline_name: str,
    baseline_scores: np.ndarray,
    method_scores: Dict[str, np.ndarray],
    correction: str = 'fdr_bh',
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None
) -> str:
    """
    Generate a comprehensive statistical comparison report.

    Args:
        baseline_name: Name of baseline method
        baseline_scores: Scores for baseline
        method_scores: Dictionary of method names to scores
        correction: Multiple testing correction method
        alpha: Significance level
        n_bootstrap: Bootstrap resamples
        random_state: Random seed

    Returns:
        Formatted report string

    Example:
        >>> baseline = np.random.normal(0.70, 0.05, 50)
        >>> methods = {
        ...     'Method A': np.random.normal(0.75, 0.05, 50),
        ...     'Method B': np.random.normal(0.72, 0.05, 50)
        ... }
        >>> report = comprehensive_comparison_report('Baseline', baseline, methods)
        >>> print(report)
    """
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")

    # Sample size check
    n_samples = len(baseline_scores)
    report.append(f"Sample size: {n_samples}")
    if n_samples < 20:
        report.append("⚠️  WARNING: Sample size < 20. Results may not be reliable.")
    report.append(f"Significance level (α): {alpha}")
    report.append(f"Multiple testing correction: {correction}")
    report.append("")

    # Baseline statistics with CI
    baseline_mean, baseline_ci = bootstrap_ci(
        baseline_scores,
        confidence_level=1-alpha,
        n_resamples=n_bootstrap,
        method='BCa',
        random_state=random_state
    )

    report.append("-" * 80)
    report.append(f"BASELINE: {baseline_name}")
    report.append("-" * 80)
    report.append(f"Mean: {baseline_mean:.4f}")
    report.append(f"95% CI: [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]")
    report.append(f"Std Dev: {np.std(baseline_scores, ddof=1):.4f}")
    report.append("")

    # Compare each method to baseline
    comparison_results = compare_multiple_methods_to_baseline(
        baseline_scores,
        method_scores,
        correction=correction,
        alpha=alpha,
        n_resamples=n_bootstrap,
        random_state=random_state
    )

    report.append("-" * 80)
    report.append("METHOD COMPARISONS (vs Baseline)")
    report.append("-" * 80)
    report.append("")

    for method_name in sorted(method_scores.keys()):
        scores = method_scores[method_name]
        res = comparison_results[method_name]

        # Method statistics with CI
        method_mean, method_ci = bootstrap_ci(
            scores,
            confidence_level=1-alpha,
            n_resamples=n_bootstrap,
            method='BCa',
            random_state=random_state
        )

        report.append(f"Method: {method_name}")
        report.append(f"  Mean: {method_mean:.4f}")
        report.append(f"  95% CI: [{method_ci[0]:.4f}, {method_ci[1]:.4f}]")
        report.append(f"  Difference from baseline: {res['difference']:+.4f}")
        report.append(f"  Improvement: {(res['difference']/baseline_mean)*100:+.2f}%")
        report.append(f"  Raw p-value: {res['p_value']:.4f}")
        report.append(f"  Corrected p-value: {res['corrected_p_value']:.4f}")

        if res['significant']:
            report.append(f"  ✓ STATISTICALLY SIGNIFICANT at α={alpha}")
        else:
            report.append(f"  ✗ Not statistically significant")
        report.append("")

    # Power analysis
    report.append("-" * 80)
    report.append("POWER ANALYSIS")
    report.append("-" * 80)

    # Estimate required samples to detect various effect sizes
    std_pooled = np.sqrt(np.mean([np.var(s, ddof=1) for s in [baseline_scores] + list(method_scores.values())]))

    report.append(f"Pooled standard deviation: {std_pooled:.4f}")
    report.append("")
    report.append("Required sample sizes to detect effects (80% power, α=0.05):")

    for effect_pct in [1, 2, 5, 10]:
        effect_abs = baseline_mean * (effect_pct / 100)
        n_required = estimate_required_samples(effect_abs, alpha=0.05, power=0.80, std_dev=std_pooled)
        report.append(f"  {effect_pct}% improvement ({effect_abs:.4f}): {n_required} samples")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    print("Statistical Testing Utilities for ML Experiments")
    print("=" * 80)
    print()

    # Simulate experimental data (50 test examples)
    n_samples = 50
    baseline_scores = np.random.beta(7, 3, n_samples)  # Mean ~0.70
    method_a_scores = np.random.beta(7.5, 2.5, n_samples)  # Mean ~0.75 (better)
    method_b_scores = np.random.beta(7.2, 2.8, n_samples)  # Mean ~0.72 (slightly better)
    method_c_scores = np.random.beta(6.8, 3.2, n_samples)  # Mean ~0.68 (slightly worse)

    # Example 1: Bootstrap CI for a single method
    print("1. BOOTSTRAP CONFIDENCE INTERVAL")
    print("-" * 80)
    mean_val, (lower, upper) = bootstrap_ci(baseline_scores, n_resamples=10000, random_state=42)
    print(f"Baseline mean: {mean_val:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
    print()

    # Example 2: Paired bootstrap test
    print("2. PAIRED BOOTSTRAP TEST")
    print("-" * 80)
    diff, p_val, stats = paired_bootstrap_test(
        method_a_scores,
        baseline_scores,
        n_resamples=10000,
        random_state=42
    )
    print(f"Method A vs Baseline:")
    print(f"  Difference: {diff:+.4f} ({(diff/stats['mean_b'])*100:+.2f}%)")
    print(f"  p-value: {p_val:.4f}")
    print(f"  Method A mean: {stats['mean_a']:.4f} ± {stats['std_a']:.4f}")
    print(f"  Baseline mean: {stats['mean_b']:.4f} ± {stats['std_b']:.4f}")
    if p_val < 0.05:
        print("  ✓ Statistically significant at α=0.05")
    print()

    # Example 3: Multiple comparisons with correction
    print("3. MULTIPLE COMPARISONS WITH FDR CORRECTION")
    print("-" * 80)
    methods = {
        'Method A': method_a_scores,
        'Method B': method_b_scores,
        'Method C': method_c_scores
    }

    results = compare_multiple_methods_to_baseline(
        baseline_scores,
        methods,
        correction='fdr_bh',
        alpha=0.05,
        n_resamples=10000,
        random_state=42
    )

    for name, res in results.items():
        print(f"{name}:")
        print(f"  Raw p-value: {res['p_value']:.4f}")
        print(f"  Corrected p-value: {res['corrected_p_value']:.4f}")
        print(f"  Significant: {res['significant']}")
        print()

    # Example 4: Comprehensive report
    print("4. COMPREHENSIVE COMPARISON REPORT")
    print("-" * 80)
    report = comprehensive_comparison_report(
        'Baseline',
        baseline_scores,
        methods,
        correction='fdr_bh',
        alpha=0.05,
        n_bootstrap=10000,
        random_state=42
    )
    print(report)

    # Example 5: Power analysis
    print("\n5. POWER ANALYSIS")
    print("-" * 80)
    effect_size = 0.05  # Want to detect 5% improvement
    std_dev = np.std(baseline_scores, ddof=1)
    n_required = estimate_required_samples(effect_size, alpha=0.05, power=0.80, std_dev=std_dev)
    print(f"To detect {effect_size:.3f} effect size with 80% power:")
    print(f"  Required samples: {n_required}")
    print(f"  Current samples: {n_samples}")
    if n_samples >= n_required:
        print("  ✓ Adequate sample size")
    else:
        print(f"  ✗ Need {n_required - n_samples} more samples")
