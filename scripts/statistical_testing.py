"""
Statistical Testing Utilities for Machine Learning Experiments

Implements rigorous statistical methods for comparing ML models:
- Bootstrap confidence intervals (95% CI with BCa method)
- Paired and independent t-tests with proper ddof=1 for sample std
- Paired bootstrap test for comparing methods
- McNemar's test for classification comparison
- Multiple comparison corrections (Bonferroni, Holm, FDR)
- Cohen's d effect size (pooled and paired variants)
- Power analysis for determining required sample sizes
- Multi-seed aggregation with low-sample warnings
- Summary table generation with significance stars

References:
- Dietterich (1998): "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
- Colas et al. (2018): "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments"
- Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
- Hedges & Olkin (1985): "Statistical Methods for Meta-Analysis"
"""

import numpy as np
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Dict, Optional, Callable, Union
import warnings
import pandas as pd


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
# 2. T-TESTS (PAIRED AND INDEPENDENT)
# =============================================================================

def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float, Dict[str, float]]:
    """
    Paired t-test for within-subject comparisons (same test set, different methods).

    Use when comparing two methods evaluated on the same test examples or same random seeds.
    Accounts for correlation between measurements.

    Args:
        scores_a: Scores for method A (shape: [n_examples])
        scores_b: Scores for method B (shape: [n_examples])
        alternative: 'two-sided', 'greater' (A > B), or 'less' (A < B)

    Returns:
        observed_diff: Mean difference (mean_a - mean_b)
        p_value: p-value from paired t-test
        stats: Dictionary with additional statistics

    Notes:
        - Uses ddof=1 (Bessel's correction) for unbiased sample std
        - Paired t-test is more powerful than independent when measurements are correlated
        - For n=3 seeds: t-distribution has 2 degrees of freedom → wide intervals, low power

    Example:
        >>> # Same 3 random seeds evaluated with two methods
        >>> method_a = np.array([0.75, 0.77, 0.73])  # 3 seeds
        >>> method_b = np.array([0.70, 0.72, 0.68])
        >>> diff, p_val, stats = paired_ttest(method_a, method_b)
        >>> print(f"Difference: {diff:.4f}, p={p_val:.4f}")
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length for paired t-test")

    n = len(scores_a)

    # Warning for low sample sizes
    if n < 5:
        warnings.warn(
            f"Only {n} paired samples. Paired t-test has very low power with n<5. "
            "Results should be interpreted with extreme caution."
        )

    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Paired t-test using scipy (uses ddof=1 internally)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b, alternative=alternative)

    # Additional statistics
    diffs = scores_a - scores_b
    stats_dict = {
        'mean_a': np.mean(scores_a),
        'mean_b': np.mean(scores_b),
        'std_a': np.std(scores_a, ddof=1),
        'std_b': np.std(scores_b, ddof=1),
        'std_diff': np.std(diffs, ddof=1),
        't_statistic': t_stat,
        'degrees_of_freedom': n - 1,
        'n': n
    }

    return observed_diff, p_value, stats_dict


def independent_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    equal_var: bool = True,
    alternative: str = 'two-sided'
) -> Tuple[float, float, Dict[str, float]]:
    """
    Independent t-test for between-condition comparisons.

    Use when comparing two methods on different test sets or different samples
    (no pairing between measurements).

    Args:
        scores_a: Scores for method A (shape: [n_a])
        scores_b: Scores for method B (shape: [n_b])
        equal_var: Assume equal variances? If False, uses Welch's t-test
        alternative: 'two-sided', 'greater' (A > B), or 'less' (A < B)

    Returns:
        observed_diff: Mean difference (mean_a - mean_b)
        p_value: p-value from independent t-test
        stats: Dictionary with additional statistics

    Notes:
        - Uses ddof=1 for unbiased sample std
        - Welch's t-test (equal_var=False) is more robust when variances differ
        - For small samples (n<10 per group), consider bootstrap test instead

    Example:
        >>> # Different test sets for two methods
        >>> method_a = np.array([0.75, 0.77, 0.73, 0.76, 0.74])
        >>> method_b = np.array([0.70, 0.72, 0.68, 0.71])
        >>> diff, p_val, stats = independent_ttest(method_a, method_b, equal_var=False)
        >>> print(f"Difference: {diff:.4f}, p={p_val:.4f}")
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    n_a = len(scores_a)
    n_b = len(scores_b)

    # Warning for low sample sizes
    if n_a < 5 or n_b < 5:
        warnings.warn(
            f"Small sample size (n_a={n_a}, n_b={n_b}). "
            "Independent t-test may have low power. Consider bootstrap test."
        )

    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Independent t-test using scipy (uses ddof=1 internally)
    t_stat, p_value = stats.ttest_ind(
        scores_a, scores_b,
        equal_var=equal_var,
        alternative=alternative
    )

    # Additional statistics
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)

    # Degrees of freedom calculation
    if equal_var:
        # Pooled variance t-test
        df = n_a + n_b - 2
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / df)
    else:
        # Welch's t-test (unequal variances)
        df = (std_a**2 / n_a + std_b**2 / n_b)**2 / (
            (std_a**2 / n_a)**2 / (n_a - 1) + (std_b**2 / n_b)**2 / (n_b - 1)
        )
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

    stats_dict = {
        'mean_a': np.mean(scores_a),
        'mean_b': np.mean(scores_b),
        'std_a': std_a,
        'std_b': std_b,
        'pooled_std': pooled_std,
        't_statistic': t_stat,
        'degrees_of_freedom': df,
        'n_a': n_a,
        'n_b': n_b,
        'equal_var': equal_var
    }

    return observed_diff, p_value, stats_dict


# =============================================================================
# 3. EFFECT SIZES (COHEN'S D)
# =============================================================================

def cohens_d_pooled(
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> float:
    """
    Cohen's d effect size for independent samples (pooled standard deviation).

    Measures standardized difference between two independent groups.
    Interpretation (Cohen, 1988):
        |d| < 0.2: negligible
        |d| < 0.5: small
        |d| < 0.8: medium
        |d| ≥ 0.8: large

    Args:
        scores_a: Scores for group A
        scores_b: Scores for group B

    Returns:
        Cohen's d using pooled standard deviation

    Example:
        >>> method_a = np.array([0.75, 0.77, 0.73, 0.76, 0.74])
        >>> method_b = np.array([0.70, 0.72, 0.68, 0.71, 0.69])
        >>> d = cohens_d_pooled(method_a, method_b)
        >>> print(f"Cohen's d = {d:.3f}")
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    n_a = len(scores_a)
    n_b = len(scores_b)

    mean_diff = np.mean(scores_a) - np.mean(scores_b)

    # Pooled standard deviation (uses ddof=1)
    var_a = np.var(scores_a, ddof=1)
    var_b = np.var(scores_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return float('inf') if mean_diff != 0 else 0.0

    return mean_diff / pooled_std


def cohens_d_paired(
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> float:
    """
    Cohen's d effect size for paired samples.

    Measures standardized difference for within-subject comparisons.
    Uses the standard deviation of the differences (more appropriate for paired data).

    Args:
        scores_a: Scores for condition A
        scores_b: Scores for condition B

    Returns:
        Cohen's d using standard deviation of differences

    Example:
        >>> method_a = np.array([0.75, 0.77, 0.73])  # 3 seeds
        >>> method_b = np.array([0.70, 0.72, 0.68])
        >>> d = cohens_d_paired(method_a, method_b)
        >>> print(f"Cohen's d (paired) = {d:.3f}")
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Arrays must have same length for paired Cohen's d")

    diffs = scores_a - scores_b
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    if std_diff == 0:
        return float('inf') if mean_diff != 0 else 0.0

    return mean_diff / std_diff


# =============================================================================
# 4. SIGNIFICANCE STARS AND FORMATTING
# =============================================================================

def p_value_to_stars(p_value: float) -> str:
    """
    Convert p-value to significance stars.

    Convention:
        p < 0.001: ***
        p < 0.01:  **
        p < 0.05:  *
        p ≥ 0.05:  (empty string)

    Args:
        p_value: p-value from statistical test

    Returns:
        String with stars ('***', '**', '*', or '')

    Example:
        >>> print(f"p = {0.0001:.4f} {p_value_to_stars(0.0001)}")
        p = 0.0001 ***
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def format_mean_ci(
    data: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    method: str = 'BCa',
    random_state: Optional[int] = None,
    decimals: int = 3
) -> str:
    """
    Format mean with confidence interval as a string.

    Args:
        data: Array of scores
        confidence_level: Confidence level (default: 0.95)
        n_resamples: Bootstrap resamples
        method: Bootstrap method
        random_state: Random seed
        decimals: Number of decimal places

    Returns:
        Formatted string: "mean [lower, upper]"

    Example:
        >>> scores = np.array([0.75, 0.77, 0.73])
        >>> print(format_mean_ci(scores))
        0.750 [0.730, 0.770]
    """
    mean_val, (lower, upper) = bootstrap_ci(
        data,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method=method,
        random_state=random_state
    )

    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean_val)} [{fmt.format(lower)}, {fmt.format(upper)}]"


# =============================================================================
# 5. MULTI-SEED AGGREGATION
# =============================================================================

def aggregate_multiseed_results(
    seed_scores: Dict[int, float],
    metric_name: str = "metric"
) -> Dict[str, Union[float, int]]:
    """
    Aggregate results across multiple random seeds with proper statistics.

    Handles the common case of n=3 seeds with appropriate warnings about
    statistical power and wide confidence intervals.

    Args:
        seed_scores: Dictionary mapping seed ID to score
        metric_name: Name of metric for warning messages

    Returns:
        Dictionary with mean, std (ddof=1), CI, and sample size

    Notes:
        - Uses ddof=1 (Bessel's correction) for unbiased sample std
        - For n=3: very wide CIs, interpret with caution
        - For n<3: Cannot compute meaningful statistics

    Example:
        >>> seeds = {42: 0.75, 123: 0.77, 456: 0.73}
        >>> stats = aggregate_multiseed_results(seeds, metric_name="F1")
        >>> print(f"F1: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n']})")
    """
    scores = np.array(list(seed_scores.values()))
    n = len(scores)

    if n < 2:
        warnings.warn(
            f"Only {n} seed(s) for {metric_name}. Cannot compute meaningful statistics. "
            "Need at least 2 seeds for standard deviation."
        )
        return {
            'mean': float(np.mean(scores)),
            'std': float('nan'),
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
            'n': n,
            'seeds': list(seed_scores.keys())
        }

    if n == 3:
        warnings.warn(
            f"Only {n} seeds for {metric_name}. "
            "With n=3, confidence intervals are very wide (t-dist has 2 df). "
            "Results have low statistical power. Consider using 5-10 seeds for robust comparisons."
        )

    mean_val = np.mean(scores)
    std_val = np.std(scores, ddof=1)  # Unbiased sample std

    # Bootstrap CI
    if n >= 3:
        _, (ci_lower, ci_upper) = bootstrap_ci(
            scores,
            confidence_level=0.95,
            n_resamples=10000,
            method='percentile',  # Use percentile for small n (BCa can fail)
            random_state=42
        )
    else:
        ci_lower = ci_upper = float('nan')

    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n': n,
        'seeds': list(seed_scores.keys())
    }


# =============================================================================
# 6. SUMMARY TABLE GENERATION
# =============================================================================

def generate_comparison_table(
    results: Dict[str, Dict[str, np.ndarray]],
    baseline_name: str,
    metric_names: List[str],
    correction: str = 'bonferroni',
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None,
    output_format: str = 'markdown'
) -> Union[pd.DataFrame, str]:
    """
    Generate a formatted comparison table with statistical tests.

    Compares multiple methods across multiple metrics with proper significance testing
    and multiple comparison correction.

    Args:
        results: Nested dict {method_name: {metric_name: scores_array}}
        baseline_name: Name of baseline method (must be in results)
        metric_names: List of metrics to include in table
        correction: Multiple comparison correction method
        alpha: Significance level
        n_bootstrap: Bootstrap resamples
        random_state: Random seed
        output_format: 'markdown', 'latex', or 'dataframe'

    Returns:
        Formatted table as string (markdown/latex) or pandas DataFrame

    Example:
        >>> results = {
        ...     'Baseline': {'F1': np.array([0.70, 0.72, 0.68]), 'EM': np.array([0.50, 0.52, 0.48])},
        ...     'Method A': {'F1': np.array([0.75, 0.77, 0.73]), 'EM': np.array([0.55, 0.57, 0.53])},
        ...     'Method B': {'F1': np.array([0.72, 0.74, 0.70]), 'EM': np.array([0.52, 0.54, 0.50])}
        ... }
        >>> table = generate_comparison_table(results, 'Baseline', ['F1', 'EM'])
        >>> print(table)
    """
    if baseline_name not in results:
        raise ValueError(f"Baseline '{baseline_name}' not found in results")

    # Build rows for table
    rows = []

    for method_name in sorted(results.keys()):
        row = {'Method': method_name}

        for metric_name in metric_names:
            if metric_name not in results[method_name]:
                row[metric_name] = 'N/A'
                continue

            scores = results[method_name][metric_name]
            mean_val, (ci_lower, ci_upper) = bootstrap_ci(
                scores,
                confidence_level=1-alpha,
                n_resamples=n_bootstrap,
                method='BCa',
                random_state=random_state
            )

            # Format: mean ± std
            std_val = np.std(scores, ddof=1)
            cell = f"{mean_val:.3f} ± {std_val:.3f}"

            # Add significance test if not baseline
            if method_name != baseline_name:
                baseline_scores = results[baseline_name][metric_name]

                # Paired t-test
                _, p_val, _ = paired_ttest(scores, baseline_scores)

                # Apply Bonferroni correction manually (per metric)
                n_comparisons = len([m for m in results.keys() if m != baseline_name])
                if correction == 'bonferroni':
                    p_val_corrected = min(p_val * n_comparisons, 1.0)
                else:
                    p_val_corrected = p_val

                stars = p_value_to_stars(p_val_corrected)
                if stars:
                    cell += f" {stars}"

            row[metric_name] = cell

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    if output_format == 'dataframe':
        return df
    elif output_format == 'latex':
        return df.to_latex(index=False, escape=False)
    else:  # markdown
        return df.to_markdown(index=False)


def generate_detailed_comparison_table(
    baseline_scores: np.ndarray,
    method_scores: Dict[str, np.ndarray],
    baseline_name: str = 'Baseline',
    correction: str = 'bonferroni',
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None,
    output_format: str = 'markdown'
) -> Union[pd.DataFrame, str]:
    """
    Generate a detailed comparison table for a single metric.

    Includes mean, CI, difference from baseline, p-value, and significance stars.

    Args:
        baseline_scores: Scores for baseline method
        method_scores: Dictionary mapping method names to score arrays
        baseline_name: Name of baseline
        correction: Multiple comparison correction
        alpha: Significance level
        n_bootstrap: Bootstrap resamples
        random_state: Random seed
        output_format: 'markdown', 'latex', or 'dataframe'

    Returns:
        Formatted table as string or DataFrame

    Example:
        >>> baseline = np.array([0.70, 0.72, 0.68, 0.75, 0.71])
        >>> methods = {
        ...     'Method A': np.array([0.75, 0.77, 0.73, 0.80, 0.76]),
        ...     'Method B': np.array([0.72, 0.74, 0.70, 0.77, 0.73])
        ... }
        >>> table = generate_detailed_comparison_table(baseline, methods)
        >>> print(table)
    """
    rows = []

    # Baseline row
    baseline_mean, baseline_ci = bootstrap_ci(
        baseline_scores,
        confidence_level=1-alpha,
        n_resamples=n_bootstrap,
        method='BCa',
        random_state=random_state
    )

    rows.append({
        'Method': baseline_name,
        'Mean': f"{baseline_mean:.4f}",
        '95% CI': f"[{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]",
        'Δ': '—',
        'p-value': '—',
        'Sig': ''
    })

    # Compare each method to baseline
    comparison_results = compare_multiple_methods_to_baseline(
        baseline_scores,
        method_scores,
        correction=correction,
        alpha=alpha,
        n_resamples=n_bootstrap,
        random_state=random_state
    )

    for method_name in sorted(method_scores.keys()):
        scores = method_scores[method_name]
        res = comparison_results[method_name]

        method_mean, method_ci = bootstrap_ci(
            scores,
            confidence_level=1-alpha,
            n_resamples=n_bootstrap,
            method='BCa',
            random_state=random_state
        )

        rows.append({
            'Method': method_name,
            'Mean': f"{method_mean:.4f}",
            '95% CI': f"[{method_ci[0]:.4f}, {method_ci[1]:.4f}]",
            'Δ': f"{res['difference']:+.4f}",
            'p-value': f"{res['corrected_p_value']:.4f}",
            'Sig': p_value_to_stars(res['corrected_p_value'])
        })

    df = pd.DataFrame(rows)

    if output_format == 'dataframe':
        return df
    elif output_format == 'latex':
        return df.to_latex(index=False, escape=False)
    else:  # markdown
        return df.to_markdown(index=False)


# =============================================================================
# 7. PAIRED BOOTSTRAP TEST
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


# =============================================================================
# 8. MULTI-SEED EXPERIMENT FRAMEWORK
# =============================================================================

# Standard seeds for reproducibility across all experiments
STANDARD_SEEDS = [42, 123, 456, 789, 2024]


class MultiSeedExperiment:
    """
    Framework for running experiments across multiple random seeds with
    comprehensive statistical validation.

    This class manages:
    - Running experiments across 5 standard seeds
    - Collecting per-seed and per-example results
    - Computing aggregate statistics with proper CIs
    - Performing significance tests between methods
    - Generating paper-ready tables and reports

    Example:
        >>> experiment = MultiSeedExperiment(
        ...     name="bridge_vs_baselines",
        ...     datasets=["sst2", "agnews", "trec"],
        ...     methods=["bridge", "prompt_tuning", "zeroshot"],
        ...     output_dir="runs/statistical_validation"
        ... )
        >>> # Add results for each seed
        >>> experiment.add_result("sst2", "bridge", seed=42, accuracy=96.5, predictions=[...], labels=[...])
        >>> # Generate analysis
        >>> experiment.compute_all_statistics()
        >>> experiment.save_results()
    """

    def __init__(
        self,
        name: str,
        datasets: List[str],
        methods: List[str],
        seeds: List[int] = None,
        output_dir: str = "runs/statistical_validation",
        baseline_method: str = None
    ):
        """
        Initialize multi-seed experiment.

        Args:
            name: Experiment name for file naming
            datasets: List of dataset names (e.g., ["sst2", "agnews", "trec"])
            methods: List of method names (e.g., ["bridge", "prompt_tuning"])
            seeds: List of random seeds (default: [42, 123, 456, 789, 2024])
            output_dir: Directory to save results
            baseline_method: Method to use as baseline for comparisons (default: first method)
        """
        self.name = name
        self.datasets = datasets
        self.methods = methods
        self.seeds = seeds or STANDARD_SEEDS
        self.output_dir = output_dir
        self.baseline_method = baseline_method or methods[0]

        # Results storage
        # Structure: {dataset: {method: {seed: {metrics}}}}
        self.per_seed_results: Dict[str, Dict[str, Dict[int, Dict]]] = {
            ds: {m: {} for m in methods} for ds in datasets
        }

        # Per-example predictions for McNemar's test
        # Structure: {dataset: {method: {seed: {"predictions": [], "labels": []}}}}
        self.per_example_results: Dict[str, Dict[str, Dict[int, Dict]]] = {
            ds: {m: {} for m in methods} for ds in datasets
        }

        # Aggregated statistics (computed after all seeds complete)
        self.aggregated_results: Dict[str, Dict[str, Dict]] = {}

        # Statistical test results
        self.significance_tests: Dict[str, Dict] = {}

        # Paper tables (generated on demand)
        self.paper_tables: Dict[str, str] = {}

    def add_result(
        self,
        dataset: str,
        method: str,
        seed: int,
        accuracy: float,
        predictions: Optional[List] = None,
        labels: Optional[List] = None,
        f1: Optional[float] = None,
        latency_ms: Optional[float] = None,
        memory_mb: Optional[float] = None,
        extra_metrics: Optional[Dict] = None
    ) -> None:
        """
        Add result for a single seed run.

        Args:
            dataset: Dataset name
            method: Method name
            seed: Random seed used
            accuracy: Accuracy score (0-100)
            predictions: Per-example predictions (for McNemar's test)
            labels: Ground truth labels (for McNemar's test)
            f1: Optional F1 score
            latency_ms: Optional latency in milliseconds
            memory_mb: Optional memory usage in MB
            extra_metrics: Any additional metrics to store
        """
        if dataset not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset}. Valid: {self.datasets}")
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Valid: {self.methods}")
        if seed not in self.seeds:
            warnings.warn(f"Seed {seed} not in standard seeds {self.seeds}")

        # Store metrics
        metrics = {
            "accuracy": accuracy,
            "seed": seed
        }
        if f1 is not None:
            metrics["f1"] = f1
        if latency_ms is not None:
            metrics["latency_ms"] = latency_ms
        if memory_mb is not None:
            metrics["memory_mb"] = memory_mb
        if extra_metrics:
            metrics.update(extra_metrics)

        self.per_seed_results[dataset][method][seed] = metrics

        # Store per-example results for McNemar's test
        if predictions is not None and labels is not None:
            self.per_example_results[dataset][method][seed] = {
                "predictions": list(predictions),
                "labels": list(labels)
            }

    def get_completion_status(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Get which seeds have been completed for each dataset/method.

        Returns:
            Dictionary showing completed seeds for each condition
        """
        status = {}
        for dataset in self.datasets:
            status[dataset] = {}
            for method in self.methods:
                completed = list(self.per_seed_results[dataset][method].keys())
                status[dataset][method] = sorted(completed)
        return status

    def is_complete(self) -> bool:
        """Check if all seeds have been run for all conditions."""
        for dataset in self.datasets:
            for method in self.methods:
                if len(self.per_seed_results[dataset][method]) < len(self.seeds):
                    return False
        return True

    def compute_aggregate_statistics(self, n_bootstrap: int = 10000) -> None:
        """
        Compute mean, std, and 95% CI across seeds for all conditions.

        Args:
            n_bootstrap: Number of bootstrap resamples for CI
        """
        self.aggregated_results = {}

        for dataset in self.datasets:
            self.aggregated_results[dataset] = {}

            for method in self.methods:
                seed_results = self.per_seed_results[dataset][method]

                if not seed_results:
                    continue

                # Collect accuracy values across seeds
                accuracies = [r["accuracy"] for r in seed_results.values()]
                accuracies = np.array(accuracies)

                n = len(accuracies)

                # Compute statistics
                agg = {
                    "n_seeds": n,
                    "seeds": list(seed_results.keys()),
                    "accuracy_mean": float(np.mean(accuracies)),
                    "accuracy_std": float(np.std(accuracies, ddof=1)) if n > 1 else 0.0,
                    "accuracy_min": float(np.min(accuracies)),
                    "accuracy_max": float(np.max(accuracies)),
                    "accuracy_values": accuracies.tolist()
                }

                # Bootstrap CI (need at least 3 samples)
                if n >= 3:
                    _, (ci_lower, ci_upper) = bootstrap_ci(
                        accuracies,
                        confidence_level=0.95,
                        n_resamples=n_bootstrap,
                        method='percentile',  # Use percentile for small n
                        random_state=42
                    )
                    agg["accuracy_ci_lower"] = float(ci_lower)
                    agg["accuracy_ci_upper"] = float(ci_upper)
                    agg["ci_method"] = "bootstrap_percentile"
                elif n == 2:
                    # Use t-distribution CI for n=2
                    se = agg["accuracy_std"] / np.sqrt(n)
                    t_val = stats.t.ppf(0.975, df=n-1)
                    agg["accuracy_ci_lower"] = float(agg["accuracy_mean"] - t_val * se)
                    agg["accuracy_ci_upper"] = float(agg["accuracy_mean"] + t_val * se)
                    agg["ci_method"] = "t_distribution"
                else:
                    agg["accuracy_ci_lower"] = agg["accuracy_mean"]
                    agg["accuracy_ci_upper"] = agg["accuracy_mean"]
                    agg["ci_method"] = "none"

                # Also aggregate other metrics if present
                for metric in ["f1", "latency_ms", "memory_mb"]:
                    values = [r.get(metric) for r in seed_results.values() if r.get(metric) is not None]
                    if values:
                        agg[f"{metric}_mean"] = float(np.mean(values))
                        agg[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

                self.aggregated_results[dataset][method] = agg

                # Warn about low seed count
                if n < 5:
                    warnings.warn(
                        f"{dataset}/{method}: Only {n} seeds. "
                        f"Recommend 5 seeds for robust statistics."
                    )

    def compute_significance_tests(self, correction: str = "bonferroni", alpha: float = 0.05) -> None:
        """
        Compute statistical significance between baseline and other methods.

        Performs:
        1. Paired t-tests across seeds (method vs baseline)
        2. McNemar's test on per-example predictions (when available)
        3. Multiple comparison correction
        4. Cohen's d effect sizes

        Args:
            correction: Multiple testing correction ("bonferroni", "holm", "fdr_bh")
            alpha: Significance level
        """
        self.significance_tests = {}

        for dataset in self.datasets:
            self.significance_tests[dataset] = {}

            # Get baseline results
            baseline_seeds = self.per_seed_results[dataset].get(self.baseline_method, {})
            if not baseline_seeds:
                continue

            baseline_accuracies = np.array([r["accuracy"] for r in baseline_seeds.values()])

            # Compare each method to baseline
            p_values = []
            method_names = []

            for method in self.methods:
                if method == self.baseline_method:
                    continue

                method_seeds = self.per_seed_results[dataset][method]
                if not method_seeds:
                    continue

                method_accuracies = np.array([r["accuracy"] for r in method_seeds.values()])

                test_result = {
                    "method": method,
                    "baseline": self.baseline_method,
                    "n_seeds_method": len(method_accuracies),
                    "n_seeds_baseline": len(baseline_accuracies),
                }

                # Paired t-test (if same seeds available)
                common_seeds = set(baseline_seeds.keys()) & set(method_seeds.keys())
                if len(common_seeds) >= 2:
                    # Get paired values
                    paired_baseline = [baseline_seeds[s]["accuracy"] for s in sorted(common_seeds)]
                    paired_method = [method_seeds[s]["accuracy"] for s in sorted(common_seeds)]

                    _, p_val, t_stats = paired_ttest(
                        np.array(paired_method),
                        np.array(paired_baseline)
                    )

                    test_result["paired_t_test"] = {
                        "t_statistic": t_stats["t_statistic"],
                        "p_value": p_val,
                        "degrees_of_freedom": t_stats["degrees_of_freedom"],
                        "mean_difference": t_stats["mean_a"] - t_stats["mean_b"],
                        "n_pairs": len(common_seeds)
                    }

                    # Cohen's d (paired)
                    test_result["cohens_d"] = cohens_d_paired(
                        np.array(paired_method),
                        np.array(paired_baseline)
                    )

                    p_values.append(p_val)
                    method_names.append(method)
                else:
                    # Independent t-test fallback
                    _, p_val, t_stats = independent_ttest(
                        method_accuracies,
                        baseline_accuracies,
                        equal_var=False  # Welch's t-test
                    )

                    test_result["independent_t_test"] = {
                        "t_statistic": t_stats["t_statistic"],
                        "p_value": p_val,
                        "degrees_of_freedom": t_stats["degrees_of_freedom"],
                        "mean_difference": t_stats["mean_a"] - t_stats["mean_b"]
                    }

                    # Cohen's d (pooled)
                    test_result["cohens_d"] = cohens_d_pooled(
                        method_accuracies,
                        baseline_accuracies
                    )

                    p_values.append(p_val)
                    method_names.append(method)

                # McNemar's test (if per-example predictions available)
                baseline_examples = self.per_example_results[dataset].get(self.baseline_method, {})
                method_examples = self.per_example_results[dataset].get(method, {})

                # Use first seed with both predictions available
                for seed in common_seeds if len(common_seeds) > 0 else []:
                    if seed in baseline_examples and seed in method_examples:
                        b_preds = np.array(baseline_examples[seed]["predictions"])
                        m_preds = np.array(method_examples[seed]["predictions"])
                        labels = np.array(baseline_examples[seed]["labels"])

                        if len(b_preds) == len(m_preds) == len(labels):
                            stat, mcn_p_val, table = mcnemar_test(
                                m_preds, b_preds, labels
                            )
                            test_result["mcnemar_test"] = {
                                "statistic": float(stat),
                                "p_value": float(mcn_p_val),
                                "contingency_table": table.tolist(),
                                "seed_used": seed,
                                "n_examples": len(labels)
                            }
                            break

                self.significance_tests[dataset][method] = test_result

            # Apply multiple comparison correction
            if p_values:
                reject, corrected_p, _, _ = multiple_comparison_correction(
                    p_values, alpha=alpha, method=correction
                )

                for i, method in enumerate(method_names):
                    self.significance_tests[dataset][method]["corrected_p_value"] = float(corrected_p[i])
                    self.significance_tests[dataset][method]["significant"] = bool(reject[i])
                    self.significance_tests[dataset][method]["correction_method"] = correction
                    self.significance_tests[dataset][method]["alpha"] = alpha

    def compute_all_statistics(self, n_bootstrap: int = 10000, correction: str = "bonferroni") -> None:
        """Compute all aggregate statistics and significance tests."""
        self.compute_aggregate_statistics(n_bootstrap=n_bootstrap)
        self.compute_significance_tests(correction=correction)

    def generate_paper_table(
        self,
        format: str = "markdown",
        metrics: List[str] = None,
        include_ci: bool = True,
        include_significance: bool = True
    ) -> str:
        """
        Generate publication-ready comparison table.

        Args:
            format: Output format ("markdown", "latex", "csv")
            metrics: Metrics to include (default: ["accuracy"])
            include_ci: Include 95% CI in output
            include_significance: Include significance stars

        Returns:
            Formatted table string
        """
        if not self.aggregated_results:
            self.compute_aggregate_statistics()

        metrics = metrics or ["accuracy"]

        # Build table data
        rows = []

        # Header
        header = ["Method"] + [ds.upper() for ds in self.datasets]
        if len(self.datasets) > 1:
            header.append("Average")

        rows.append(header)

        # Data rows for each method
        for method in self.methods:
            row = [method.replace("_", " ").title()]
            dataset_means = []

            for dataset in self.datasets:
                agg = self.aggregated_results.get(dataset, {}).get(method, {})

                if not agg:
                    row.append("--")
                    continue

                mean = agg.get("accuracy_mean", 0)
                std = agg.get("accuracy_std", 0)
                n = agg.get("n_seeds", 0)
                dataset_means.append(mean)

                # Format cell
                if include_ci and "accuracy_ci_lower" in agg:
                    ci_low = agg["accuracy_ci_lower"]
                    ci_high = agg["accuracy_ci_upper"]
                    cell = f"{mean:.1f} [{ci_low:.1f}, {ci_high:.1f}]"
                elif n > 1:
                    cell = f"{mean:.1f} +/- {std:.1f}"
                else:
                    cell = f"{mean:.1f}"

                # Add significance stars
                if include_significance and method != self.baseline_method:
                    sig_test = self.significance_tests.get(dataset, {}).get(method, {})
                    if sig_test.get("significant", False):
                        # Get raw p-value for star level
                        p_val = sig_test.get("corrected_p_value", 1.0)
                        cell += " " + p_value_to_stars(p_val)

                row.append(cell)

            # Average across datasets
            if len(self.datasets) > 1 and dataset_means:
                avg = np.mean(dataset_means)
                row.append(f"{avg:.1f}")

            rows.append(row)

        # Format output
        if format == "latex":
            return self._format_latex_table(rows, include_significance)
        elif format == "csv":
            return self._format_csv_table(rows)
        else:  # markdown
            return self._format_markdown_table(rows)

    def _format_markdown_table(self, rows: List[List[str]]) -> str:
        """Format as markdown table."""
        lines = []

        # Header
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")

        # Data rows
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        # Footer
        lines.append("")
        lines.append("*Note: Values show mean and 95% CI across seeds. ")
        lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05 vs baseline.*")

        return "\n".join(lines)

    def _format_latex_table(self, rows: List[List[str]], include_significance: bool) -> str:
        """Format as LaTeX table."""
        n_cols = len(rows[0])

        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\caption{Performance comparison across datasets}")
        lines.append("\\label{tab:results}")
        lines.append("\\begin{tabular}{l" + "c" * (n_cols - 1) + "}")
        lines.append("\\toprule")

        # Header
        header = " & ".join(rows[0]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Data rows
        for row in rows[1:]:
            # Escape special LaTeX characters
            escaped = [cell.replace("+/-", "$\\pm$").replace("_", "\\_") for cell in row]
            lines.append(" & ".join(escaped) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        if include_significance:
            lines.append("\\begin{tablenotes}")
            lines.append("\\small")
            lines.append("\\item Significance: $^{***}$ p<0.001, $^{**}$ p<0.01, $^{*}$ p<0.05")
            lines.append("\\end{tablenotes}")

        lines.append("\\end{table}")

        return "\n".join(lines)

    def _format_csv_table(self, rows: List[List[str]]) -> str:
        """Format as CSV."""
        lines = []
        for row in rows:
            # Clean up for CSV
            cleaned = [cell.replace(",", ";") for cell in row]
            lines.append(",".join(cleaned))
        return "\n".join(lines)

    def generate_significance_report(self) -> str:
        """Generate detailed significance test report."""
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL SIGNIFICANCE REPORT")
        lines.append("=" * 80)
        lines.append(f"\nExperiment: {self.name}")
        lines.append(f"Seeds: {self.seeds}")
        lines.append(f"Baseline: {self.baseline_method}")
        lines.append("")

        for dataset in self.datasets:
            lines.append("-" * 80)
            lines.append(f"DATASET: {dataset.upper()}")
            lines.append("-" * 80)

            # Baseline stats
            baseline_agg = self.aggregated_results.get(dataset, {}).get(self.baseline_method, {})
            if baseline_agg:
                lines.append(f"\nBaseline ({self.baseline_method}):")
                lines.append(f"  Mean: {baseline_agg.get('accuracy_mean', 0):.2f}%")
                lines.append(f"  Std: {baseline_agg.get('accuracy_std', 0):.2f}%")
                if "accuracy_ci_lower" in baseline_agg:
                    lines.append(f"  95% CI: [{baseline_agg['accuracy_ci_lower']:.2f}, {baseline_agg['accuracy_ci_upper']:.2f}]")
                lines.append(f"  Seeds: {baseline_agg.get('n_seeds', 0)}")

            # Comparisons
            lines.append("\nComparisons to baseline:")

            for method in self.methods:
                if method == self.baseline_method:
                    continue

                test = self.significance_tests.get(dataset, {}).get(method, {})
                agg = self.aggregated_results.get(dataset, {}).get(method, {})

                if not test or not agg:
                    continue

                lines.append(f"\n  {method}:")
                lines.append(f"    Mean: {agg.get('accuracy_mean', 0):.2f}%")
                lines.append(f"    Std: {agg.get('accuracy_std', 0):.2f}%")

                # Effect size interpretation
                d = test.get("cohens_d", 0)
                if abs(d) >= 0.8:
                    d_interp = "large"
                elif abs(d) >= 0.5:
                    d_interp = "medium"
                elif abs(d) >= 0.2:
                    d_interp = "small"
                else:
                    d_interp = "negligible"
                lines.append(f"    Cohen's d: {d:.3f} ({d_interp})")

                # T-test results
                if "paired_t_test" in test:
                    t_test = test["paired_t_test"]
                    lines.append(f"    Paired t-test:")
                    lines.append(f"      t = {t_test['t_statistic']:.3f}, df = {t_test['degrees_of_freedom']}")
                    lines.append(f"      p = {t_test['p_value']:.4f}")
                elif "independent_t_test" in test:
                    t_test = test["independent_t_test"]
                    lines.append(f"    Independent t-test (Welch's):")
                    lines.append(f"      t = {t_test['t_statistic']:.3f}, df = {t_test['degrees_of_freedom']:.1f}")
                    lines.append(f"      p = {t_test['p_value']:.4f}")

                # Corrected p-value
                if "corrected_p_value" in test:
                    lines.append(f"    Corrected p ({test.get('correction_method', 'unknown')}): {test['corrected_p_value']:.4f}")

                # Significance
                if test.get("significant", False):
                    lines.append(f"    --> SIGNIFICANT at alpha={test.get('alpha', 0.05)}")
                else:
                    lines.append(f"    --> NOT significant")

                # McNemar's test
                if "mcnemar_test" in test:
                    mcn = test["mcnemar_test"]
                    lines.append(f"    McNemar's test (seed {mcn['seed_used']}, n={mcn['n_examples']}):")
                    lines.append(f"      statistic = {mcn['statistic']:.3f}, p = {mcn['p_value']:.4f}")

            lines.append("")

        # Power analysis
        lines.append("=" * 80)
        lines.append("POWER ANALYSIS")
        lines.append("=" * 80)

        # Estimate pooled std from all conditions
        all_stds = []
        for ds in self.datasets:
            for method in self.methods:
                agg = self.aggregated_results.get(ds, {}).get(method, {})
                if "accuracy_std" in agg and agg["accuracy_std"] > 0:
                    all_stds.append(agg["accuracy_std"])

        if all_stds:
            pooled_std = np.mean(all_stds)
            lines.append(f"\nPooled standard deviation: {pooled_std:.2f}%")
            lines.append("\nSample sizes needed for 80% power at alpha=0.05:")

            for effect_pct in [2, 5, 10]:
                effect_d = effect_pct / pooled_std
                n_required = estimate_required_samples(
                    effect_size=effect_pct,
                    alpha=0.05,
                    power=0.80,
                    std_dev=pooled_std
                )
                lines.append(f"  {effect_pct}% effect (d={effect_d:.2f}): {n_required} seeds")

            current_n = len(self.seeds)
            lines.append(f"\nCurrent seeds: {current_n}")
            if current_n >= 5:
                lines.append("  Status: ADEQUATE for detecting medium effects")
            else:
                lines.append("  Status: MAY BE INSUFFICIENT - consider more seeds")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Export all results as a dictionary (for JSON serialization)."""
        return {
            "experiment_name": self.name,
            "config": {
                "datasets": self.datasets,
                "methods": self.methods,
                "seeds": self.seeds,
                "baseline_method": self.baseline_method
            },
            "per_seed_results": self.per_seed_results,
            "aggregated_results": self.aggregated_results,
            "significance_tests": self.significance_tests,
            "completion_status": self.get_completion_status(),
            "is_complete": self.is_complete()
        }

    def save_results(self, filename: str = None) -> str:
        """
        Save all results to JSON file.

        Args:
            filename: Output filename (default: {name}_results.json)

        Returns:
            Path to saved file
        """
        import os
        import json

        os.makedirs(self.output_dir, exist_ok=True)

        if filename is None:
            filename = f"{self.name}_results.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Also save tables and report
        tables_path = os.path.join(self.output_dir, f"{self.name}_tables.md")
        with open(tables_path, 'w') as f:
            f.write("# Results Tables\n\n")
            f.write("## Main Results (Markdown)\n\n")
            f.write(self.generate_paper_table(format="markdown"))
            f.write("\n\n## LaTeX Table\n\n```latex\n")
            f.write(self.generate_paper_table(format="latex"))
            f.write("\n```\n")

        report_path = os.path.join(self.output_dir, f"{self.name}_significance_report.txt")
        with open(report_path, 'w') as f:
            f.write(self.generate_significance_report())

        print(f"Results saved to:")
        print(f"  - {filepath}")
        print(f"  - {tables_path}")
        print(f"  - {report_path}")

        return filepath

    @classmethod
    def load_results(cls, filepath: str) -> 'MultiSeedExperiment':
        """
        Load experiment from saved JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            MultiSeedExperiment instance with loaded data
        """
        import json
        import os

        with open(filepath, 'r') as f:
            data = json.load(f)

        config = data["config"]
        experiment = cls(
            name=data["experiment_name"],
            datasets=config["datasets"],
            methods=config["methods"],
            seeds=config["seeds"],
            output_dir=os.path.dirname(filepath),
            baseline_method=config["baseline_method"]
        )

        experiment.per_seed_results = data["per_seed_results"]
        experiment.aggregated_results = data.get("aggregated_results", {})
        experiment.significance_tests = data.get("significance_tests", {})

        return experiment


# =============================================================================
# 9. WILSON SCORE CONFIDENCE INTERVAL
# =============================================================================

def wilson_score_ci(
    successes: int,
    total: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Wilson score confidence interval for binomial proportion.

    Better than normal approximation for small samples or extreme proportions.
    Recommended when accuracy is close to 0 or 1, or when sample size < 30.

    Args:
        successes: Number of correct predictions
        total: Total number of predictions
        confidence_level: Confidence level (default: 0.95)

    Returns:
        (lower_bound, upper_bound) as proportions

    Example:
        >>> # 85 correct out of 100 samples
        >>> lower, upper = wilson_score_ci(85, 100)
        >>> print(f"Accuracy: 85% [{lower*100:.1f}%, {upper*100:.1f}%]")
    """
    from scipy.stats import norm

    if total == 0:
        return (0.0, 0.0)

    p_hat = successes / total
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return (lower, upper)


# =============================================================================
# 10. CONVENIENCE FUNCTIONS FOR TELEPATHY EXPERIMENTS
# =============================================================================

def run_telepathy_statistical_validation(
    results_json_path: str,
    output_dir: str = "runs/statistical_validation",
    baseline_method: str = "prompt_tuning",
    correction: str = "bonferroni"
) -> MultiSeedExperiment:
    """
    Convenience function to run full statistical validation on Telepathy results.

    Args:
        results_json_path: Path to unified_results JSON from run_unified_comparison.py
        output_dir: Output directory for validation results
        baseline_method: Method to use as baseline for comparisons
        correction: Multiple testing correction method

    Returns:
        MultiSeedExperiment with all statistics computed

    Example:
        >>> experiment = run_telepathy_statistical_validation(
        ...     "runs/unified/unified_results_20250104.json",
        ...     baseline_method="mistral_zeroshot"
        ... )
        >>> print(experiment.generate_paper_table())
    """
    import json

    with open(results_json_path, 'r') as f:
        data = json.load(f)

    meta = data.get("meta", {})
    per_seed = data.get("per_seed_results", {})

    # Extract datasets and methods
    datasets = list(per_seed.keys())
    if not datasets:
        raise ValueError("No per-seed results found in JSON")

    # Get methods from first dataset/seed
    first_dataset = datasets[0]
    first_seed = list(per_seed[first_dataset].keys())[0]
    methods = [m for m in per_seed[first_dataset][first_seed].keys()
               if m not in ["random_chance"] and isinstance(per_seed[first_dataset][first_seed][m], dict)]

    seeds = meta.get("seeds", [42, 123, 456])

    # Create experiment
    experiment = MultiSeedExperiment(
        name="telepathy_validation",
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        output_dir=output_dir,
        baseline_method=baseline_method
    )

    # Add results
    for dataset in datasets:
        for seed_str, seed_results in per_seed[dataset].items():
            seed = int(seed_str)

            for method, method_results in seed_results.items():
                if method in ["random_chance"] or not isinstance(method_results, dict):
                    continue

                if "accuracy" in method_results:
                    experiment.add_result(
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        accuracy=method_results["accuracy"],
                        latency_ms=method_results.get("latency_ms"),
                        f1=method_results.get("f1")
                    )

    # Compute statistics
    experiment.compute_all_statistics(correction=correction)
    experiment.save_results()

    return experiment


def generate_paper_tables_from_json(
    results_json_path: str,
    output_dir: str = "paper_tables"
) -> Dict[str, str]:
    """
    Generate all paper-ready tables from unified results JSON.

    Args:
        results_json_path: Path to results JSON
        output_dir: Directory to save tables

    Returns:
        Dictionary of table names to table strings
    """
    experiment = run_telepathy_statistical_validation(
        results_json_path,
        output_dir=output_dir
    )

    tables = {
        "main_results_markdown": experiment.generate_paper_table(format="markdown"),
        "main_results_latex": experiment.generate_paper_table(format="latex"),
        "main_results_csv": experiment.generate_paper_table(format="csv"),
        "significance_report": experiment.generate_significance_report()
    }

    return tables
