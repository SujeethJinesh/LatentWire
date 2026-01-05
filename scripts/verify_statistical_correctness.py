#!/usr/bin/env python
"""
Verification Script for Statistical Testing Module Correctness

This script verifies the correctness of statistical implementations in:
- telepathy/statistical_tests.py
- scripts/statistical_testing.py
- telepathy/aggregate_results.py

Tests against scipy.stats documentation and known statistical formulas.
"""

import numpy as np
import sys
import warnings
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
from scipy.stats import binom, t, chi2, norm
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
import json

# Add project root to path
sys.path.append('.')

# Import our implementations
from telepathy.statistical_tests import (
    bootstrap_ci as telepathy_bootstrap_ci,
    mcnemar_test as telepathy_mcnemar,
    bonferroni_correction as telepathy_bonferroni,
    calculate_effect_size as telepathy_cohens_d,
    determine_sample_size as telepathy_power
)

from scripts.statistical_testing import (
    bootstrap_ci as scripts_bootstrap_ci,
    paired_ttest,
    independent_ttest,
    cohens_d_pooled,
    cohens_d_paired,
    mcnemar_test as scripts_mcnemar,
    multiple_comparison_correction,
    estimate_required_samples
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_test(test_name, passed, details=""):
    """Print test result with formatting."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print("{}: {}".format(test_name, status))
    if details:
        print("  {}".format(details))

# =============================================================================
# 1. BOOTSTRAP CI TESTS
# =============================================================================

def test_bootstrap_ci():
    """Test Bootstrap CI implementation against scipy.stats.bootstrap."""
    print_section("1. BOOTSTRAP CONFIDENCE INTERVALS")

    # Test data
    np.random.seed(42)
    data = np.random.normal(0.75, 0.1, 100)

    # Test 1: BCa method with sufficient samples
    print("\nTest 1.1: BCa method (n=100)")

    # Our implementation (telepathy)
    tel_mean, (tel_low, tel_high) = telepathy_bootstrap_ci(
        data, n_bootstrap=10000, confidence_level=0.95, method='BCa', random_state=42
    )

    # Direct scipy implementation for comparison
    rng = np.random.default_rng(42)
    scipy_res = scipy_bootstrap(
        (data,),
        np.mean,
        n_resamples=10000,
        confidence_level=0.95,
        method='bca',
        random_state=rng
    )
    scipy_low = scipy_res.confidence_interval.low
    scipy_high = scipy_res.confidence_interval.high

    # Check if results are close (within tolerance due to randomness)
    tolerance = 0.01
    ci_match = (abs(tel_low - scipy_low) < tolerance and
                abs(tel_high - scipy_high) < tolerance)

    print_test("BCa CI matches scipy", ci_match,
               "Telepathy: [{tel_low:.4f}, {tel_high:.4f}], ".format()
               "Scipy: [{scipy_low:.4f}, {scipy_high:.4f}]".format())

    # Test 2: Wider CI with fewer samples
    print("\nTest 1.2: CI width increases with fewer samples")

    small_data = data[:10]
    _, (small_low, small_high) = telepathy_bootstrap_ci(
        small_data, n_bootstrap=10000, random_state=42
    )

    small_width = small_high - small_low
    large_width = tel_high - tel_low

    width_test = small_width > large_width
    print_test("Smaller sample → wider CI", width_test,
               "n=10 width: {small_width:.4f}, n=100 width: {large_width:.4f}".format())

    # Test 3: Warning for too few samples
    print("\nTest 1.3: Warning for insufficient samples")

    tiny_data = data[:2]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tel_mean_tiny, (tel_low_tiny, tel_high_tiny) = telepathy_bootstrap_ci(
            tiny_data, n_bootstrap=1000, random_state=42
        )
        warning_raised = len(w) > 0 and "samples" in str(w[0].message).lower()

    print_test("Warning for n<3", warning_raised,
               "Correctly warns about insufficient samples")

    return ci_match and width_test and warning_raised

# =============================================================================
# 2. MCNEMAR TEST
# =============================================================================

def test_mcnemar():
    """Test McNemar's test implementation."""
    print_section("2. MCNEMAR'S TEST")

    np.random.seed(42)

    # Test 1: Exact test for small samples (n < 25)
    print("\nTest 2.1: Exact binomial test for small samples")

    # Create predictions with known contingency table
    n = 50
    labels = np.array([0, 1] * 25)
    pred1 = labels.copy()
    pred2 = labels.copy()

    # Make pred1 correct on 5 examples where pred2 is wrong
    # Make pred2 correct on 2 examples where pred1 is wrong
    pred1[:5] = 1 - pred1[:5]  # pred1 wrong on first 5
    pred2[5:7] = 1 - pred2[5:7]  # pred2 wrong on positions 5-6
    pred1[5:7] = labels[5:7]  # pred1 correct where pred2 wrong
    pred2[:5] = labels[:5]  # pred2 correct where pred1 wrong

    # Our implementation
    tel_stat, tel_p, tel_table = telepathy_mcnemar(pred1, pred2, labels)

    # Manual calculation for verification
    b = np.sum((pred1 == labels) & (pred2 != labels))  # pred1 correct, pred2 wrong
    c = np.sum((pred1 != labels) & (pred2 == labels))  # pred1 wrong, pred2 correct

    print("  Contingency: b={b} (1 correct, 2 wrong), c={c} (1 wrong, 2 correct)".format())

    # For small samples (b+c < 25), should use exact binomial test
    if b + c < 25:
        # Exact binomial test
        expected_p = 2 * binom.cdf(min(b, c), b + c, 0.5)
        exact_test_used = abs(tel_p - expected_p) < 0.01
        print_test("Uses exact test for n<25", exact_test_used,
                   "p-value: {tel_p:.4f}, Expected: {expected_p:.4f}".format())
    else:
        print_test("Uses exact test for n<25", False, "Sample too large for this test")

    # Test 2: Chi-square with continuity correction for large samples
    print("\nTest 2.2: Chi-square test for large samples")

    # Create larger sample
    n_large = 200
    labels_large = np.random.randint(0, 2, n_large)
    pred1_large = labels_large.copy()
    pred2_large = labels_large.copy()

    # Create disagreements: pred1 correct on 30, pred2 correct on 20
    disagree_indices = np.random.choice(n_large, 50, replace=False)
    pred1_large[disagree_indices[:30]] = 1 - pred1_large[disagree_indices[:30]]
    pred2_large[disagree_indices[30:]] = 1 - pred2_large[disagree_indices[30:]]

    scripts_stat, scripts_p, scripts_table = scripts_mcnemar(
        pred1_large, pred2_large, labels_large
    )

    # Build contingency table manually
    correct1 = (pred1_large == labels_large)
    correct2 = (pred2_large == labels_large)
    b_large = np.sum(correct1 & ~correct2)
    c_large = np.sum(~correct1 & correct2)

    # McNemar chi-square with continuity correction
    if b_large + c_large >= 25:
        expected_stat = (abs(b_large - c_large) - 1) ** 2 / (b_large + c_large)
        expected_p = 1 - chi2.cdf(expected_stat, 1)

        chi2_correct = abs(scripts_p - expected_p) < 0.01
        print_test("Chi-square formula correct", chi2_correct,
                   "b={b_large}, c={c_large}, p={scripts_p:.4f}".format())

    # Test 3: Perfect agreement case
    print("\nTest 2.3: Perfect agreement handling")

    pred_same = labels.copy()
    tel_stat_same, tel_p_same, _ = telepathy_mcnemar(pred_same, pred_same, labels)

    perfect_agreement = tel_p_same == 1.0 and tel_stat_same == 0.0
    print_test("Perfect agreement → p=1.0", perfect_agreement,
               "Statistic: {tel_stat_same}, p-value: {tel_p_same}".format())

    return exact_test_used and chi2_correct and perfect_agreement

# =============================================================================
# 3. BONFERRONI CORRECTION
# =============================================================================

def test_bonferroni():
    """Test Bonferroni correction implementation."""
    print_section("3. BONFERRONI CORRECTION")

    # Test 1: Basic Bonferroni formula
    print("\nTest 3.1: Bonferroni formula")

    p_values = [0.01, 0.03, 0.04, 0.08, 0.15]
    alpha = 0.05

    # Our implementation
    tel_reject, tel_p_adj, tel_alpha_adj = telepathy_bonferroni(p_values, alpha)

    # Manual calculation
    n_tests = len(p_values)
    expected_alpha_adj = alpha / n_tests
    expected_p_adj = [min(p * n_tests, 1.0) for p in p_values]
    expected_reject = [p < alpha for p in expected_p_adj]

    formula_correct = (
        abs(tel_alpha_adj - expected_alpha_adj) < 1e-10 and
        all(abs(a - b) < 1e-10 for a, b in zip(tel_p_adj, expected_p_adj))
    )

    print_test("Bonferroni formula", formula_correct,
               "Adjusted alpha: {tel_alpha_adj:.4f} (expected: {expected_alpha_adj:.4f})".format())

    # Test 2: Rejection decisions
    print("\nTest 3.2: Correct rejection decisions")

    reject_correct = tel_reject == expected_reject
    print_test("Rejection decisions", reject_correct,
               "Reject: {tel_reject}".format())

    # Test 3: P-values capped at 1.0
    print("\nTest 3.3: P-values capped at 1.0")

    large_p = [0.5, 0.8, 0.9]
    _, large_p_adj, _ = telepathy_bonferroni(large_p, alpha)

    capped = all(p <= 1.0 for p in large_p_adj)
    print_test("P-values ≤ 1.0", capped,
               "Adjusted p-values: {large_p_adj}".format())

    # Test 4: Compare with statsmodels
    print("\nTest 3.4: Match statsmodels multipletests")

    from statsmodels.stats.multitest import multipletests

    sm_reject, sm_p_adj, sm_alpha_c, _ = multipletests(
        p_values, alpha=alpha, method='bonferroni'
    )

    statsmodels_match = (
        np.allclose(tel_p_adj, sm_p_adj) and
        abs(tel_alpha_adj - sm_alpha_c) < 1e-10
    )

    print_test("Matches statsmodels", statsmodels_match)

    return formula_correct and reject_correct and capped and statsmodels_match

# =============================================================================
# 4. COHEN'S D EFFECT SIZE
# =============================================================================

def test_cohens_d():
    """Test Cohen's d effect size calculations."""
    print_section("4. COHEN'S D EFFECT SIZE")

    np.random.seed(42)

    # Test 1: Pooled Cohen's d (independent samples)
    print("\nTest 4.1: Pooled Cohen's d formula")

    group1 = np.array([75.0, 77.0, 73.0, 76.0, 74.0])
    group2 = np.array([70.0, 72.0, 68.0, 71.0, 69.0])

    # Our implementation
    d_pooled = cohens_d_pooled(group1, group2)

    # Manual calculation
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    expected_d = (mean1 - mean2) / pooled_std

    pooled_correct = abs(d_pooled - expected_d) < 1e-10
    print_test("Pooled formula correct", pooled_correct,
               "d = {d_pooled:.4f} (expected: {expected_d:.4f})".format())

    # Test 2: Paired Cohen's d
    print("\nTest 4.2: Paired Cohen's d formula")

    # Paired data (same subjects, two conditions)
    scores_a = np.array([75.0, 77.0, 73.0])
    scores_b = np.array([70.0, 72.0, 68.0])

    d_paired = cohens_d_paired(scores_a, scores_b)

    # Manual calculation for paired
    diffs = scores_a - scores_b
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    expected_d_paired = mean_diff / std_diff

    paired_correct = abs(d_paired - expected_d_paired) < 1e-10
    print_test("Paired formula correct", paired_correct,
               "d = {d_paired:.4f} (expected: {expected_d_paired:.4f})".format())

    # Test 3: Effect size interpretation
    print("\nTest 4.3: Effect size magnitude")

    # Large effect (d > 0.8)
    large_group1 = np.array([80.0, 82.0, 78.0, 81.0, 79.0])
    large_group2 = np.array([70.0, 72.0, 68.0, 71.0, 69.0])
    d_large = cohens_d_pooled(large_group1, large_group2)

    large_effect = abs(d_large) >= 0.8
    print_test("Detects large effect", large_effect,
               "d = {d_large:.4f} (|d| ≥ 0.8 is large)".format())

    # Test 4: Zero variance handling
    print("\nTest 4.4: Zero variance handling")

    const_group1 = np.array([75.0, 75.0, 75.0])
    const_group2 = np.array([70.0, 70.0, 70.0])

    # Should handle zero variance gracefully
    try:
        # Using telepathy implementation for paired
        d_const = telepathy_cohens_d(const_group1, const_group2, paired=True)
        if np.isinf(d_const) and np.mean(const_group1) != np.mean(const_group2):
            zero_var_handled = True
        elif d_const == 0.0 and np.mean(const_group1) == np.mean(const_group2):
            zero_var_handled = True
        else:
            zero_var_handled = False
    except:
        zero_var_handled = False

    print_test("Handles zero variance", zero_var_handled,
               "Returns inf for different means, 0 for same means")

    return pooled_correct and paired_correct and large_effect and zero_var_handled

# =============================================================================
# 5. POWER ANALYSIS
# =============================================================================

def test_power_analysis():
    """Test power analysis and sample size calculations."""
    print_section("5. POWER ANALYSIS")

    # Test 1: Sample size formula
    print("\nTest 5.1: Sample size calculation formula")

    effect_size = 0.5  # Medium effect (Cohen's d)
    power = 0.8
    alpha = 0.05

    # Our implementation (telepathy)
    n_tel = telepathy_power(effect_size, power, alpha, test_type='paired')

    # Manual calculation
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = norm.ppf(power)
    n_expected = ((z_alpha + z_beta) / effect_size) ** 2
    n_expected = int(np.ceil(n_expected))

    formula_correct = abs(n_tel - n_expected) <= 1
    print_test("Power formula correct", formula_correct,
               "n = {n_tel} (expected: {n_expected})".format())

    # Test 2: Larger effect = smaller sample needed
    print("\nTest 5.2: Effect size vs sample size relationship")

    n_small_effect = telepathy_power(0.2, power, alpha, test_type='paired')
    n_medium_effect = telepathy_power(0.5, power, alpha, test_type='paired')
    n_large_effect = telepathy_power(0.8, power, alpha, test_type='paired')

    relationship_correct = n_small_effect > n_medium_effect > n_large_effect
    print_test("Larger effect → smaller n", relationship_correct,
               "d=0.2: n={n_small_effect}, d=0.5: n={n_medium_effect}, d=0.8: n={n_large_effect}".format())

    # Test 3: Independent vs paired samples
    print("\nTest 5.3: Independent samples need more data")

    n_paired = telepathy_power(0.5, power, alpha, test_type='paired')
    n_independent = telepathy_power(0.5, power, alpha, test_type='independent')

    paired_more_efficient = n_independent > n_paired
    print_test("Independent needs more samples", paired_more_efficient,
               "Paired: n={n_paired}, Independent: n={n_independent}".format())

    # Test 4: Alternative implementation (scripts)
    print("\nTest 5.4: Alternative implementation consistency")

    effect_abs = 0.05  # 5% absolute difference
    std_dev = 0.1
    effect_d = effect_abs / std_dev

    n_scripts = estimate_required_samples(effect_abs, alpha, power, std_dev)

    # Should be approximately 2x the paired sample size (for independent)
    n_check = telepathy_power(effect_d, power, alpha, test_type='independent')

    implementations_consistent = abs(n_scripts - n_check) <= 2
    print_test("Implementations consistent", implementations_consistent,
               "Scripts: n={n_scripts}, Telepathy: n={n_check}".format())

    return formula_correct and relationship_correct and paired_more_efficient and implementations_consistent

# =============================================================================
# 6. T-TESTS
# =============================================================================

def test_t_tests():
    """Test t-test implementations."""
    print_section("6. T-TESTS (PAIRED AND INDEPENDENT)")

    np.random.seed(42)

    # Test 1: Paired t-test
    print("\nTest 6.1: Paired t-test")

    scores_a = np.array([75.0, 77.0, 73.0, 76.0, 74.0])
    scores_b = np.array([70.0, 72.0, 68.0, 71.0, 69.0])

    # Our implementation
    diff, p_val, stats_dict = paired_ttest(scores_a, scores_b)

    # Scipy verification
    scipy_t, scipy_p = stats.ttest_rel(scores_a, scores_b)

    paired_correct = (
        abs(stats_dict['t_statistic'] - scipy_t) < 1e-10 and
        abs(p_val - scipy_p) < 1e-10
    )

    print_test("Paired t-test matches scipy", paired_correct,
               "t={stats_dict['t_statistic']:.4f}, p={p_val:.4f}".format())

    # Test 2: Independent t-test (equal variance)
    print("\nTest 6.2: Independent t-test (equal variance)")

    group1 = np.random.normal(0.75, 0.1, 30)
    group2 = np.random.normal(0.70, 0.1, 30)

    diff_ind, p_val_ind, stats_ind = independent_ttest(
        group1, group2, equal_var=True
    )

    scipy_t_ind, scipy_p_ind = stats.ttest_ind(group1, group2, equal_var=True)

    independent_correct = (
        abs(stats_ind['t_statistic'] - scipy_t_ind) < 1e-10 and
        abs(p_val_ind - scipy_p_ind) < 1e-10
    )

    print_test("Independent t-test matches scipy", independent_correct,
               "t={stats_ind['t_statistic']:.4f}, p={p_val_ind:.4f}".format())

    # Test 3: Welch's t-test (unequal variance)
    print("\nTest 6.3: Welch's t-test (unequal variance)")

    group1_var = np.random.normal(0.75, 0.15, 25)  # Higher variance
    group2_var = np.random.normal(0.70, 0.05, 35)  # Lower variance

    diff_welch, p_val_welch, stats_welch = independent_ttest(
        group1_var, group2_var, equal_var=False
    )

    scipy_t_welch, scipy_p_welch = stats.ttest_ind(
        group1_var, group2_var, equal_var=False
    )

    welch_correct = (
        abs(stats_welch['t_statistic'] - scipy_t_welch) < 1e-10 and
        abs(p_val_welch - scipy_p_welch) < 1e-10
    )

    print_test("Welch's t-test matches scipy", welch_correct,
               "t={stats_welch['t_statistic']:.4f}, p={p_val_welch:.4f}".format())

    # Test 4: Degrees of freedom calculation
    print("\nTest 6.4: Degrees of freedom")

    # Paired: df = n - 1
    n_paired = len(scores_a)
    df_paired_expected = n_paired - 1
    df_paired_correct = stats_dict['degrees_of_freedom'] == df_paired_expected

    # Independent equal var: df = n1 + n2 - 2
    df_ind_expected = len(group1) + len(group2) - 2
    df_ind_correct = abs(stats_ind['degrees_of_freedom'] - df_ind_expected) < 1e-10

    print_test("Degrees of freedom correct", df_paired_correct and df_ind_correct,
               "Paired df={stats_dict['degrees_of_freedom']}, ".format()
               "Independent df={stats_ind['degrees_of_freedom']:.1f}".format())

    return paired_correct and independent_correct and welch_correct and (df_paired_correct and df_ind_correct)

# =============================================================================
# 7. INTEGRATION TESTS
# =============================================================================

def test_integration():
    """Test integration between modules."""
    print_section("7. INTEGRATION TESTS")

    # Test that telepathy and scripts implementations give consistent results
    print("\nTest 7.1: Consistency between implementations")

    np.random.seed(42)
    data = np.random.normal(0.75, 0.1, 50)

    # Bootstrap CI from both
    tel_mean, (tel_low, tel_high) = telepathy_bootstrap_ci(
        data, n_bootstrap=5000, method='BCa', random_state=42
    )

    scripts_mean, (scripts_low, scripts_high) = scripts_bootstrap_ci(
        data, n_resamples=5000, method='BCa', random_state=42
    )

    bootstrap_consistent = (
        abs(tel_mean - scripts_mean) < 0.01 and
        abs(tel_low - scripts_low) < 0.02 and
        abs(tel_high - scripts_high) < 0.02
    )

    print_test("Bootstrap implementations consistent", bootstrap_consistent,
               "Telepathy: [{tel_low:.3f}, {tel_high:.3f}], ".format()
               "Scripts: [{scripts_low:.3f}, {scripts_high:.3f}]".format())

    # Test effect size consistency
    print("\nTest 7.2: Effect size consistency")

    group1 = np.array([75.0, 77.0, 73.0, 76.0, 74.0])
    group2 = np.array([70.0, 72.0, 68.0, 71.0, 69.0])

    tel_d = telepathy_cohens_d(group1, group2, paired=False)
    scripts_d = cohens_d_pooled(group1, group2)

    effect_consistent = abs(tel_d - scripts_d) < 1e-10
    print_test("Effect size implementations consistent", effect_consistent,
               "Telepathy: {tel_d:.4f}, Scripts: {scripts_d:.4f}".format())

    return bootstrap_consistent and effect_consistent

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all verification tests."""
    print("=" * 80)
    print("STATISTICAL TESTING MODULE VERIFICATION")
    print("=" * 80)
    print("\nVerifying correctness against scipy.stats documentation...")

    all_passed = True
    results = {}

    # Run all test suites
    test_functions = [
        ("Bootstrap CI", test_bootstrap_ci),
        ("McNemar Test", test_mcnemar),
        ("Bonferroni Correction", test_bonferroni),
        ("Cohen's d", test_cohens_d),
        ("Power Analysis", test_power_analysis),
        ("T-Tests", test_t_tests),
        ("Integration", test_integration)
    ]

    for name, test_func in test_functions:
        try:
            passed = test_func()
            results[name] = passed
            all_passed = all_passed and passed
        except Exception as e:
            print("\n❌ {name} tests failed with error: {e}".format())
            results[name] = False
            all_passed = False

    # Summary
    print_section("SUMMARY")
    print("\nTest Results:")
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print("  {name}: {status}".format())

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Statistical implementations are correct!")
        print("\nKey findings:")
        print("1. Bootstrap CI: Correctly implements BCa method, matches scipy.stats")
        print("2. McNemar test: Uses exact test for n<25, chi-square for larger samples")
        print("3. Bonferroni: Properly adjusts alpha by num_comparisons")
        print("4. Cohen's d: Accurate for both pooled and paired variants")
        print("5. Power analysis: Formula validated against statistical theory")
        print("6. T-tests: Match scipy.stats for paired, independent, and Welch's")
        print("7. Integration: Implementations are consistent across modules")
    else:
        print("❌ SOME TESTS FAILED - Please review the issues above")
        failed = [name for name, passed in results.items() if not passed]
        print("\nFailed test suites: {', '.join(failed)}".format())

    print("=" * 80)

    # Save verification results
    verification_results = {
        "timestamp": str(np.datetime64('now')),
        "all_passed": all_passed,
        "test_results": results,
        "scipy_version": stats.__version__,
        "numpy_version": np.__version__
    }

    with open('telepathy/statistical_verification_results.json', 'w') as f:
        json.dump(verification_results, f, indent=2)

    print("\nVerification results saved to: telepathy/statistical_verification_results.json")

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())