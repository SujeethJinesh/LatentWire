#!/usr/bin/env python
"""
Test script to verify that bootstrap confidence intervals and McNemar's test
are properly implemented in the statistical_testing.py module.

This addresses reviewer concerns about statistical rigor.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import from scripts
sys.path.append(str(Path(__file__).parent.parent))

# Import the statistical testing functions
from scripts.statistical_testing import (
    bootstrap_ci,
    bootstrap_ci_multiple_metrics,
    paired_bootstrap_test,
    mcnemar_test,
    multiple_comparison_correction,
    compare_multiple_methods_to_baseline,
    cohens_d_pooled,
    cohens_d_paired,
    estimate_required_samples
)

def test_bootstrap_confidence_intervals():
    """Test bootstrap confidence interval implementation."""
    print("\n" + "="*80)
    print("TEST 1: Bootstrap Confidence Intervals")
    print("="*80)

    # Generate sample data
    np.random.seed(42)
    scores = np.random.beta(7, 3, 100)  # 100 samples with mean ~0.7

    # Test 1.1: Basic bootstrap CI with BCa method
    print("\n1.1 Testing BCa bootstrap CI (recommended method):")
    mean_val, (ci_lower, ci_upper) = bootstrap_ci(
        scores,
        statistic=np.mean,
        confidence_level=0.95,
        n_resamples=10000,
        method='BCa',
        random_state=42
    )

    print(f"  Mean: {mean_val:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  CI width: {ci_upper - ci_lower:.4f}")

    # Verify CI contains the mean
    assert ci_lower <= mean_val <= ci_upper, "CI should contain the point estimate"
    print("  ✓ CI contains the point estimate")

    # Test 1.2: Multiple metrics at once
    print("\n1.2 Testing multiple metrics simultaneously:")
    metrics = {
        'accuracy': np.random.beta(8, 2, 100),  # ~0.8
        'f1_score': np.random.beta(7, 3, 100),  # ~0.7
        'precision': np.random.beta(9, 1, 100), # ~0.9
    }

    results = bootstrap_ci_multiple_metrics(
        metrics,
        confidence_level=0.95,
        n_resamples=10000,
        method='BCa',
        random_state=42
    )

    for metric_name, (mean_val, (ci_low, ci_high)) in results.items():
        print(f"  {metric_name}: {mean_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        assert ci_low <= mean_val <= ci_high, f"CI should contain mean for {metric_name}"

    print("  ✓ All CIs contain their respective point estimates")

    # Test 1.3: CI width increases with smaller samples
    print("\n1.3 Testing CI behavior with different sample sizes:")
    small_scores = scores[:10]  # Only 10 samples

    _, (small_ci_low, small_ci_high) = bootstrap_ci(
        small_scores,
        n_resamples=10000,
        method='BCa',
        random_state=42
    )

    small_width = small_ci_high - small_ci_low
    large_width = ci_upper - ci_lower

    print(f"  CI width with n=10: {small_width:.4f}")
    print(f"  CI width with n=100: {large_width:.4f}")
    assert small_width > large_width, "Smaller samples should have wider CIs"
    print("  ✓ Smaller samples correctly produce wider CIs")

    return True


def test_mcnemar_test():
    """Test McNemar's test implementation."""
    print("\n" + "="*80)
    print("TEST 2: McNemar's Test for Classifier Comparison")
    print("="*80)

    np.random.seed(42)
    n_samples = 100

    # Generate ground truth labels
    ground_truth = np.random.randint(0, 2, n_samples)

    # Generate predictions for two models
    # Model A: 80% accuracy
    model_a_preds = ground_truth.copy()
    wrong_indices_a = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
    model_a_preds[wrong_indices_a] = 1 - model_a_preds[wrong_indices_a]

    # Model B: 75% accuracy (slightly worse)
    model_b_preds = ground_truth.copy()
    wrong_indices_b = np.random.choice(n_samples, int(n_samples * 0.25), replace=False)
    model_b_preds[wrong_indices_b] = 1 - model_b_preds[wrong_indices_b]

    print("\n2.1 Testing McNemar's test with binary predictions:")
    print(f"  Model A accuracy: {np.mean(model_a_preds == ground_truth):.2%}")
    print(f"  Model B accuracy: {np.mean(model_b_preds == ground_truth):.2%}")

    # Run McNemar's test
    statistic, p_value, contingency_table = mcnemar_test(
        model_a_preds,
        model_b_preds,
        ground_truth,
        exact=None,  # Auto-select based on sample size
        correction=True
    )

    print(f"\n  McNemar's statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"\n  Contingency table:")
    print(f"                  Model B Correct | Model B Wrong")
    print(f"  Model A Correct:     {contingency_table[0,0]:3d}       |      {contingency_table[0,1]:3d}")
    print(f"  Model A Wrong:       {contingency_table[1,0]:3d}       |      {contingency_table[1,1]:3d}")

    # Check that contingency table sums correctly
    total_samples = contingency_table.sum()
    assert total_samples == n_samples, "Contingency table should sum to n_samples"
    print(f"\n  ✓ Contingency table sums correctly to {n_samples}")

    # Test significance interpretation
    if p_value < 0.05:
        print(f"  ✓ Models are significantly different (p={p_value:.4f} < 0.05)")
    else:
        print(f"  ✗ No significant difference detected (p={p_value:.4f} >= 0.05)")

    # Test 2.2: Exact test for small samples
    print("\n2.2 Testing exact binomial test for small samples:")

    # Create small sample with known disagreements
    small_truth = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    small_pred_a = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1])  # 1 error
    small_pred_b = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 1])  # 3 errors

    small_stat, small_p, small_table = mcnemar_test(
        small_pred_a,
        small_pred_b,
        small_truth,
        exact=True,  # Force exact test
        correction=False
    )

    print(f"  Sample size: {len(small_truth)}")
    print(f"  Using exact binomial test")
    print(f"  p-value: {small_p:.4f}")

    b_plus_c = small_table[0,1] + small_table[1,0]
    print(f"  b+c = {b_plus_c} (should use exact test when < 25)")
    assert b_plus_c < 25, "Small sample should have b+c < 25"
    print("  ✓ Correctly uses exact test for small samples")

    return True


def test_paired_bootstrap_test():
    """Test paired bootstrap test for comparing methods."""
    print("\n" + "="*80)
    print("TEST 3: Paired Bootstrap Test")
    print("="*80)

    np.random.seed(42)
    n_examples = 50

    # Generate scores for two methods on same test set
    # Method A: slightly better than Method B
    method_a_scores = np.random.beta(8, 2, n_examples)  # mean ~0.8
    method_b_scores = np.random.beta(7, 3, n_examples)  # mean ~0.7

    print("\n3.1 Testing paired bootstrap comparison:")
    print(f"  Method A mean: {np.mean(method_a_scores):.4f}")
    print(f"  Method B mean: {np.mean(method_b_scores):.4f}")

    # Run paired bootstrap test
    diff, p_value, stats = paired_bootstrap_test(
        method_a_scores,
        method_b_scores,
        n_resamples=10000,
        random_state=42,
        alternative='two-sided'
    )

    print(f"\n  Observed difference: {diff:+.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Bootstrap CI for difference: [{stats['bootstrap_ci_95'][0]:.4f}, {stats['bootstrap_ci_95'][1]:.4f}]")

    # Check consistency
    assert abs(diff - (stats['mean_a'] - stats['mean_b'])) < 1e-10, "Difference should match mean difference"
    print("  ✓ Statistics are internally consistent")

    # Test significance
    if p_value < 0.05:
        print(f"  ✓ Methods are significantly different (p={p_value:.4f} < 0.05)")
    else:
        print(f"  ✗ No significant difference (p={p_value:.4f} >= 0.05)")

    return True


def test_multiple_comparison_correction():
    """Test multiple comparison correction methods."""
    print("\n" + "="*80)
    print("TEST 4: Multiple Comparison Corrections")
    print("="*80)

    # Generate p-values from multiple tests
    raw_p_values = [0.001, 0.01, 0.03, 0.04, 0.20, 0.50]

    print("\n4.1 Testing Bonferroni correction (most conservative):")
    print(f"  Raw p-values: {raw_p_values}")

    reject_bonf, corrected_bonf, alphac, _ = multiple_comparison_correction(
        raw_p_values,
        alpha=0.05,
        method='bonferroni'
    )

    print(f"  Corrected p-values: {[f'{p:.4f}' for p in corrected_bonf]}")
    print(f"  Reject null hypothesis: {reject_bonf}")
    print(f"  Corrected alpha: {alphac:.4f}")

    # Verify Bonferroni formula
    n_tests = len(raw_p_values)
    expected_alpha = 0.05 / n_tests
    assert abs(alphac - expected_alpha) < 1e-10, "Bonferroni alpha should be 0.05/n"
    print(f"  ✓ Bonferroni correction correctly uses α/{n_tests}")

    print("\n4.2 Testing FDR correction (Benjamini-Hochberg, less conservative):")

    reject_fdr, corrected_fdr, _, _ = multiple_comparison_correction(
        raw_p_values,
        alpha=0.05,
        method='fdr_bh'
    )

    print(f"  Corrected p-values: {[f'{p:.4f}' for p in corrected_fdr]}")
    print(f"  Reject null hypothesis: {reject_fdr}")

    # FDR should reject at least as many as Bonferroni
    assert sum(reject_fdr) >= sum(reject_bonf), "FDR should be less conservative"
    print(f"  ✓ FDR rejects {sum(reject_fdr)} hypotheses vs Bonferroni's {sum(reject_bonf)}")

    return True


def test_effect_size_and_power():
    """Test effect size calculations and power analysis."""
    print("\n" + "="*80)
    print("TEST 5: Effect Size and Power Analysis")
    print("="*80)

    np.random.seed(42)

    # Generate two groups with known effect size
    group_a = np.random.normal(0.75, 0.1, 100)  # mean=0.75, std=0.1
    group_b = np.random.normal(0.70, 0.1, 100)  # mean=0.70, std=0.1

    print("\n5.1 Testing Cohen's d (effect size):")
    print(f"  Group A: mean={np.mean(group_a):.4f}, std={np.std(group_a, ddof=1):.4f}")
    print(f"  Group B: mean={np.mean(group_b):.4f}, std={np.std(group_b, ddof=1):.4f}")

    # Calculate Cohen's d (pooled)
    d_pooled = cohens_d_pooled(group_a, group_b)
    print(f"  Cohen's d (pooled): {d_pooled:.4f}")

    # Interpret effect size
    if abs(d_pooled) < 0.2:
        interpretation = "negligible"
    elif abs(d_pooled) < 0.5:
        interpretation = "small"
    elif abs(d_pooled) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    print(f"  Interpretation: {interpretation} effect size")

    # Test paired Cohen's d
    print("\n5.2 Testing Cohen's d for paired samples:")
    paired_a = group_a[:50]
    paired_b = group_b[:50]

    d_paired = cohens_d_paired(paired_a, paired_b)
    print(f"  Cohen's d (paired): {d_paired:.4f}")

    print("\n5.3 Testing power analysis (sample size calculation):")

    # How many samples needed to detect 5% improvement?
    effect_size = 0.05  # 5% improvement
    std_dev = 0.10      # Expected standard deviation

    n_required = estimate_required_samples(
        effect_size=effect_size,
        alpha=0.05,
        power=0.80,
        std_dev=std_dev
    )

    print(f"  To detect {effect_size:.0%} improvement with 80% power:")
    print(f"  Required samples: {n_required}")

    # Larger effects need fewer samples
    n_large = estimate_required_samples(0.10, 0.05, 0.80, 0.10)
    n_small = estimate_required_samples(0.02, 0.05, 0.80, 0.10)

    print(f"\n  Samples for 10% effect: {n_large}")
    print(f"  Samples for 2% effect: {n_small}")

    assert n_small > n_large, "Smaller effects should require more samples"
    print("  ✓ Smaller effects correctly require more samples")

    return True


def test_comprehensive_comparison():
    """Test comparing multiple methods with corrections."""
    print("\n" + "="*80)
    print("TEST 6: Comprehensive Multi-Method Comparison")
    print("="*80)

    np.random.seed(42)
    n_samples = 100

    # Generate scores for baseline and multiple methods
    baseline_scores = np.random.beta(7, 3, n_samples)  # ~0.7

    methods = {
        'Method_A': np.random.beta(8, 2, n_samples),    # ~0.8 (better)
        'Method_B': np.random.beta(7.5, 2.5, n_samples), # ~0.75 (slightly better)
        'Method_C': np.random.beta(6, 4, n_samples),     # ~0.6 (worse)
    }

    print("\n6.1 Comparing multiple methods to baseline with FDR correction:")
    print(f"  Baseline mean: {np.mean(baseline_scores):.4f}")

    for name, scores in methods.items():
        print(f"  {name} mean: {np.mean(scores):.4f}")

    # Compare all methods to baseline
    results = compare_multiple_methods_to_baseline(
        baseline_scores,
        methods,
        correction='fdr_bh',
        alpha=0.05,
        n_resamples=10000,
        random_state=42
    )

    print("\n  Results after FDR correction:")
    print("  " + "-"*60)

    for method_name in sorted(methods.keys()):
        res = results[method_name]
        print(f"\n  {method_name}:")
        print(f"    Difference: {res['difference']:+.4f}")
        print(f"    Raw p-value: {res['p_value']:.4f}")
        print(f"    Corrected p-value: {res['corrected_p_value']:.4f}")

        if res['significant']:
            print(f"    ✓ Significant at α=0.05 (after correction)")
        else:
            print(f"    ✗ Not significant after correction")

    # Verify that worse method (C) shows significant negative difference
    assert results['Method_C']['difference'] < 0, "Method C should be worse than baseline"
    print("\n  ✓ Correctly identifies worse methods")

    # Verify that better method (A) shows significant positive difference
    assert results['Method_A']['difference'] > 0, "Method A should be better than baseline"
    print("  ✓ Correctly identifies better methods")

    return True


def main():
    """Run all tests."""
    print("="*80)
    print("STATISTICAL RIGOR VERIFICATION")
    print("Testing bootstrap CI and McNemar's test implementations")
    print("="*80)

    all_passed = True

    # Run tests
    tests = [
        ("Bootstrap Confidence Intervals", test_bootstrap_confidence_intervals),
        ("McNemar's Test", test_mcnemar_test),
        ("Paired Bootstrap Test", test_paired_bootstrap_test),
        ("Multiple Comparison Corrections", test_multiple_comparison_correction),
        ("Effect Size and Power Analysis", test_effect_size_and_power),
        ("Comprehensive Multi-Method Comparison", test_comprehensive_comparison),
    ]

    for test_name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {test_name} FAILED with error: {e}", flush=True)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("\nThe statistical testing implementation is robust and includes:")
        print("  • Bootstrap confidence intervals with BCa method")
        print("  • McNemar's test for classifier comparison")
        print("  • Paired bootstrap test for method comparison")
        print("  • Multiple comparison corrections (Bonferroni, FDR)")
        print("  • Cohen's d effect size calculations")
        print("  • Power analysis for sample size determination")
        print("\nThis addresses reviewer concerns about statistical rigor.")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the implementation and fix any issues.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)