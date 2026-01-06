#!/usr/bin/env python3
"""
Test script to demonstrate enhanced statistical evaluation capabilities.

This script shows how the new statistical testing features work with
synthetic data to validate the implementation before running on real experiments.
"""

import numpy as np
import json
from latentwire.statistical_eval import (
    enhanced_bootstrap_ci,
    mcnemar_test_binary,
    cohens_d,
    paired_bootstrap_test,
    compute_pairwise_statistics
)


def generate_synthetic_results(n_samples=100, seed=42):
    """Generate synthetic evaluation results for testing."""
    np.random.seed(seed)

    # Simulate model performance (text baseline typically best)
    text_em = np.random.beta(8, 2, n_samples)  # Mean ~0.80
    text_f1 = np.random.beta(7.5, 2.5, n_samples)  # Mean ~0.75

    # Latent model (slightly worse)
    latent_em = np.random.beta(6, 4, n_samples)  # Mean ~0.60
    latent_f1 = np.random.beta(5.5, 4.5, n_samples)  # Mean ~0.55

    # Token budget (in between)
    trunc_em = np.random.beta(7, 3, n_samples)  # Mean ~0.70
    trunc_f1 = np.random.beta(6.5, 3.5, n_samples)  # Mean ~0.65

    return {
        'text': {'em': text_em, 'f1': text_f1},
        'latent': {'em': latent_em, 'f1': latent_f1},
        'token_budget': {'em': trunc_em, 'f1': trunc_f1}
    }


def test_bootstrap_ci():
    """Test enhanced bootstrap confidence intervals."""
    print("=" * 60)
    print("TEST: Bootstrap Confidence Intervals with BCa Correction")
    print("=" * 60)

    # Generate sample data
    scores = np.random.beta(7, 3, 50)  # Mean ~0.70

    # Compute CI with different methods
    methods = ['percentile', 'BCa']

    for method in methods:
        result = enhanced_bootstrap_ci(
            scores,
            confidence_level=0.95,
            n_resamples=10000,
            method=method,
            random_state=42
        )

        print(f"\nMethod: {method}")
        print(f"  Mean: {result['mean']:.4f}")
        print(f"  Std:  {result['std']:.4f}")
        print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  N samples: {result['n_samples']}")


def test_mcnemar():
    """Test McNemar's test for paired binary outcomes."""
    print("\n" + "=" * 60)
    print("TEST: McNemar's Test for Binary Classification")
    print("=" * 60)

    n = 100
    # Simulate predictions where model A is generally better
    correct_a = np.random.binomial(1, 0.75, n)  # 75% accuracy
    correct_b = np.random.binomial(1, 0.65, n)  # 65% accuracy

    # Add some correlation (same examples tend to be hard/easy)
    correlation = np.random.binomial(1, 0.3, n)
    correct_b[correlation == 1] = correct_a[correlation == 1]

    result = mcnemar_test_binary(correct_a, correct_b)

    print(f"\nContingency Table:")
    print(f"                Model B Correct | Model B Wrong")
    print(f"Model A Correct:     {result['contingency_table'][0][0]:3d}      |      {result['contingency_table'][0][1]:3d}")
    print(f"Model A Wrong:       {result['contingency_table'][1][0]:3d}      |      {result['contingency_table'][1][1]:3d}")
    print(f"\nTest Type: {result['test_type']}")
    print(f"Statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Model A better on {result['model_a_better']} examples")
    print(f"Model B better on {result['model_b_better']} examples")

    if result['p_value'] < 0.05:
        print("✓ Models are significantly different (p < 0.05)")
    else:
        print("✗ No significant difference between models")


def test_paired_bootstrap():
    """Test paired bootstrap test."""
    print("\n" + "=" * 60)
    print("TEST: Paired Bootstrap Test")
    print("=" * 60)

    n = 50
    # Create paired samples with known difference
    scores_a = np.random.beta(7, 3, n)  # Mean ~0.70
    scores_b = scores_a - 0.1 + np.random.normal(0, 0.05, n)  # Slightly worse
    scores_b = np.clip(scores_b, 0, 1)

    result = paired_bootstrap_test(
        scores_a,
        scores_b,
        n_resamples=10000,
        random_state=42
    )

    print(f"\nModel A mean: {np.mean(scores_a):.4f}")
    print(f"Model B mean: {np.mean(scores_b):.4f}")
    print(f"Observed difference: {result['observed_difference']:.4f}")
    print(f"Bootstrap 95% CI for difference: [{result['ci_difference'][0]:.4f}, {result['ci_difference'][1]:.4f}]")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['effect_size']:.3f}")

    if result['p_value'] < 0.05:
        print("✓ Significant difference (p < 0.05)")
    else:
        print("✗ No significant difference")

    # Interpret effect size
    d = abs(result['effect_size'])
    if d < 0.2:
        effect = "negligible"
    elif d < 0.5:
        effect = "small"
    elif d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"Effect size interpretation: {effect}")


def test_full_comparison():
    """Test full statistical comparison workflow."""
    print("\n" + "=" * 60)
    print("TEST: Complete Statistical Comparison")
    print("=" * 60)

    # Generate synthetic results
    results = generate_synthetic_results(n_samples=100)

    print("\nSynthetic Model Performance:")
    for model, scores in results.items():
        em_mean = np.mean(scores['em'])
        f1_mean = np.mean(scores['f1'])
        print(f"  {model:12s}: EM={em_mean:.3f}, F1={f1_mean:.3f}")

    # Perform comprehensive comparisons
    print("\n" + "-" * 40)
    print("Statistical Comparisons (vs Text Baseline)")
    print("-" * 40)

    for comparison in ['latent', 'token_budget']:
        print(f"\n{comparison.upper()} vs TEXT:")

        for metric in ['em', 'f1']:
            baseline_scores = results['text'][metric]
            compare_scores = results[comparison][metric]

            # Paired bootstrap test
            boot_result = paired_bootstrap_test(
                compare_scores,
                baseline_scores,
                n_resamples=5000,  # Fewer for demo
                random_state=42
            )

            # Effect size
            d = cohens_d(compare_scores, baseline_scores, paired=True)

            print(f"\n  {metric.upper()}:")
            print(f"    Difference: {boot_result['observed_difference']:+.4f}")
            print(f"    95% CI: [{boot_result['ci_difference'][0]:+.4f}, {boot_result['ci_difference'][1]:+.4f}]")
            print(f"    P-value: {boot_result['p_value']:.4f}", end="")

            if boot_result['p_value'] < 0.001:
                print(" ***")
            elif boot_result['p_value'] < 0.01:
                print(" **")
            elif boot_result['p_value'] < 0.05:
                print(" *")
            else:
                print(" (ns)")

            print(f"    Cohen's d: {d:.3f}", end="")

            # Interpret effect size
            abs_d = abs(d)
            if abs_d < 0.2:
                print(" (negligible)")
            elif abs_d < 0.5:
                print(" (small)")
            elif abs_d < 0.8:
                print(" (medium)")
            else:
                print(" (large)")


def test_output_format():
    """Test JSON output format for integration with eval.py."""
    print("\n" + "=" * 60)
    print("TEST: JSON Output Format")
    print("=" * 60)

    # Generate synthetic results
    results = generate_synthetic_results(n_samples=50)

    # Create output structure similar to eval.py
    output = {
        'models': ['llama'],
        'metrics': {},
        'statistical_analysis': {}
    }

    # Add basic metrics
    for model, scores in results.items():
        output['metrics'][model] = {
            'em': {
                'mean': float(np.mean(scores['em'])),
                'std': float(np.std(scores['em'], ddof=1))
            },
            'f1': {
                'mean': float(np.mean(scores['f1'])),
                'std': float(np.std(scores['f1'], ddof=1))
            }
        }

    # Add statistical comparisons
    for comparison in ['latent', 'token_budget']:
        comp_key = f"{comparison}_vs_text"
        output['statistical_analysis'][comp_key] = {}

        for metric in ['em', 'f1']:
            boot_result = paired_bootstrap_test(
                results[comparison][metric],
                results['text'][metric],
                n_resamples=1000,  # Fewer for demo
                random_state=42
            )

            output['statistical_analysis'][comp_key][metric] = {
                'difference': boot_result['observed_difference'],
                'p_value': boot_result['p_value'],
                'ci_95': boot_result['ci_difference'],
                'effect_size': boot_result['effect_size'],
                'significant': boot_result['p_value'] < 0.05
            }

    # Pretty print JSON
    print("\nJSON Output Structure:")
    print(json.dumps(output, indent=2))

    # Summary interpretation
    print("\n" + "-" * 40)
    print("Summary Interpretation:")
    print("-" * 40)

    for comp_key, comp_data in output['statistical_analysis'].items():
        print(f"\n{comp_key}:")
        for metric, stats in comp_data.items():
            sig_marker = "✓" if stats['significant'] else "✗"
            print(f"  {metric}: {sig_marker} p={stats['p_value']:.4f}, d={stats['effect_size']:.3f}")


if __name__ == '__main__':
    print("Enhanced Statistical Evaluation Test Suite")
    print("=" * 60)
    print("\nThis demonstrates the statistical testing capabilities")
    print("that are now integrated into eval.py for rigorous")
    print("model comparison and significance testing.")
    print()

    # Run all tests
    test_bootstrap_ci()
    test_mcnemar()
    test_paired_bootstrap()
    test_full_comparison()
    test_output_format()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print("\nThe evaluation now includes:")
    print("  • Bootstrap confidence intervals with BCa correction")
    print("  • McNemar's test for paired binary comparisons")
    print("  • Paired bootstrap tests for significance")
    print("  • Cohen's d effect size measurements")
    print("  • Multiple comparison corrections (when needed)")
    print("\nThese address reviewer concerns about statistical rigor.")