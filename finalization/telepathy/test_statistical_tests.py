#!/usr/bin/env python3
"""
Test script for the statistical tests module
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from the module file to avoid telepathy package imports
from statistical_tests import (
    bootstrap_ci,
    mcnemar_test,
    bonferroni_correction,
    calculate_effect_size,
    determine_sample_size,
    analyze_unified_results,
    generate_comparison_table,
    generate_statistical_report
)

def test_bootstrap_ci():
    """Test bootstrap confidence intervals"""
    print("\n1. Testing Bootstrap CI")
    print("-" * 40)

    # Test with typical accuracy scores from 5 seeds
    scores = [75.2, 77.3, 74.8, 76.5, 75.9]
    mean, (ci_low, ci_high) = bootstrap_ci(scores, n_bootstrap=5000)

    print(f"Scores: {scores}")
    print(f"Mean: {mean:.2f}%")
    print(f"95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")

    # Test with only 3 seeds (common case)
    scores_3 = [75.2, 77.3, 74.8]
    mean_3, (ci_low_3, ci_high_3) = bootstrap_ci(scores_3, n_bootstrap=5000)
    print(f"\nWith only 3 seeds: {mean_3:.2f}% [{ci_low_3:.2f}%, {ci_high_3:.2f}%]")
    print("Note: CI is wider with fewer seeds")

    return True


def test_mcnemar():
    """Test McNemar's test for paired classification"""
    print("\n2. Testing McNemar's Test")
    print("-" * 40)

    np.random.seed(42)
    n_examples = 200

    # Simulate 4-class classification (AG News)
    true_labels = np.random.randint(0, 4, n_examples)

    # Bridge predictions (75% accuracy)
    bridge_preds = true_labels.copy()
    wrong_idx = np.random.choice(n_examples, size=50, replace=False)
    bridge_preds[wrong_idx] = (bridge_preds[wrong_idx] + 1) % 4

    # Baseline predictions (70% accuracy)
    baseline_preds = true_labels.copy()
    wrong_idx = np.random.choice(n_examples, size=60, replace=False)
    baseline_preds[wrong_idx] = (baseline_preds[wrong_idx] + 1) % 4

    stat, p_val, table = mcnemar_test(bridge_preds, baseline_preds, true_labels)

    print(f"Bridge accuracy: {100 * np.mean(bridge_preds == true_labels):.1f}%")
    print(f"Baseline accuracy: {100 * np.mean(baseline_preds == true_labels):.1f}%")
    print(f"\nContingency table:")
    print(f"              Baseline Correct | Baseline Wrong")
    print(f"Bridge Correct:    {table[0,0]:3d}         |    {table[0,1]:3d}")
    print(f"Bridge Wrong:      {table[1,0]:3d}         |    {table[1,1]:3d}")
    print(f"\nMcNemar statistic: {stat:.3f}")
    print(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        print("✓ Significant difference between methods")
    else:
        print("✗ No significant difference")

    return True


def test_bonferroni():
    """Test Bonferroni correction"""
    print("\n3. Testing Bonferroni Correction")
    print("-" * 40)

    # Simulate p-values from comparing Bridge to 5 baselines
    p_values = [0.001, 0.03, 0.04, 0.08, 0.15]
    baseline_names = ["Prompt-Tuning", "Llama-0shot", "Mistral-0shot", "Mistral-5shot", "Text-Relay"]

    reject, p_adj, alpha_adj = bonferroni_correction(p_values, alpha=0.05)

    print("Multiple comparison correction (5 baselines):")
    print(f"Original α = 0.05")
    print(f"Bonferroni-adjusted α = {alpha_adj:.4f}")
    print("\n{:<20} {:>10} {:>15} {:>10}".format("Baseline", "p-value", "p-adjusted", "Reject"))
    print("-" * 60)

    for name, p_raw, p_corr, rej in zip(baseline_names, p_values, p_adj, reject):
        print(f"{name:<20} {p_raw:>10.4f} {p_corr:>15.4f} {str(rej):>10}")

    print(f"\nSignificant comparisons: {sum(reject)}/{len(reject)}")

    return True


def test_effect_size():
    """Test effect size calculation"""
    print("\n4. Testing Effect Size (Cohen's d)")
    print("-" * 40)

    # Paired comparison (same test set, different methods)
    bridge_scores = np.array([75.2, 77.3, 74.8, 76.5, 75.9])
    baseline_scores = np.array([70.1, 71.5, 69.8, 70.8, 70.3])

    d_paired = calculate_effect_size(bridge_scores, baseline_scores, paired=True)

    print("Paired comparison (5 seeds):")
    print(f"Bridge: {np.mean(bridge_scores):.2f} ± {np.std(bridge_scores, ddof=1):.2f}")
    print(f"Baseline: {np.mean(baseline_scores):.2f} ± {np.std(baseline_scores, ddof=1):.2f}")
    print(f"Cohen's d = {d_paired:.3f}")

    # Interpret effect size
    if abs(d_paired) >= 0.8:
        interpretation = "large"
    elif abs(d_paired) >= 0.5:
        interpretation = "medium"
    elif abs(d_paired) >= 0.2:
        interpretation = "small"
    else:
        interpretation = "negligible"

    print(f"Interpretation: {interpretation} effect")

    # Independent samples
    method_a = np.random.normal(75, 3, 10)
    method_b = np.random.normal(70, 3, 10)
    d_indep = calculate_effect_size(method_a, method_b, paired=False)

    print(f"\nIndependent samples: d = {d_indep:.3f}")

    return True


def test_power_analysis():
    """Test power analysis and sample size determination"""
    print("\n5. Testing Power Analysis")
    print("-" * 40)

    print("Sample sizes needed for 80% power (α=0.05):")
    print("\n{:<20} {:>15} {:>15}".format("Effect Size", "Paired", "Independent"))
    print("-" * 50)

    for d in [0.2, 0.5, 0.8, 1.0]:
        n_paired = determine_sample_size(d, power=0.8, test_type='paired')
        n_indep = determine_sample_size(d, power=0.8, test_type='independent')

        if d == 0.2:
            desc = f"{d:.1f} (small)"
        elif d == 0.5:
            desc = f"{d:.1f} (medium)"
        elif d == 0.8:
            desc = f"{d:.1f} (large)"
        else:
            desc = f"{d:.1f} (very large)"

        print(f"{desc:<20} {n_paired:>15} {n_indep:>15}")

    print("\nInterpretation:")
    print("- Paired tests need fewer samples (same test set)")
    print("- Small effects require many seeds for detection")
    print("- With 3 seeds, can only detect large effects (d > 0.8)")

    return True


def test_unified_integration():
    """Test integration with unified comparison results"""
    print("\n6. Testing Unified Results Integration")
    print("-" * 40)

    # Create a mock results file
    mock_results = {
        "meta": {
            "timestamp": "20250104_120000",
            "seeds": [42, 123, 456],
            "soft_tokens": 8,
            "train_steps": 2000,
            "eval_samples": 200
        },
        "aggregated_results": {
            "sst2": {
                "bridge": {
                    "accuracy_mean": 75.3,
                    "accuracy_std": 1.8,
                    "num_seeds": 3
                },
                "mistral_zeroshot": {
                    "accuracy_mean": 70.2,
                    "accuracy_std": 1.5,
                    "num_seeds": 3
                },
                "prompt_tuning": {
                    "accuracy_mean": 68.5,
                    "accuracy_std": 2.1,
                    "num_seeds": 3
                }
            },
            "agnews": {
                "bridge": {
                    "accuracy_mean": 82.1,
                    "accuracy_std": 1.2,
                    "num_seeds": 3
                },
                "mistral_zeroshot": {
                    "accuracy_mean": 78.5,
                    "accuracy_std": 1.0,
                    "num_seeds": 3
                },
                "prompt_tuning": {
                    "accuracy_mean": 75.3,
                    "accuracy_std": 1.8,
                    "num_seeds": 3
                }
            }
        }
    }

    # Save mock results
    mock_path = Path("test_unified_results.json")
    with open(mock_path, 'w') as f:
        json.dump(mock_results, f, indent=2)

    # Test analysis
    try:
        analysis = analyze_unified_results(
            str(mock_path),
            baseline_method='mistral_zeroshot',
            target_method='bridge'
        )

        print("Analysis results:")
        for dataset, stats in analysis.items():
            print(f"\n{dataset.upper()}:")
            print(f"  Bridge: {stats['target_mean']:.1f}%")
            print(f"  Baseline: {stats['baseline_mean']:.1f}%")
            print(f"  Difference: {stats['difference']:+.1f}%")
            print(f"  Effect size: {stats['effect_size']:.2f}")
            print(f"  p-value: {stats['p_value']:.4f}")
            if stats['significant']:
                print("  ✓ Significant")

        # Test table generation
        print("\n" + "="*60)
        table = generate_comparison_table(str(mock_path))
        print(table)

    finally:
        # Clean up
        if mock_path.exists():
            mock_path.unlink()

    return True


def main():
    """Run all tests"""
    print("="*60)
    print("TELEPATHY STATISTICAL TESTS MODULE - TEST SUITE")
    print("="*60)

    tests = [
        ("Bootstrap CI", test_bootstrap_ci),
        ("McNemar's Test", test_mcnemar),
        ("Bonferroni Correction", test_bonferroni),
        ("Effect Size", test_effect_size),
        ("Power Analysis", test_power_analysis),
        ("Unified Integration", test_unified_integration)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} passed")
            else:
                failed += 1
                print(f"\n✗ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} failed with error: {e}")

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("All tests passed successfully! ✓")
    else:
        print(f"Warning: {failed} test(s) failed")

    print("="*60)


if __name__ == "__main__":
    main()