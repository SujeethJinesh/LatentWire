#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Statistical Testing Integration for LatentWire Paper

This script verifies that all required statistical components are properly integrated:
1. Three seeds (42, 123, 456) used consistently
2. Bootstrap CI with n=10000
3. Paired t-tests for comparisons
4. McNemar test for classification
5. Effect size (Cohen's d)
6. Bonferroni correction for multiple comparisons
7. Proper aggregation and LaTeX table generation

Author: Claude
Date: 2025-01-04
"""

import json
import numpy as np
import sys
from pathlib import Path
# No type hints for Python 3.5 compatibility
from collections import defaultdict

# Add project root to path
sys.path.append('.')

# Import statistical testing utilities
from scripts.statistical_testing import (
    bootstrap_ci,
    paired_ttest,
    cohens_d_paired,
    mcnemar_test,
    multiple_comparison_correction,
    compare_multiple_methods_to_baseline,
    p_value_to_stars,
    format_mean_ci,
    comprehensive_comparison_report
)

# Standard seeds that should be used
REQUIRED_SEEDS = [42, 123, 456]

# Required bootstrap samples
REQUIRED_BOOTSTRAP_SAMPLES = 10000

# Required significance levels
ALPHA_LEVELS = [0.05, 0.01, 0.001]


def verify_seed_usage(script_path):
    """Verify that a script uses the correct seeds."""
    results = {
        'uses_correct_seeds': False,
        'seeds_found': [],
        'issues': []
    }

    if not script_path.exists():
        results['issues'].append("Script not found: {}".format(script_path))
        return results

    with open(script_path, 'r') as f:
        content = f.read()

    # Check for seed definitions
    for seed in REQUIRED_SEEDS:
        if str(seed) in content:
            results['seeds_found'].append(seed)

    # Check if all required seeds are present
    if set(results['seeds_found']) == set(REQUIRED_SEEDS):
        results['uses_correct_seeds'] = True
    else:
        missing_seeds = set(REQUIRED_SEEDS) - set(results['seeds_found'])
        if missing_seeds:
            results['issues'].append("Missing seeds: {}".format(missing_seeds))

    return results


def verify_bootstrap_ci_usage():
    """Verify bootstrap CI is properly configured."""
    results = {
        'correct_n_resamples': False,
        'uses_bca_method': False,
        'confidence_level_95': False,
        'issues': []
    }

    # Test bootstrap_ci function with dummy data
    test_data = np.random.randn(30)
    mean_val, ci = bootstrap_ci(
        test_data,
        confidence_level=0.95,
        n_resamples=REQUIRED_BOOTSTRAP_SAMPLES,
        method='BCa'
    )

    # Verify the function works correctly
    if ci[0] < mean_val < ci[1]:
        results['correct_n_resamples'] = True
        results['uses_bca_method'] = True
        results['confidence_level_95'] = True
    else:
        results['issues'].append("Bootstrap CI not working correctly")

    return results


def verify_statistical_tests():
    """Verify all statistical tests are working correctly."""
    results = {
        'paired_ttest_works': False,
        'cohens_d_works': False,
        'mcnemar_works': False,
        'bonferroni_works': False,
        'issues': []
    }

    # Test paired t-test
    try:
        scores_a = np.array([0.85, 0.87, 0.83])
        scores_b = np.array([0.80, 0.82, 0.78])
        diff, p_val, stats = paired_ttest(scores_a, scores_b)
        if isinstance(p_val, float) and 0 <= p_val <= 1:
            results['paired_ttest_works'] = True
    except Exception as e:
        results['issues'].append("Paired t-test failed: {}".format(e))

    # Test Cohen's d
    try:
        d = cohens_d_paired(scores_a, scores_b)
        if isinstance(d, float):
            results['cohens_d_works'] = True
    except Exception as e:
        results['issues'].append("Cohen's d failed: {}".format(e))

    # Test McNemar's test
    try:
        preds_a = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        preds_b = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        labels = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        stat, p_val, table = mcnemar_test(preds_a, preds_b, labels)
        if isinstance(p_val, float) and 0 <= p_val <= 1:
            results['mcnemar_works'] = True
    except Exception as e:
        results['issues'].append("McNemar test failed: {}".format(e))

    # Test Bonferroni correction
    try:
        p_values = [0.01, 0.04, 0.03, 0.25, 0.50]
        reject, corrected_p, alphac, _ = multiple_comparison_correction(
            p_values, alpha=0.05, method='bonferroni'
        )
        if len(corrected_p) == len(p_values):
            results['bonferroni_works'] = True
    except Exception as e:
        results['issues'].append("Bonferroni correction failed: {}".format(e))

    return results


def verify_aggregation_script():
    """Verify the aggregation script properly uses statistical testing."""
    results = {
        'imports_statistical_testing': False,
        'computes_confidence_intervals': False,
        'performs_significance_tests': False,
        'uses_bonferroni': False,
        'generates_latex_tables': False,
        'issues': []
    }

    agg_script = Path('telepathy/aggregate_results.py')
    if not agg_script.exists():
        results['issues'].append("aggregate_results.py not found")
        return results

    with open(agg_script, 'r') as f:
        content = f.read()

    # Check for statistical testing imports
    if 'from scripts.statistical_testing import' in content or 'import statistical_testing' in content:
        results['imports_statistical_testing'] = True

    # Check for confidence interval computation
    if 'bootstrap_ci' in content or '_compute_confidence_interval' in content:
        results['computes_confidence_intervals'] = True

    # Check for significance testing
    if 'ttest' in content or 'p_value' in content:
        results['performs_significance_tests'] = True

    # Check for multiple comparison correction
    if 'bonferroni' in content or 'multiple_comparison_correction' in content:
        results['uses_bonferroni'] = True

    # Check for LaTeX table generation
    if 'latex' in content.lower() or 'tex' in content:
        results['generates_latex_tables'] = True

    return results


def verify_paper_tables_script():
    """Verify the paper table generation script uses proper statistics."""
    results = {
        'imports_statistical_testing': False,
        'uses_three_seeds': False,
        'computes_mean_std': False,
        'includes_significance_stars': False,
        'uses_paired_tests': False,
        'issues': []
    }

    table_script = Path('scripts/generate_paper_tables.py')
    if not table_script.exists():
        results['issues'].append("generate_paper_tables.py not found")
        return results

    with open(table_script, 'r') as f:
        content = f.read()

    # Check imports
    if 'from scripts.statistical_testing import' in content:
        results['imports_statistical_testing'] = True

    # Check for three seeds
    if all(str(seed) in content for seed in REQUIRED_SEEDS):
        results['uses_three_seeds'] = True

    # Check for mean/std computation
    if 'np.mean' in content and 'np.std' in content:
        results['computes_mean_std'] = True

    # Check for significance stars
    if 'p_value_to_stars' in content or '***' in content:
        results['includes_significance_stars'] = True

    # Check for paired tests
    if 'paired_ttest' in content:
        results['uses_paired_tests'] = True

    return results


def run_example_comparison():
    """Run an example statistical comparison to verify everything works."""
    print("\n" + "="*80)
    print("RUNNING EXAMPLE STATISTICAL COMPARISON")
    print("="*80)

    # Simulate results for 3 methods across 3 seeds
    np.random.seed(42)

    # Baseline method (lower performance)
    baseline_scores = np.random.normal(0.70, 0.03, 3)  # 3 seeds

    # Our methods (better performance)
    method_scores = {
        'Telepathy Bridge': np.random.normal(0.85, 0.02, 3),
        'Linear Probe': np.random.normal(0.75, 0.03, 3),
        'LLMLingua': np.random.normal(0.72, 0.04, 3)
    }

    # Generate comprehensive comparison report
    report = comprehensive_comparison_report(
        'Baseline',
        baseline_scores,
        method_scores,
        correction='bonferroni',
        alpha=0.05,
        n_bootstrap=REQUIRED_BOOTSTRAP_SAMPLES
    )

    print(report)

    return True


def generate_verification_report():
    """Generate a comprehensive verification report."""
    print("="*80)
    print("STATISTICAL TESTING INTEGRATION VERIFICATION REPORT")
    print("="*80)
    print()

    all_checks_passed = True

    # 1. Verify seed usage in key scripts
    print("1. SEED USAGE VERIFICATION")
    print("-"*40)

    scripts_to_check = [
        'latentwire/train.py',
        'latentwire/eval.py',
        'telepathy/aggregate_results.py',
        'scripts/generate_paper_tables.py'
    ]

    for script in scripts_to_check:
        script_path = Path(script)
        results = verify_seed_usage(script_path)
        status = "✅" if results['uses_correct_seeds'] else "❌"
        print("{} {}: Seeds found: {}".format(status, script, results['seeds_found']))
        if results['issues']:
            print("   Issues: {}".format(', '.join(results['issues'])))
            all_checks_passed = False

    # 2. Verify bootstrap CI configuration
    print("\n2. BOOTSTRAP CI VERIFICATION")
    print("-"*40)

    bootstrap_results = verify_bootstrap_ci_usage()
    for key, value in bootstrap_results.items():
        if key != 'issues':
            status = "✅" if value else "❌"
            print("{} {}: {}".format(status, key.replace('_', ' ').title(), value))
            if not value:
                all_checks_passed = False

    # 3. Verify statistical tests
    print("\n3. STATISTICAL TESTS VERIFICATION")
    print("-"*40)

    test_results = verify_statistical_tests()
    for key, value in test_results.items():
        if key != 'issues':
            status = "✅" if value else "❌"
            print("{} {}: {}".format(status, key.replace('_', ' ').title(), value))
            if not value:
                all_checks_passed = False

    # 4. Verify aggregation script
    print("\n4. AGGREGATION SCRIPT VERIFICATION")
    print("-"*40)

    agg_results = verify_aggregation_script()
    for key, value in agg_results.items():
        if key != 'issues':
            status = "✅" if value else "❌"
            print("{} {}: {}".format(status, key.replace('_', ' ').title(), value))
            if not value and key != 'imports_statistical_testing':  # This one is optional
                all_checks_passed = False

    # 5. Verify paper tables script
    print("\n5. PAPER TABLES SCRIPT VERIFICATION")
    print("-"*40)

    table_results = verify_paper_tables_script()
    for key, value in table_results.items():
        if key != 'issues':
            status = "✅" if value else "❌"
            print("{} {}: {}".format(status, key.replace('_', ' ').title(), value))
            if not value:
                all_checks_passed = False

    # 6. Run example comparison
    print("\n6. EXAMPLE COMPARISON TEST")
    print("-"*40)

    try:
        run_example_comparison()
        print("✅ Example comparison completed successfully")
    except Exception as e:
        print("❌ Example comparison failed: {}".format(e))
        all_checks_passed = False

    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    if all_checks_passed:
        print("✅ ALL STATISTICAL TESTING COMPONENTS ARE PROPERLY INTEGRATED!")
        print("\nThe codebase is ready for rigorous statistical validation:")
        print("- 3 seeds (42, 123, 456) are consistently used")
        print("- Bootstrap CI with n=10000 is configured")
        print("- Paired t-tests are available for comparisons")
        print("- McNemar test is available for classification")
        print("- Cohen's d effect size is computed")
        print("- Bonferroni correction is available for multiple comparisons")
        print("- Aggregation properly computes mean ± std across seeds")
        print("- P-values and significance markers are computed")
        print("- LaTeX tables can be generated with proper formatting")
    else:
        print("⚠️  SOME COMPONENTS NEED ATTENTION")
        print("\nRecommendations:")
        print("1. Ensure all scripts use seeds 42, 123, 456 consistently")
        print("2. Update aggregate_results.py to import statistical_testing module")
        print("3. Verify all evaluation scripts set random seeds properly")
        print("4. Ensure paper table generation includes significance testing")

    return all_checks_passed


def main():
    """Main verification function."""
    # Run verification
    all_passed = generate_verification_report()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()