# Statistical Testing Functions Available

The `statistical_testing.py` file in the finalization directory contains all critical statistical functions needed for Phase 1 experiments:

## Core Statistical Tests
- `bootstrap_ci()` - Bootstrap confidence intervals with BCa method
- `paired_ttest()` - Paired t-test for within-subject comparisons 
- `independent_ttest()` - Independent t-test for between-group comparisons
- `paired_bootstrap_test()` - Paired bootstrap test for comparing methods
- `mcnemar_test()` - McNemar's test for classification comparison

## Effect Sizes
- `cohens_d_pooled()` - Cohen's d for independent samples
- `cohens_d_paired()` - Cohen's d for paired samples

## Multiple Comparison Corrections
- `multiple_comparison_correction()` - Bonferroni, Holm, FDR corrections
- `compare_multiple_methods_to_baseline()` - Compare multiple methods with corrections

## Power Analysis
- `estimate_required_samples()` - Estimate sample size needed for effect detection
- `estimate_required_seeds_from_data()` - Estimate seeds needed based on pilot data

## Reporting Functions
- `comprehensive_comparison_report()` - Generate detailed statistical report
- `generate_comparison_table()` - Create formatted comparison tables
- `generate_detailed_comparison_table()` - Detailed single-metric comparison

## Utility Functions
- `p_value_to_stars()` - Convert p-values to significance stars
- `format_mean_ci()` - Format mean with confidence intervals
- `aggregate_multiseed_results()` - Aggregate results across seeds with warnings

All functions use proper statistical methods with ddof=1 for unbiased sample standard deviation and include warnings for low sample sizes.
