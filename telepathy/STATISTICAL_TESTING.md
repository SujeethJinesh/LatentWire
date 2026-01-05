# Statistical Testing Module for Telepathy

## Overview

The `statistical_tests.py` module provides comprehensive statistical methods for rigorously comparing the Bridge architecture with baselines. It integrates seamlessly with `run_unified_comparison.py` results.

## Key Features

### 1. Bootstrap Confidence Intervals (n=10000)
- **Method**: Bias-corrected accelerated (BCa) bootstrap
- **Purpose**: Robust confidence intervals for accuracy estimates
- **Usage**: Essential when working with small numbers of seeds (3-5)

### 2. McNemar's Test for Paired Comparisons
- **Purpose**: Compare two classifiers on the same test set
- **Advantage**: More powerful than independent tests
- **Requirement**: Per-example predictions from both methods

### 3. Bonferroni Correction for Multiple Testing
- **Purpose**: Control family-wise error rate when comparing to multiple baselines
- **Implementation**: Adjusts p-values to maintain α=0.05 across all tests
- **Alternative**: FDR methods available for less conservative correction

### 4. Effect Size Calculations (Cohen's d)
- **Purpose**: Quantify magnitude of improvement (not just significance)
- **Interpretation**:
  - |d| < 0.2: negligible
  - |d| < 0.5: small
  - |d| < 0.8: medium
  - |d| ≥ 0.8: large

### 5. Power Analysis for Sample Size Determination
- **Purpose**: Determine how many seeds/examples needed
- **Default**: 80% power at α=0.05
- **Output**: Required samples for detecting various effect sizes

## Integration with run_unified_comparison.py

### Quick Start

```python
from telepathy.statistical_tests import (
    analyze_unified_results,
    generate_statistical_report
)

# After running unified comparison
results = analyze_unified_results(
    'runs/unified/unified_results_20250104.json',
    baseline_method='mistral_zeroshot',
    target_method='bridge'
)

# Generate comprehensive report
report = generate_statistical_report(
    'runs/unified/unified_results_20250104.json',
    output_path='statistical_report.txt'
)
```

## API Reference

### Core Functions

#### `bootstrap_ci(scores, n_bootstrap=10000, confidence_level=0.95)`
Compute bootstrap confidence interval for accuracy scores.

**Returns**: `(mean, (ci_lower, ci_upper))`

#### `mcnemar_test(pred1, pred2, labels)`
McNemar's test for paired classification comparison.

**Returns**: `(statistic, p_value, contingency_table)`

#### `bonferroni_correction(p_values, alpha=0.05)`
Apply Bonferroni correction for multiple comparisons.

**Returns**: `(reject, corrected_p_values, corrected_alpha)`

#### `calculate_effect_size(scores1, scores2, paired=False)`
Calculate Cohen's d effect size between two methods.

**Returns**: `float` (Cohen's d)

#### `determine_sample_size(effect_size, power=0.8, alpha=0.05)`
Determine required sample size for desired statistical power.

**Returns**: `int` (number of samples needed)

### Integration Functions

#### `analyze_unified_results(results_path, baseline_method, target_method)`
Analyze results from run_unified_comparison.py with statistical tests.

**Returns**: Dictionary with statistical analysis for each dataset

#### `generate_comparison_table(results_path, methods_to_compare, datasets)`
Generate formatted comparison table from unified results.

**Returns**: Markdown table string

#### `generate_statistical_report(results_path, output_path, alpha=0.05)`
Generate comprehensive statistical analysis report.

**Returns**: Formatted report string

## Example Usage

### 1. Analyzing Multi-Seed Results

```python
from telepathy.statistical_tests import bootstrap_ci, calculate_effect_size

# Accuracy from 3 seeds
bridge_scores = [75.2, 77.3, 74.8]
baseline_scores = [70.1, 71.5, 69.8]

# Bootstrap CI
mean, (ci_low, ci_high) = bootstrap_ci(bridge_scores)
print(f"Bridge: {mean:.1f}% [{ci_low:.1f}%, {ci_high:.1f}%]")

# Effect size
d = calculate_effect_size(bridge_scores, baseline_scores, paired=True)
print(f"Cohen's d = {d:.2f}")
```

### 2. Multiple Comparison Correction

```python
from telepathy.statistical_tests import bonferroni_correction

# P-values from comparing Bridge to 5 baselines
p_values = [0.001, 0.03, 0.04, 0.08, 0.15]

# Apply correction
reject, p_adj, alpha_adj = bonferroni_correction(p_values)

for i, (p_raw, p_corr, rej) in enumerate(zip(p_values, p_adj, reject)):
    status = "SIGNIFICANT" if rej else "not significant"
    print(f"Baseline {i+1}: p={p_raw:.3f} → p_adj={p_corr:.3f} ({status})")
```

### 3. Power Analysis

```python
from telepathy.statistical_tests import determine_sample_size

# How many seeds to detect medium effect?
n_seeds = determine_sample_size(effect_size=0.5, power=0.8)
print(f"Need {n_seeds} seeds to detect d=0.5 with 80% power")

# Results:
# d=0.2 (small): 64 seeds
# d=0.5 (medium): 11 seeds
# d=0.8 (large): 5 seeds
```

### 4. Full Analysis Pipeline

```python
from telepathy.statistical_tests import (
    analyze_unified_results,
    generate_statistical_report
)

# Analyze experiment results
analysis = analyze_unified_results(
    'runs/unified/unified_results.json',
    baseline_method='mistral_zeroshot',
    target_method='bridge'
)

# Print results for each dataset
for dataset, stats in analysis.items():
    print(f"{dataset.upper()}:")
    print(f"  Improvement: {stats['relative_improvement']:.1f}%")
    print(f"  Effect size: {stats['effect_size']:.2f}")
    print(f"  p-value: {stats['p_value']:.4f}")
    if stats['significant']:
        print("  ✓ Statistically significant")

# Generate report
report = generate_statistical_report(
    'runs/unified/unified_results.json',
    output_path='statistical_analysis.txt'
)
```

## Recommendations for Experiments

### Minimum Seeds for Reliable Results

| Effect Size | Description | Seeds Needed | Detectable Improvement |
|------------|-------------|--------------|------------------------|
| d = 0.2 | Small | 64 | ~2% accuracy |
| d = 0.5 | Medium | 11 | ~5% accuracy |
| d = 0.8 | Large | 5 | ~8% accuracy |
| d = 1.0 | Very Large | 4 | ~10% accuracy |

### Best Practices

1. **Use at least 5 seeds** for detecting medium effects
2. **Apply Bonferroni correction** when comparing to multiple baselines
3. **Report both p-values and effect sizes** for complete picture
4. **Save per-example predictions** to enable McNemar's test
5. **Include bootstrap CIs** in all reported results

### Modifications Needed for McNemar's Test

To enable McNemar's test, modify evaluation functions to save predictions:

```python
# In eval_bridge() and other eval functions:
predictions = []
for item in eval_ds:
    # ... evaluation code ...
    predictions.append(pred)

# Save predictions
np.save(f"{output_dir}/{dataset}_{method}_predictions.npy", predictions)
```

## Statistical Significance Conventions

- `*` : p < 0.05 (significant)
- `**` : p < 0.01 (highly significant)
- `***` : p < 0.001 (very highly significant)
- No star: p ≥ 0.05 (not significant)

## Dependencies

```python
# Required packages (install with pip)
scipy>=1.7.0
numpy>=1.20.0
```

## Files Created

1. **telepathy/statistical_tests.py** - Main statistical testing module
2. **telepathy/test_statistical_tests.py** - Comprehensive test suite
3. **telepathy/example_statistical_usage.py** - Usage examples
4. **telepathy/STATISTICAL_TESTING.md** - This documentation

## Citation

When using these statistical methods, consider citing:

- Dietterich (1998): "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
- Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
- Colas et al. (2018): "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments"