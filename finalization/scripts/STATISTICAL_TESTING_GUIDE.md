# Statistical Testing Guide for ML Experiments

## Quick Reference

### Required Dependencies
```bash
pip install scipy statsmodels numpy
```

### Key Statistical Methods

| Method | Use Case | Minimum Samples | Python Function |
|--------|----------|-----------------|-----------------|
| Bootstrap CI | Estimate uncertainty of a metric | 20+ | `bootstrap_ci()` |
| Paired Bootstrap | Compare two methods on same test set | 20+ | `paired_bootstrap_test()` |
| McNemar's Test | Compare two classifiers (expensive models) | Any | `mcnemar_test()` |
| Bonferroni | Multiple comparisons (conservative) | Any | `multiple_comparison_correction()` |
| FDR (Benjamini-Hochberg) | Multiple comparisons (more power) | Any | `multiple_comparison_correction(method='fdr_bh')` |

---

## 1. Bootstrap Confidence Intervals (95% CI)

**When to use**: Report uncertainty for any metric (F1, EM, accuracy, etc.)

**Minimum samples**: 20+ test examples

**Method**: BCa (Bias-Corrected and Accelerated) is recommended over percentile method

### Example Usage

```python
from scripts.statistical_testing import bootstrap_ci
import numpy as np

# Your evaluation scores on N test examples
f1_scores = np.array([0.70, 0.72, 0.68, 0.75, 0.71, ...])  # N scores

# Compute mean and 95% CI
mean_f1, (lower, upper) = bootstrap_ci(
    f1_scores,
    statistic=np.mean,
    confidence_level=0.95,
    n_resamples=10000,
    method='BCa',
    random_state=42
)

print(f"F1: {mean_f1:.3f} [95% CI: {lower:.3f}, {upper:.3f}]")
# Output: F1: 0.712 [95% CI: 0.687, 0.735]
```

### Multiple Metrics at Once

```python
from scripts.statistical_testing import bootstrap_ci_multiple_metrics

results = {
    'f1': f1_scores,
    'em': em_scores,
    'nll': nll_scores
}

cis = bootstrap_ci_multiple_metrics(results, n_resamples=10000, random_state=42)

for metric, (val, (lo, hi)) in cis.items():
    print(f"{metric}: {val:.3f} [{lo:.3f}, {hi:.3f}]")
```

---

## 2. Paired Bootstrap Test (Comparing Methods)

**When to use**: Compare two methods evaluated on the **same test set**

**Minimum samples**: 20+ test examples (warns if fewer)

**Null hypothesis**: The two methods have the same expected performance

**Alternative hypotheses**:
- `'two-sided'`: Methods differ (default)
- `'greater'`: Method A > Method B
- `'less'`: Method A < Method B

### Example Usage

```python
from scripts.statistical_testing import paired_bootstrap_test

# Scores from two methods on the SAME test examples
latent_f1_scores = np.array([...])  # N scores
text_f1_scores = np.array([...])    # N scores (same test set)

diff, p_value, stats = paired_bootstrap_test(
    latent_f1_scores,
    text_f1_scores,
    n_resamples=10000,
    random_state=42,
    alternative='two-sided'
)

print(f"Difference: {diff:+.4f} ({(diff/stats['mean_b'])*100:+.2f}%)")
print(f"p-value: {p_value:.4f}")
print(f"Latent: {stats['mean_a']:.4f} ± {stats['std_a']:.4f}")
print(f"Text: {stats['mean_b']:.4f} ± {stats['std_b']:.4f}")

if p_value < 0.05:
    print("✓ Statistically significant at α=0.05")
else:
    print("✗ Not statistically significant")
```

### Interpretation

- **p < 0.05**: Strong evidence that methods differ (reject null hypothesis)
- **p ≥ 0.05**: Insufficient evidence to claim difference (fail to reject null)
- **p < 0.01**: Very strong evidence
- **p < 0.001**: Extremely strong evidence

---

## 3. McNemar's Test (Classification Comparison)

**When to use**: Compare two classifiers when you can only afford **one evaluation run**

**Best for**: Expensive models (e.g., large LLMs) where training multiple times is impractical

**Requirements**: Binary predictions (correct/incorrect) on same test set

**Recommended by**: Dietterich (1998) for single test set comparisons

### Example Usage

```python
from scripts.statistical_testing import mcnemar_test

# Binary predictions from two models on same test set
preds_model_a = np.array([...])  # Predictions from model A
preds_model_b = np.array([...])  # Predictions from model B
ground_truth = np.array([...])   # True labels

statistic, p_value, contingency_table = mcnemar_test(
    preds_model_a,
    preds_model_b,
    ground_truth,
    exact=None,  # Auto-select based on sample size
    correction=True
)

print(f"McNemar statistic: {statistic:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"\nContingency table:")
print(contingency_table)
#           Model B Correct | Model B Wrong
# Model A Correct:    85    |      10
# Model A Wrong:       8    |      12

if p_value < 0.05:
    print("✓ Significant difference between models")
```

### When to Use Exact vs Chi-Square

- **Exact binomial test**: Use when `b + c < 25` (small number of disagreements)
- **Chi-square approximation**: Use when `b + c ≥ 25` (large sample)
- **Set `exact=None`**: Automatically chooses based on sample size (recommended)

---

## 4. Multiple Comparison Corrections

**When to use**: When comparing **multiple methods** to a baseline (3+ comparisons)

**Problem**: With 20 comparisons at α=0.05, you expect 1 false positive by chance

**Solutions**:
- **Bonferroni**: Most conservative, controls family-wise error rate (FWER)
- **Holm-Bonferroni**: Less conservative than Bonferroni, still controls FWER
- **FDR (Benjamini-Hochberg)**: More powerful, controls false discovery rate

### When to Use Each Method

| Scenario | Recommended Method | Reasoning |
|----------|-------------------|-----------|
| 2-5 comparisons, false positives very costly | Bonferroni | Conservative, strict control |
| 5-20 comparisons, balanced approach | Holm | Less conservative than Bonferroni |
| 20+ comparisons, want to find true effects | FDR (fdr_bh) | More statistical power |
| Positively correlated tests | FDR (fdr_by) | More robust to correlation |

### Example: Compare Multiple Methods to Baseline

```python
from scripts.statistical_testing import compare_multiple_methods_to_baseline

# Baseline scores (text prompting)
baseline_scores = np.array([...])  # N test examples

# Multiple methods to compare
method_scores = {
    'Latent M=32': np.array([...]),   # N scores
    'Latent M=48': np.array([...]),   # N scores
    'Latent M=64': np.array([...]),   # N scores
    'Token-budget M=32': np.array([...])
}

# Compare with FDR correction
results = compare_multiple_methods_to_baseline(
    baseline_scores,
    method_scores,
    correction='fdr_bh',  # or 'bonferroni', 'holm'
    alpha=0.05,
    n_resamples=10000,
    random_state=42
)

# Print results
for method_name, res in results.items():
    print(f"\n{method_name}:")
    print(f"  Difference: {res['difference']:+.4f}")
    print(f"  Raw p-value: {res['p_value']:.4f}")
    print(f"  Corrected p-value: {res['corrected_p_value']:.4f}")
    if res['significant']:
        print(f"  ✓ SIGNIFICANT at α={alpha} (after correction)")
    else:
        print(f"  ✗ Not significant (after correction)")
```

### Manual Correction (if you have raw p-values)

```python
from scripts.statistical_testing import multiple_comparison_correction

# Raw p-values from multiple tests
p_values = [0.001, 0.04, 0.06, 0.15, 0.30]

# Apply Bonferroni correction
reject, p_corrected, alphac, _ = multiple_comparison_correction(
    p_values,
    alpha=0.05,
    method='bonferroni'
)

for i, (p_raw, p_corr, rej) in enumerate(zip(p_values, p_corrected, reject)):
    print(f"Test {i+1}: p={p_raw:.4f} → {p_corr:.4f} (reject={rej})")

# Output:
# Test 1: p=0.0010 → 0.0050 (reject=True)
# Test 2: p=0.0400 → 0.2000 (reject=False)
# Test 3: p=0.0600 → 0.3000 (reject=False)
# Test 4: p=0.1500 → 0.7500 (reject=False)
# Test 5: p=0.3000 → 1.0000 (reject=False)
```

---

## 5. Sample Size Requirements (Power Analysis)

**Question**: How many test examples (or random seeds) do I need?

**Answer**: Depends on the **effect size** you want to detect

### Rule of Thumb

| Effect Size | Description | Approx. Samples Needed* |
|-------------|-------------|------------------------|
| 1-2% | Very small, hard to detect | 500+ |
| 5% | Small but meaningful | 100-200 |
| 10% | Medium effect | 30-50 |
| 20%+ | Large effect | 10-20 |

*For 80% statistical power at α=0.05, assuming std dev ~0.10

### Calculate Exact Sample Size

```python
from scripts.statistical_testing import estimate_required_samples

# Want to detect 5% F1 improvement with 80% power
effect_size = 0.05  # 5% improvement
std_dev = 0.10      # Expected standard deviation

n_required = estimate_required_samples(
    effect_size=effect_size,
    alpha=0.05,
    power=0.80,
    std_dev=std_dev
)

print(f"Need {n_required} test examples to detect {effect_size:.3f} effect")
# Output: Need 126 test examples to detect 0.050 effect
```

### Estimate from Pilot Data

```python
from scripts.statistical_testing import estimate_required_seeds_from_data

# Ran 5 experiments with different random seeds
pilot_scores = np.array([0.70, 0.72, 0.68, 0.75, 0.71])

# Want to detect 3% improvement
n_seeds = estimate_required_seeds_from_data(
    pilot_scores,
    effect_size=0.03,
    power=0.80
)

print(f"Recommend {n_seeds} random seeds for robust comparison")
```

### How Many Random Seeds? (Key Research)

**Reference**: Colas et al. (2018) - "How Many Random Seeds? Statistical Power Analysis in Deep RL"

**Key findings**:
- Most papers use only 5 seeds or fewer (insufficient!)
- Bootstrap tests require **at least N=20 samples** for reliability
- For detecting small effects (~3-5%), need **30-50+ seeds**
- For large effects (10%+), 10-20 seeds may suffice

**Recommendation**:
- **Quick experiments**: 5-10 seeds (exploratory only)
- **Publication-quality results**: 20-50 seeds
- **Small expected effects**: 50+ seeds

---

## 6. Comprehensive Comparison Report

**Use case**: Generate a complete statistical report comparing multiple methods

```python
from scripts.statistical_testing import comprehensive_comparison_report

report = comprehensive_comparison_report(
    baseline_name='Text Baseline',
    baseline_scores=baseline_scores,
    method_scores={
        'Latent M=32': latent_32_scores,
        'Latent M=48': latent_48_scores,
        'Token-budget': token_budget_scores
    },
    correction='fdr_bh',
    alpha=0.05,
    n_bootstrap=10000,
    random_state=42
)

print(report)
# Saves to file
with open('results/statistical_report.txt', 'w') as f:
    f.write(report)
```

### Example Output

```
================================================================================
STATISTICAL COMPARISON REPORT
================================================================================

Sample size: 200
Significance level (α): 0.05
Multiple testing correction: fdr_bh

--------------------------------------------------------------------------------
BASELINE: Text Baseline
--------------------------------------------------------------------------------
Mean: 0.7523
95% CI: [0.7312, 0.7721]
Std Dev: 0.0987

--------------------------------------------------------------------------------
METHOD COMPARISONS (vs Baseline)
--------------------------------------------------------------------------------

Method: Latent M=32
  Mean: 0.1245
  95% CI: [0.1078, 0.1421]
  Difference from baseline: -0.6278
  Improvement: -83.45%
  Raw p-value: 0.0000
  Corrected p-value: 0.0000
  ✓ STATISTICALLY SIGNIFICANT at α=0.05

Method: Token-budget
  Mean: 0.4123
  95% CI: [0.3891, 0.4367]
  Difference from baseline: -0.3400
  Improvement: -45.19%
  Raw p-value: 0.0000
  Corrected p-value: 0.0000
  ✓ STATISTICALLY SIGNIFICANT at α=0.05

--------------------------------------------------------------------------------
POWER ANALYSIS
--------------------------------------------------------------------------------
Pooled standard deviation: 0.1102

Required sample sizes to detect effects (80% power, α=0.05):
  1% improvement (0.0075): 850 samples
  2% improvement (0.0150): 213 samples
  5% improvement (0.0376): 35 samples
  10% improvement (0.0752): 9 samples

================================================================================
```

---

## 7. Practical Workflow for LatentWire Experiments

### Step 1: Run Evaluation

```bash
# Run evaluation with sufficient test samples
python latentwire/eval.py \
  --ckpt runs/exp_name/epoch24 \
  --samples 200 \
  --dataset squad \
  --fresh_eval
```

### Step 2: Extract Scores

```python
import json
import numpy as np

# Load evaluation results
with open('runs/exp_name/epoch24/eval_results.json', 'r') as f:
    results = json.load(f)

# Extract per-example scores
text_f1 = np.array([ex['text_f1'] for ex in results['examples']])
latent_f1 = np.array([ex['latent_f1'] for ex in results['examples']])
budget_f1 = np.array([ex['budget_f1'] for ex in results['examples']])
```

### Step 3: Statistical Analysis

```python
from scripts.statistical_testing import (
    bootstrap_ci,
    paired_bootstrap_test,
    compare_multiple_methods_to_baseline,
    comprehensive_comparison_report
)

# 1. Compute CIs for each method
text_mean, text_ci = bootstrap_ci(text_f1)
latent_mean, latent_ci = bootstrap_ci(latent_f1)

print(f"Text: {text_mean:.3f} [{text_ci[0]:.3f}, {text_ci[1]:.3f}]")
print(f"Latent: {latent_mean:.3f} [{latent_ci[0]:.3f}, {latent_ci[1]:.3f}]")

# 2. Paired comparison
diff, p_val, stats = paired_bootstrap_test(latent_f1, text_f1)
print(f"\nLatent vs Text: {diff:+.4f} (p={p_val:.4f})")

# 3. Multiple comparisons with correction
methods = {
    'Latent': latent_f1,
    'Token-budget': budget_f1
}

results = compare_multiple_methods_to_baseline(
    text_f1,
    methods,
    correction='fdr_bh'
)

# 4. Generate comprehensive report
report = comprehensive_comparison_report(
    'Text Baseline',
    text_f1,
    methods
)

# Save report
with open('runs/exp_name/statistical_report.txt', 'w') as f:
    f.write(report)
```

### Step 4: Report in Paper/Log

```markdown
## Results

We evaluated on 200 SQuAD examples. Statistical significance assessed via
paired bootstrap test (10,000 resamples) with Benjamini-Hochberg FDR correction
for multiple comparisons.

| Method | F1 Score | 95% CI | vs Text | p-value |
|--------|----------|--------|---------|---------|
| Text baseline | 0.752 | [0.731, 0.772] | — | — |
| Latent (M=32) | 0.125 | [0.108, 0.142] | -0.628 | <0.001*** |
| Token-budget | 0.412 | [0.389, 0.437] | -0.340 | <0.001*** |

*p < 0.05, **p < 0.01, ***p < 0.001 (after FDR correction)
```

---

## 8. Common Pitfalls & Best Practices

### Pitfall 1: Too Few Samples
❌ **Bad**: 10 test examples, bootstrap CI
✅ **Good**: 50+ test examples, or acknowledge limitation

### Pitfall 2: Multiple Testing Without Correction
❌ **Bad**: Compare 5 methods, report raw p-values
✅ **Good**: Use Bonferroni or FDR correction

### Pitfall 3: Cherry-Picking Seeds
❌ **Bad**: Run 10 seeds, report best one
✅ **Good**: Pre-register N seeds, report all

### Pitfall 4: P-Hacking
❌ **Bad**: Try many metrics until one is significant
✅ **Good**: Pre-specify primary metric, adjust for multiple metrics

### Pitfall 5: Misinterpreting Non-Significance
❌ **Bad**: "p > 0.05 means the methods are the same"
✅ **Good**: "p > 0.05 means insufficient evidence to claim difference"

### Pitfall 6: Ignoring Effect Size
❌ **Bad**: "p < 0.05, so it's important"
✅ **Good**: "p < 0.05 AND 10% improvement → meaningful"

### Best Practices Summary

1. **Pre-register experiments**: Decide metrics, comparisons, and sample sizes before running
2. **Use adequate samples**: 50+ test examples, 20+ random seeds
3. **Correct for multiple comparisons**: Always use Bonferroni/FDR when comparing 3+ methods
4. **Report CIs, not just means**: Bootstrap CIs show uncertainty
5. **Report effect sizes**: Absolute and relative differences matter
6. **Use appropriate tests**: Paired bootstrap for same test set, McNemar for expensive models
7. **Document everything**: Random seeds, sample sizes, correction methods

---

## 9. References & Further Reading

### Key Papers

1. **Dietterich (1998)**: "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
   - Recommends McNemar's test for single evaluations
   - Discusses 5×2 cross-validation paired t-test

2. **Colas et al. (2018)**: "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments"
   - Provides formulas for required number of seeds
   - GitHub: https://github.com/flowersteam/rl-difference-testing

3. **Efron & Tibshirani (1993)**: "An Introduction to the Bootstrap"
   - Classic reference on bootstrap methods

4. **Benjamini & Hochberg (1995)**: "Controlling the False Discovery Rate"
   - Original FDR paper

### Online Resources

- [Machine Learning Mastery - Bootstrap CIs](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)
- [SciPy Bootstrap Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Statsmodels Multiple Testing](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)

### When to Consult a Statistician

- Designing complex experimental protocols
- Dealing with non-standard data (e.g., hierarchical, time-series)
- Publication in top-tier venues requiring rigorous statistics
- Results seem contradictory or counterintuitive

---

## 10. Quick Checklist for Paper Submission

Before submitting results, verify:

- [ ] **Sample size**: At least 50 test examples OR 20+ random seeds
- [ ] **Statistical test**: Appropriate test chosen and documented
- [ ] **Multiple comparisons**: Corrected if comparing 3+ methods
- [ ] **Confidence intervals**: Reported for all key metrics
- [ ] **P-values**: Reported with correction method
- [ ] **Effect sizes**: Both absolute and relative differences reported
- [ ] **Random seeds**: All seeds reported or pre-registered
- [ ] **Reproducibility**: Code, data, and random seeds available
- [ ] **Assumptions**: Test assumptions verified (e.g., sample size)
- [ ] **Interpretation**: Claims match statistical evidence

---

## Appendix: Common Statistical Tests for ML

| Test | Use Case | Assumptions | Python |
|------|----------|-------------|--------|
| **Paired t-test** | Compare means, paired data | Normality, independence | `scipy.stats.ttest_rel()` |
| **Wilcoxon signed-rank** | Non-parametric paired comparison | Symmetric differences | `scipy.stats.wilcoxon()` |
| **Bootstrap** | Any statistic, no assumptions | Sufficient samples (20+) | `scipy.stats.bootstrap()` |
| **McNemar's test** | Binary classification comparison | Paired nominal data | `statsmodels.stats.contingency_tables.mcnemar()` |
| **Permutation test** | Any statistic, no assumptions | Exchangeability | Custom implementation |
| **Mann-Whitney U** | Compare distributions, unpaired | Independent samples | `scipy.stats.mannwhitneyu()` |

**Recommendation for ML**: Start with **bootstrap methods** (most flexible, fewest assumptions)
