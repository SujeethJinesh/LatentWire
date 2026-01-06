# Statistical Testing Documentation

## Overview

The LatentWire evaluation framework now includes comprehensive statistical testing to ensure rigorous comparison between models. This addresses reviewer concerns about the statistical significance of reported results.

## Features

### 1. Bootstrap Confidence Intervals (BCa Method)

- **Method**: Bias-corrected and accelerated (BCa) bootstrap
- **Resamples**: 10,000 by default
- **Confidence Level**: 95%
- **Implementation**: `latentwire/statistical_eval.py::enhanced_bootstrap_ci()`

The BCa method provides more accurate confidence intervals than percentile bootstrap, especially for skewed distributions and small samples.

### 2. McNemar's Test

- **Purpose**: Compare two models on binary classification outcomes (e.g., exact match)
- **Null Hypothesis**: Both models have the same error rate
- **Implementation**: `latentwire/statistical_eval.py::mcnemar_test_binary()`

McNemar's test is specifically designed for paired binary data and tests whether two models make different types of errors.

### 3. Paired Bootstrap Test

- **Purpose**: Non-parametric test for comparing paired continuous scores
- **Resamples**: 10,000
- **Alternative Hypotheses**: two-sided, greater, less
- **Implementation**: `latentwire/statistical_eval.py::paired_bootstrap_test()`

This test is robust to non-normal distributions and provides p-values for significance testing.

### 4. Effect Sizes (Cohen's d)

- **Purpose**: Measure the magnitude of differences
- **Interpretation**:
  - |d| < 0.2: negligible effect
  - |d| < 0.5: small effect
  - |d| < 0.8: medium effect
  - |d| ≥ 0.8: large effect
- **Implementation**: `latentwire/statistical_eval.py::cohens_d()`

Effect sizes complement p-values by quantifying the practical significance of differences.

## Integration with eval.py

The statistical testing is automatically performed during evaluation when the module is available:

```python
# In eval.py, after computing per-example scores:
if STATISTICAL_EVAL_AVAILABLE:
    statistical_results = {}

    for name in model_contexts:
        # Enhanced bootstrap CI with BCa
        model_stats['text'] = {
            'em': enhanced_bootstrap_ci(text_em_scores, method='BCa'),
            'f1': enhanced_bootstrap_ci(text_f1_scores, method='BCa')
        }

        # Paired bootstrap test
        model_stats['latent_vs_text'] = {
            'em': paired_bootstrap_test(latent_em_scores, text_em_scores),
            'f1': paired_bootstrap_test(latent_f1_scores, text_f1_scores)
        }

        # McNemar test for binary outcomes
        model_stats['mcnemar'] = {
            'latent_vs_text_em': mcnemar_test_binary(
                latent_em_scores > 0.5,
                text_em_scores > 0.5
            )
        }

        # Effect sizes
        model_stats['effect_sizes'] = {
            'latent_vs_text': {
                'em': cohens_d(latent_em_scores, text_em_scores, paired=True),
                'f1': cohens_d(latent_f1_scores, text_f1_scores, paired=True)
            }
        }
```

## Output Format

The evaluation results now include:

```json
{
  "statistical_analysis": {
    "llama": {
      "text": {
        "em": {
          "mean": 0.85,
          "std": 0.12,
          "ci_lower": 0.82,
          "ci_upper": 0.88,
          "n_samples": 100,
          "ci_method": "BCa"
        }
      },
      "latent_vs_text": {
        "em": {
          "observed_difference": -0.25,
          "p_value": 0.001,
          "ci_difference": [-0.30, -0.20],
          "effect_size": -1.5
        }
      },
      "mcnemar": {
        "latent_vs_text_em": {
          "statistic": 45.3,
          "p_value": 0.001,
          "contingency_table": [[40, 30], [5, 25]]
        }
      }
    }
  },
  "significance_summary": {
    "llama": {
      "latent_vs_text": {
        "em_significant": true,
        "em_p_value": 0.001,
        "f1_significant": true,
        "f1_p_value": 0.003
      }
    }
  },
  "statistical_methods": {
    "bootstrap": {
      "n_resamples": 10000,
      "confidence_level": 0.95,
      "method": "BCa (bias-corrected and accelerated)"
    },
    "tests": {
      "paired_bootstrap": "Non-parametric test for paired samples",
      "mcnemar": "Test for binary classification outcomes",
      "cohens_d": "Standardized effect size measure"
    },
    "interpretation": {
      "p_value": "p < 0.05 indicates statistical significance",
      "cohens_d": "|d| < 0.2: negligible, |d| < 0.5: small, |d| < 0.8: medium, |d| >= 0.8: large",
      "ci": "95% confidence intervals show uncertainty in estimates",
      "mcnemar": "Tests if models make different types of errors"
    }
  }
}
```

## Usage

### Running Evaluation with Statistical Testing

```bash
python latentwire/eval.py \
    --ckpt runs/checkpoint \
    --samples 200 \
    --dataset squad \
    --out_dir results/
```

The statistical analysis is automatically included if the module is available.

### Testing the Module

```bash
python test_statistical_eval.py
```

This runs comprehensive tests with synthetic data to validate the implementation.

## Interpretation Guide

### P-values
- **p < 0.001**: Very strong evidence against null hypothesis (⋆⋆⋆)
- **p < 0.01**: Strong evidence against null hypothesis (⋆⋆)
- **p < 0.05**: Moderate evidence against null hypothesis (⋆)
- **p ≥ 0.05**: Insufficient evidence to reject null hypothesis

### Confidence Intervals
- If CI does not include 0, the difference is statistically significant
- Narrower CIs indicate more precise estimates
- BCa method accounts for bias and skewness in the bootstrap distribution

### Effect Sizes
- Complement p-values by showing practical significance
- Large sample sizes can make tiny differences statistically significant
- Effect sizes help determine if differences are meaningful

### McNemar's Test
- Specifically designed for paired binary data
- More appropriate than chi-square for dependent samples
- Focuses on discordant pairs (where models disagree)

## Multiple Comparison Considerations

When comparing multiple models or metrics, consider adjusting for multiple comparisons:

1. **Bonferroni Correction**: Divide α by number of tests
2. **False Discovery Rate (FDR)**: Control expected proportion of false positives
3. **Report all tests**: Full transparency about number of comparisons made

## Best Practices

1. **Always report**:
   - Sample sizes
   - Confidence intervals
   - Effect sizes alongside p-values
   - Method used for statistical testing

2. **For small samples** (n < 30):
   - Use bootstrap methods over parametric tests
   - Report limitations
   - Consider collecting more data

3. **For paired data**:
   - Use paired tests (more powerful)
   - Ensure same examples in same order
   - Check for systematic biases

## References

1. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*
2. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions
3. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
4. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms

## Implementation Files

- `latentwire/statistical_eval.py`: Core statistical testing module
- `latentwire/eval.py`: Integration with evaluation pipeline
- `test_statistical_eval.py`: Test suite with examples
- `scripts/statistical_testing.py`: Additional utilities and power analysis