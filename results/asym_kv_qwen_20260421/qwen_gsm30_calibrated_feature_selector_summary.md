# Calibrated Feature Selector Summary

- Calibration examples: `15`
- Eval examples: `15`
- Train accuracy: `0.2000`
- Train target-selection rate: `0.0000`
- Target-selection penalty: `0.0000`
- Weights: `{"answer_agreement": 0.5, "completion": 0.0, "format_score": 0.0, "is_seed": -1.0, "is_target": -2.0, "numeric_consistency": 0.0, "seed_index": 0.0}`

| Method | Accuracy | Delta vs target | Method-only | Baseline-only | Both correct | Both wrong | Target selected | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated_feature_selector | 0.0000 | -0.0667 | 0 | 1 | 0 | 14 | 0.0000 | 0.2667 |

Interpretation:

This selector calibrates transparent candidate-feature weights on a held-out calibration prefix, then
evaluates on the remaining examples. It is not an oracle: labels are used only to choose the weights
on the calibration split, and every candidate score and feature vector is logged for audit.
