# HBSM Competitor Matrix

Status: local novelty guardrail before sensitivity measurements.

## Claim Boundary

HBSM cannot claim broad layer-sensitivity discovery. The surviving contribution
must be a cheaper predictor or mechanism for current hybrid reasoners, and it
must fold into HORN if directional-boundary asymmetry explains the same effect.

## Competitor Classes

| Class | Examples to cite/check | Reviewer risk | Required separation |
|---|---|---|---|
| Forward KL sensitivity | KL/sensitivity lens methods | "This has already been done." | Add frontier hybrid models and cheaper no-forward predictors. |
| Auto precision allocation | AutoQuantize/HAWQ-style tools | "Existing tools pick sensitive layers." | Compare against layer-index, norm, parameter-count, and boundary-only baselines. |
| Weight-stat predictors | Hessian, condition number, kurtosis methods | "Cheap predictors are old." | Show predictor transfer across models or train/test layer splits. |
| Activation outlier methods | SmoothQuant/AWQ/outlier-aware PTQ | "Sensitivity follows outlier magnitude." | Separate weight-only predictors from activation-derived predictors. |
| HORN overlap | Directional boundary asymmetry | "Mechanism is already HORN." | Merge if HORN accounts for sensitivity ranking. |

## Before A Paper Claim

- Define boundary-flagged layers and top-decile selection exactly.
- Include random top-decile flags, layer-index baseline, boundary-only baseline,
  parameter-count baseline, and norm baseline.
- Require Spearman rho with uncertainty and leave-one-model-out validation.
