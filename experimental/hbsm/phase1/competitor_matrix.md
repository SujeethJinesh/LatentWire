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

## Executable Baseline Columns

Every real B1/B2 result table must include:

| Column | Meaning |
|---|---|
| `forward_sensitivity` | KL or NLL/PPL drift from the frozen perturbation run. |
| `boundary_flag` | Exact boundary-derived layer flag, not an informal label. |
| `random_top_decile` | Random layer flags with the same count as the proposed flags. |
| `layer_index` | Layer-depth baseline. |
| `parameter_count` | Size baseline for sensitivity. |
| `weight_norm` | Weight-norm or max-abs baseline. |
| `cheap_predictor` | No-forward statistic being tested against sensitivity rank. |
| `train_test_split` | Held-out layer/model split if enough layers exist. |

## Source Anchors Checked

- KL Lens: `https://arxiv.org/abs/2604.13440`
- SmoothQuant: `https://arxiv.org/abs/2211.10438`
- QuaRot: `https://arxiv.org/abs/2404.00456`
- vLLM hybrid SSM disaggregated serving: `https://vllm.ai/blog/hybrid-ssm-disagg`
