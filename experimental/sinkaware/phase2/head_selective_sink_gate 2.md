# SinkAware Head-Selective Rank-2 Gate

Status: **WEAKENED; validation-selected rank-2 heads do not improve held-out output drift.**

- model: `distilgpt2`
- traces: 48
- sink tokens: 4
- selected rank-2 heads: 19 / 72 (0.264)

The rule fits predictors on the train split, selects rank-2 heads on a validation split when rank-2 beats position-only on output rel-L2, then evaluates the mixed policy on a held-out split.

| Test policy | Sink-logit RMSE | Sink-mass MAE | Attention L1 | Output rel-L2 |
|---|---:|---:|---:|---:|
| position | 1200.1446 | 0.0765 | 0.1777 | 0.1724 |
| rank2_all_heads | 998.6717 | 0.0546 | 0.1351 | 0.1419 |
| rank2_validation_selected | 935.0707 | 0.0565 | 0.1384 | 0.2035 |

## Decision

- held-out output rel-L2 improvement vs position-only: `-0.0312`
- held-out output rel-L2 margin vs all-rank2: `-0.0617` (positive means selected is better)

A GPU paper should not claim speed from this result. It only decides whether a head-selective approximation is worth native implementation.
