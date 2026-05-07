# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 24
- max length: 96
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2
- rank modes: `rank1`, `rank2`, `rank4`, `rank8`

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-k sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.072814 | +/- 0.061944 |
| rank-2 KL-to-exact improvement vs position-only | 0.076915 | +/- 0.090383 |
| minimum model loss-delta improvement | 0.041210 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.09727513 | 0.05606509 | +0.04121004 | +0.03080065 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.21128254 | 0.10686476 | +0.10441779 | +0.12302870 | yes |

## Rank Frontier Across Models

| Rank mode | Abs loss delta | Loss improvement vs position | KL improvement vs position | Top1 diff | Min model improvement |
|---|---:|---:|---:|---:|---:|
| rank1 | 0.116968 | +0.037311 | +0.037258 | 0.1447 | +0.027908 |
| rank2 | 0.081465 | +0.072814 | +0.076915 | 0.1106 | +0.041210 |
| rank4 | 0.044217 | +0.110061 | +0.103544 | 0.0851 | +0.057496 |
| rank8 | 0.029128 | +0.125151 | +0.119542 | 0.0690 | +0.071785 |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 8 | 2.824739 | +0.00000000 | +0.09815226 | +0.05943794 | 0.09576370 | 0.05980764 | 0.1325 | 0.1034 |
| distilgpt2 | 1 | 8 | 2.661540 | +0.00000000 | +0.11688188 | +0.06130437 | 0.09679198 | 0.06361491 | 0.1322 | 0.1141 |
| distilgpt2 | 2 | 8 | 2.764125 | +0.00000000 | +0.07679124 | +0.04745295 | 0.06318648 | 0.03991767 | 0.1214 | 0.0815 |
| facebook/opt-125m | 0 | 8 | 2.510296 | +0.00000000 | +0.22840234 | +0.11738335 | 0.22004922 | 0.08412426 | 0.2182 | 0.1342 |
| facebook/opt-125m | 1 | 8 | 2.355124 | +0.00000000 | +0.23354176 | +0.11062453 | 0.18112411 | 0.05934847 | 0.2018 | 0.1214 |
| facebook/opt-125m | 2 | 8 | 2.466056 | +0.00000000 | +0.17190353 | +0.09258638 | 0.19090514 | 0.07951965 | 0.1732 | 0.1089 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
