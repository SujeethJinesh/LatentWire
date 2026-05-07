# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 24
- max length: 64
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.080929 | +/- 0.081535 |
| rank-2 KL-to-exact improvement vs position-only | 0.082463 | +/- 0.107799 |
| minimum model loss-delta improvement | 0.039330 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.09469089 | 0.05536110 | +0.03932979 | +0.02746348 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.22125434 | 0.09872561 | +0.12252873 | +0.13746250 | yes |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 8 | 2.941694 | +0.00000000 | +0.10019264 | +0.06391380 | 0.09645936 | 0.06511443 | 0.1426 | 0.1205 |
| distilgpt2 | 1 | 8 | 2.677648 | +0.00000000 | +0.10993086 | +0.05754080 | 0.09308016 | 0.06467003 | 0.1290 | 0.1171 |
| distilgpt2 | 2 | 8 | 2.878404 | +0.00000000 | +0.07394916 | +0.04462870 | 0.06253521 | 0.03989983 | 0.1270 | 0.0933 |
| facebook/opt-125m | 0 | 8 | 2.659699 | +0.00000000 | +0.24015738 | +0.11132993 | 0.23380664 | 0.07941131 | 0.2485 | 0.1323 |
| facebook/opt-125m | 1 | 8 | 2.431580 | +0.00000000 | +0.23877410 | +0.09490052 | 0.18591960 | 0.05074528 | 0.2063 | 0.0992 |
| facebook/opt-125m | 2 | 8 | 2.588516 | +0.00000000 | +0.18483153 | +0.08994637 | 0.19890828 | 0.07609042 | 0.1806 | 0.1012 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
