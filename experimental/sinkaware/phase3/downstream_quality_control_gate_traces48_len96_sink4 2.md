# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 48
- max length: 96
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.072146 | +/- 0.067615 |
| rank-2 KL-to-exact improvement vs position-only | 0.069359 | +/- 0.082327 |
| minimum model loss-delta improvement | 0.037648 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.10609508 | 0.06844659 | +0.03764849 | +0.02735568 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.23056216 | 0.12391921 | +0.10664295 | +0.11136234 | yes |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 16 | 3.024935 | +0.00000000 | +0.10016294 | +0.06254286 | 0.07534835 | 0.05084758 | 0.1040 | 0.0911 |
| distilgpt2 | 1 | 16 | 3.025229 | +0.00000000 | +0.10873767 | +0.06894926 | 0.08294499 | 0.05398126 | 0.1303 | 0.1101 |
| distilgpt2 | 2 | 16 | 3.011390 | +0.00000000 | +0.10938462 | +0.07384765 | 0.09355452 | 0.06495197 | 0.1461 | 0.1143 |
| facebook/opt-125m | 0 | 16 | 2.701965 | +0.00000000 | +0.21097051 | +0.11567072 | 0.21842522 | 0.11629495 | 0.1877 | 0.1505 |
| facebook/opt-125m | 1 | 16 | 2.668158 | +0.00000000 | +0.22356601 | +0.12694360 | 0.20366237 | 0.10130860 | 0.1935 | 0.1401 |
| facebook/opt-125m | 2 | 16 | 2.613446 | +0.00000000 | +0.25714996 | +0.12914333 | 0.23288808 | 0.10328509 | 0.2209 | 0.1413 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
