# SinkAware Downstream Rank Frontier Gate

Status: **ALIVE but bounded; rank frontier repeats on 48 traces.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 48
- max length: 96
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2
- rank modes: `rank1`, `rank2`, `rank4`, `rank8`

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-k sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

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

## Rank Frontier Across Models

| Rank mode | Abs loss delta | Loss improvement vs position | KL improvement vs position | Top1 diff | Min model improvement |
|---|---:|---:|---:|---:|---:|
| rank1 | 0.137439 | +0.030889 | +0.029484 | 0.1431 | +0.022881 |
| rank2 | 0.096183 | +0.072146 | +0.069359 | 0.1246 | +0.037648 |
| rank4 | 0.061600 | +0.106729 | +0.097084 | 0.0952 | +0.055691 |
| rank8 | 0.043640 | +0.124689 | +0.114292 | 0.0801 | +0.070658 |

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

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-k rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact.

The frontier is monotonic on this 48-trace slice: higher ranks reduce loss drift, KL drift, and top-1 disagreement. This strengthens the quality/cost story but does not promote rank4/rank8 because the existing cost model estimates they lose the simple multiply-add wedge against exact four-sink QK. This gate remains a Mac-local quality-control diagnostic and should not be described as benchmark success, GPU speed, or cross-model predictor transfer.
