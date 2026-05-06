# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 48
- max length: 64
- sink tokens: 2
- train fraction: 0.67
- seeds: 0, 1, 2

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.062065 | +/- 0.070147 |
| rank-2 KL-to-exact improvement vs position-only | 0.053161 | +/- 0.077220 |
| minimum model loss-delta improvement | 0.026276 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.05136408 | 0.02508781 | +0.02627627 | +0.01376299 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.20510968 | 0.10725515 | +0.09785453 | +0.09255861 | yes |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 16 | 3.070104 | +0.00000000 | +0.05613570 | +0.02553087 | 0.04268184 | 0.02741828 | 0.0919 | 0.0709 |
| distilgpt2 | 1 | 16 | 3.057089 | +0.00000000 | +0.05973537 | +0.03119602 | 0.04034034 | 0.02607965 | 0.1045 | 0.0776 |
| distilgpt2 | 2 | 16 | 3.081672 | +0.00000000 | +0.03822117 | +0.01853654 | 0.03776113 | 0.02599642 | 0.1108 | 0.0848 |
| facebook/opt-125m | 0 | 16 | 2.727804 | +0.00000000 | +0.18833218 | +0.10436055 | 0.17974355 | 0.08678452 | 0.1924 | 0.1366 |
| facebook/opt-125m | 1 | 16 | 2.706090 | +0.00000000 | +0.20509470 | +0.11017747 | 0.16411189 | 0.07813956 | 0.1927 | 0.1370 |
| facebook/opt-125m | 2 | 16 | 2.687145 | +0.00000000 | +0.22190216 | +0.10722743 | 0.18131528 | 0.08257080 | 0.2074 | 0.1476 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
