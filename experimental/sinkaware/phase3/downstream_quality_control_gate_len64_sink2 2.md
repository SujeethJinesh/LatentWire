# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 24
- max length: 64
- sink tokens: 2
- train fraction: 0.67
- seeds: 0, 1, 2

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.054410 | +/- 0.050486 |
| rank-2 KL-to-exact improvement vs position-only | 0.048488 | +/- 0.069299 |
| minimum model loss-delta improvement | 0.028652 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.05318500 | 0.02453305 | +0.02865195 | +0.01313125 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.17027413 | 0.09010632 | +0.08016782 | +0.08384411 | yes |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 8 | 2.941694 | +0.00000000 | +0.05049317 | +0.02477874 | 0.03969168 | 0.02411681 | 0.0884 | 0.0743 |
| distilgpt2 | 1 | 8 | 2.677648 | +0.00000000 | +0.05761128 | +0.01844273 | 0.03236134 | 0.02053350 | 0.1032 | 0.0833 |
| distilgpt2 | 2 | 8 | 2.878404 | +0.00000000 | +0.05145053 | +0.03037768 | 0.03135998 | 0.01936893 | 0.0992 | 0.0813 |
| facebook/opt-125m | 0 | 8 | 2.659699 | +0.00000000 | +0.17093819 | +0.10456448 | 0.16572331 | 0.08471758 | 0.2425 | 0.1683 |
| facebook/opt-125m | 1 | 8 | 2.431580 | +0.00000000 | +0.17460543 | +0.09242081 | 0.13480720 | 0.05616898 | 0.2063 | 0.1111 |
| facebook/opt-125m | 2 | 8 | 2.588516 | +0.00000000 | +0.16527878 | +0.07333366 | 0.15594593 | 0.06405754 | 0.1706 | 0.1052 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
