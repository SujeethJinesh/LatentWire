# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 24
- max length: 96
- sink tokens: 2
- train fraction: 0.67
- seeds: 0, 1, 2

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.043328 | +/- 0.031673 |
| rank-2 KL-to-exact improvement vs position-only | 0.040809 | +/- 0.054102 |
| minimum model loss-delta improvement | 0.027168 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.05299519 | 0.02582673 | +0.02716846 | +0.01320622 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.16210150 | 0.10261351 | +0.05948799 | +0.06841226 | yes |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 8 | 2.824739 | +0.00000000 | +0.04864380 | +0.02349754 | 0.03848918 | 0.02316326 | 0.0871 | 0.0726 |
| distilgpt2 | 1 | 8 | 2.661540 | +0.00000000 | +0.05831870 | +0.01967767 | 0.03243087 | 0.01984897 | 0.0996 | 0.0797 |
| distilgpt2 | 2 | 8 | 2.764125 | +0.00000000 | +0.05202307 | +0.03430499 | 0.03133954 | 0.01962871 | 0.0960 | 0.0725 |
| facebook/opt-125m | 0 | 8 | 2.510296 | +0.00000000 | +0.16293556 | +0.10836617 | 0.15578487 | 0.08929480 | 0.2218 | 0.1610 |
| facebook/opt-125m | 1 | 8 | 2.355124 | +0.00000000 | +0.17125007 | +0.11024402 | 0.13163522 | 0.06621846 | 0.2071 | 0.1411 |
| facebook/opt-125m | 2 | 8 | 2.466056 | +0.00000000 | +0.15211887 | +0.08923033 | 0.14789514 | 0.07456521 | 0.1625 | 0.1196 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
