# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 48
- max length: 64
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.080125 | +/- 0.089411 |
| rank-2 KL-to-exact improvement vs position-only | 0.075152 | +/- 0.100267 |
| minimum model loss-delta improvement | 0.034507 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.10293630 | 0.06842952 | +0.03450678 | +0.02399561 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.24171675 | 0.11597439 | +0.12574237 | +0.12630865 | yes |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 16 | 3.070104 | +0.00000000 | +0.09528243 | +0.06187837 | 0.07393447 | 0.05233890 | 0.1009 | 0.0899 |
| distilgpt2 | 1 | 16 | 3.057089 | +0.00000000 | +0.10516735 | +0.06842471 | 0.08133194 | 0.05608767 | 0.1294 | 0.1085 |
| distilgpt2 | 2 | 16 | 3.081672 | +0.00000000 | +0.10835911 | +0.07498547 | 0.09479986 | 0.06965286 | 0.1527 | 0.1218 |
| facebook/opt-125m | 0 | 16 | 2.727804 | +0.00000000 | +0.21729482 | +0.11067976 | 0.22022817 | 0.10649872 | 0.1944 | 0.1346 |
| facebook/opt-125m | 1 | 16 | 2.706090 | +0.00000000 | +0.22638589 | +0.11204825 | 0.20450777 | 0.08914999 | 0.1946 | 0.1341 |
| facebook/opt-125m | 2 | 16 | 2.687145 | +0.00000000 | +0.28146956 | +0.12519515 | 0.24605025 | 0.09621152 | 0.2243 | 0.1356 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
