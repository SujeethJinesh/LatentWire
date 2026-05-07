# SinkAware Downstream Quality/Control Gate

Status: **ALIVE but bounded; rank-2 passes the downstream GPT2/OPT control smoke.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 48
- max length: 96
- sink tokens: 2
- train fraction: 0.67
- seeds: 0, 1, 2

This gate patches full model attention during causal-LM evaluation. It compares exact baseline attention against an exact-replacement no-op control, a position-only sink-logit replacement, and rank-2 sink-logit replacement. Predictors are fit separately per model and split. This is not cross-model predictor transfer, benchmark success, or GPU speed evidence.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| rank-2 absolute loss-delta improvement vs position-only | 0.053681 | +/- 0.050875 |
| rank-2 KL-to-exact improvement vs position-only | 0.046411 | +/- 0.057960 |
| minimum model loss-delta improvement | 0.027724 | |

Positive improvement means rank-2 is closer to exact baseline behavior than position-only.

## Per Model

| Model | Family | Status | Exact abs loss delta | Position abs loss delta | Rank2 abs loss delta | Rank2 loss improvement | Rank2 KL improvement | Exact no-op ok? |
|---|---|---|---:|---:|---:|---:|---:|---|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.05373157 | 0.02600729 | +0.02772428 | +0.01683956 | yes |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 is closer than position-only in downstream loss and KL. | 0.00000000 | 0.19177025 | 0.11213293 | +0.07963733 | +0.07598230 | yes |

## Per Seed

| Model | Seed | Held-out traces | Exact loss | Exact replacement loss delta | Position loss delta | Rank2 loss delta | Position KL | Rank2 KL | Position top1 diff | Rank2 top1 diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 16 | 3.024935 | +0.00000000 | +0.05878588 | +0.02793764 | 0.04344434 | 0.02463361 | 0.0920 | 0.0736 |
| distilgpt2 | 1 | 16 | 3.025229 | +0.00000000 | +0.06350243 | +0.03083027 | 0.04132092 | 0.02275638 | 0.1046 | 0.0752 |
| distilgpt2 | 2 | 16 | 3.011390 | +0.00000000 | +0.03890640 | +0.01925397 | 0.03703686 | 0.02389346 | 0.1053 | 0.0762 |
| facebook/opt-125m | 0 | 16 | 2.701965 | +0.00000000 | +0.17790549 | +0.10670596 | 0.17424718 | 0.09666252 | 0.1850 | 0.1432 |
| facebook/opt-125m | 1 | 16 | 2.668158 | +0.00000000 | +0.19670407 | +0.12127604 | 0.15941819 | 0.08967722 | 0.1890 | 0.1474 |
| facebook/opt-125m | 2 | 16 | 2.613446 | +0.00000000 | +0.20070120 | +0.10841678 | 0.17202125 | 0.09139998 | 0.2039 | 0.1494 |

## Decision

The exact-replacement row is the validity control: it should reproduce baseline causal-LM loss and logits. If that row passes, the position-only and rank-2 rows can be read as downstream behavior drift from replacing only fixed sink logits while keeping non-sink attention scores exact. This gate remains a small Mac-local control diagnostic and should not be described as downstream benchmark success.
