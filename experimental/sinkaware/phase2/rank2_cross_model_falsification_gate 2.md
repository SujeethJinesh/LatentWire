# SinkAware Rank-2 Held-Out/Cross-Family Falsification Gate

Status: **ALIVE but bounded; rank-2 survives the repeated held-out/cross-family falsification gate.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 48
- max length: 64
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This is the larger Mac-feasible repeat after the one-seed smoke gate. It fits all-head rank-2 predictors separately per model on whole-trace train splits and evaluates held-out traces against a position-only predictor. It does not transfer predictors across models and makes no GPU speed claim.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| output rel-L2 improvement vs position | 0.0547 | +/- 0.0472 |
| minimum model output improvement | 0.0306 | |

Positive improvement means rank-2 has lower error than position-only.

## Per Model

| Model | Family | Status | Output improvement | 95% CI | Head win rate |
|---|---|---|---:|---:|---:|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 beats position-only across held-out trace splits. | 0.0306 | +/- 0.0023 | 0.986 |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 beats position-only across held-out trace splits. | 0.0788 | +/- 0.0069 | 0.991 |

## Per Seed

| Model | Seed | Train traces | Held-out traces | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 32 | 16 | 0.1521 | 0.1238 | +0.0283 | 0.972 |
| distilgpt2 | 1 | 32 | 16 | 0.1536 | 0.1224 | +0.0312 | 0.986 |
| distilgpt2 | 2 | 32 | 16 | 0.1564 | 0.1242 | +0.0322 | 1.000 |
| facebook/opt-125m | 0 | 32 | 16 | 0.3562 | 0.2831 | +0.0731 | 0.986 |
| facebook/opt-125m | 1 | 32 | 16 | 0.3465 | 0.2686 | +0.0780 | 1.000 |
| facebook/opt-125m | 2 | 32 | 16 | 0.3528 | 0.2676 | +0.0852 | 0.986 |

## Decision

This repeated gate now matches the 48-trace distilgpt2 frozen split size while preserving strict GPT2-family versus OPT-family separation. Passing this gate keeps the branch alive only as bounded Mac-local measured drift evidence. Triton interpreter correctness is already cleared locally; promotion now requires the native NVIDIA packet gate with matched quality drift, downstream loss/KL/top-1 checks, repeated latency, and NCU memory/HBM counters.
