# SinkAware Rank-2 Held-Out/Cross-Family Falsification Gate

Status: **ALIVE but bounded; rank-2 survives the repeated held-out/cross-family falsification gate.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 24
- max length: 64
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This is the larger Mac-feasible repeat after the one-seed smoke gate. It fits all-head rank-2 predictors separately per model on whole-trace train splits and evaluates held-out traces against a position-only predictor. It does not transfer predictors across models and makes no GPU speed claim.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| output rel-L2 improvement vs position | 0.0557 | +/- 0.0424 |
| minimum model output improvement | 0.0341 | |

Positive improvement means rank-2 has lower error than position-only.

## Per Model

| Model | Family | Status | Output improvement | 95% CI | Head win rate |
|---|---|---|---:|---:|---:|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 beats position-only across held-out trace splits. | 0.0341 | +/- 0.0018 | 0.958 |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 beats position-only across held-out trace splits. | 0.0774 | +/- 0.0043 | 0.972 |

## Per Seed

| Model | Seed | Train traces | Held-out traces | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 16 | 8 | 0.1619 | 0.1272 | +0.0348 | 0.972 |
| distilgpt2 | 1 | 16 | 8 | 0.1566 | 0.1214 | +0.0352 | 0.958 |
| distilgpt2 | 2 | 16 | 8 | 0.1572 | 0.1249 | +0.0323 | 0.944 |
| facebook/opt-125m | 0 | 16 | 8 | 0.3656 | 0.2893 | +0.0763 | 0.972 |
| facebook/opt-125m | 1 | 16 | 8 | 0.3434 | 0.2618 | +0.0816 | 0.986 |
| facebook/opt-125m | 2 | 16 | 8 | 0.3451 | 0.2709 | +0.0742 | 0.958 |

## Decision

This repeated gate is still smaller than the 48-trace distilgpt2 frozen split, but it is stronger than the prior cross-family smoke because it repeats whole-trace splits and includes an OPT-family model. Passing this gate keeps the branch alive only as bounded Mac-local evidence; promotion still requires Triton interpreter correctness, native timing evidence, and broader benchmark controls.
