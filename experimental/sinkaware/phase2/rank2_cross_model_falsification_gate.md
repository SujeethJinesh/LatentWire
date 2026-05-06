# SinkAware Rank-2 Held-Out/Cross-Family Falsification Gate

Status: **ALIVE but bounded; rank-2 survives the small held-out/cross-family falsification smoke gate.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 12
- max length: 64
- sink tokens: 4
- train fraction: 0.67
- seeds: 0

This is a smallest Mac-feasible falsification gate after the Triton interpreter path was blocked locally. It fits all-head rank-2 predictors separately per model on whole-trace train splits and evaluates held-out traces against a position-only predictor. It does not transfer predictors across models and makes no GPU speed claim.

## Aggregate Across Models

| Metric | Mean | 95% CI |
|---|---:|---:|
| output rel-L2 improvement vs position | 0.0519 | +/- 0.0372 |
| minimum model output improvement | 0.0329 | |

Positive improvement means rank-2 has lower error than position-only.

## Per Model

| Model | Family | Status | Output improvement | 95% CI | Head win rate |
|---|---|---|---:|---:|---:|
| distilgpt2 | gpt2 | ALIVE but bounded on this model; rank-2 beats position-only across held-out trace splits. | 0.0329 | +/- 0.0000 | 0.958 |
| facebook/opt-125m | opt | ALIVE but bounded on this model; rank-2 beats position-only across held-out trace splits. | 0.0709 | +/- 0.0000 | 0.931 |

## Per Seed

| Model | Seed | Train traces | Held-out traces | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 0 | 8 | 4 | 0.1602 | 0.1273 | +0.0329 | 0.958 |
| facebook/opt-125m | 0 | 8 | 4 | 0.3562 | 0.2853 | +0.0709 | 0.931 |

## Decision

This smoke gate is weaker than the 48-trace distilgpt2 frozen split because it uses a smaller slice, but it is stronger as a falsification attempt because it includes a held-out OPT-family model. Passing this gate keeps the branch alive only as bounded Mac-local evidence; promotion still requires larger cross-family repeats or Triton interpreter correctness.
