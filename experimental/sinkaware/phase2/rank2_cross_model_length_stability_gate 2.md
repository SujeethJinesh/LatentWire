# SinkAware Rank-2 Cross-Model Length Stability Gate

Status: **ALIVE but bounded; rank-2 survives cross-family length stability.**

- models: `distilgpt2`, `facebook/opt-125m`
- traces: 48
- max lengths: 64, 96
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This gate broadens the held-out/cross-family falsification row across sequence lengths. It fits all-head rank-2 predictors separately per model and length on whole-trace train splits, evaluates held-out traces against a position-only predictor, keeps non-sink QK scores exact, and makes no GPU speed or downstream-quality claim.

## Aggregate Across Model/Length Rows

| Metric | Mean | 95% CI |
|---|---:|---:|
| output rel-L2 improvement vs position | 0.0535 | +/- 0.0262 |
| layer-head output win rate | 0.982 | +/- 0.008 |
| minimum model/length output improvement | 0.0301 | |
| model/length rows positive | 4 / 4 | |

Positive improvement means rank-2 has lower error than position-only.

## Per Model And Length

| Max length | Model | Family | Position output rel-L2 | Rank2 output rel-L2 | Improvement | 95% CI | Head win rate | All seeds positive? |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 64 | distilgpt2 | gpt2 | 0.1540 | 0.1235 | +0.0306 | +/- 0.0023 | 0.986 | yes |
| 64 | facebook/opt-125m | opt | 0.3519 | 0.2731 | +0.0788 | +/- 0.0069 | 0.991 | yes |
| 96 | distilgpt2 | gpt2 | 0.1506 | 0.1205 | +0.0301 | +/- 0.0018 | 0.972 | yes |
| 96 | facebook/opt-125m | opt | 0.3479 | 0.2735 | +0.0744 | +/- 0.0040 | 0.979 | yes |

## Decision

This gate strengthens the Mac-local decision surface by requiring both GPT2-family and OPT-family rows to stay positive across length. It remains bounded attention-output drift evidence only; it does not establish predictor transfer, GPU speed, or downstream quality preservation.
