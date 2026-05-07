# SinkAware All-Rank2 Split/Seed Stability Gate

Status: **ALIVE but still weak; all-head rank-2 beats position-only across randomized split seeds.**

- model: `distilgpt2`
- traces: 48
- max length: 96
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This gate randomizes the token-level train/test split across seeds and evaluates all-head rank-2 against the position-only predictor. It keeps all non-sink QK scores exact and makes no GPU speed claim.

## Aggregate Across Seeds

| Metric | Mean improvement | 95% CI |
|---|---:|---:|
| output rel-L2 vs position | 0.0368 | +/- 0.0006 |
| sink-mass MAE vs position | 0.0203 | +/- 0.0003 |
| attention L1 vs position | 0.0435 | +/- 0.0004 |
| layer-head output win rate | 0.282 | +/- 0.024 |

Positive improvement means rank-2 has lower error than position-only.

## Per Seed

| Seed | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |
|---:|---:|---:|---:|---:|
| 0 | 0.1709 | 0.1336 | +0.0373 | 0.278 |
| 1 | 0.1717 | 0.1354 | +0.0363 | 0.306 |
| 2 | 0.1751 | 0.1383 | +0.0368 | 0.264 |

## Decision

The simple validation head selector remains ruled out. This gate only tests whether the all-head rank-2 row is repeatable enough to justify interpreter/GPU work.
