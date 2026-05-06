# SinkAware All-Rank2 Trace-Level Frozen Split Gate

Status: **ALIVE but bounded; all-head rank-2 beats position-only across trace-level frozen splits.**

- model: `distilgpt2`
- traces: 24
- max length: 96
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This gate freezes whole traces into train and held-out sets for each seed. It fits all-head rank-2 on train traces and evaluates output drift on held-out traces. It keeps all non-sink QK scores exact and makes no GPU speed claim.

## Aggregate Across Trace Splits

| Metric | Mean improvement | 95% CI |
|---|---:|---:|
| output rel-L2 vs position | 0.0398 | +/- 0.0014 |
| sink-mass MAE vs position | 0.0265 | +/- 0.0006 |
| attention L1 vs position | 0.0540 | +/- 0.0014 |
| layer-head output win rate | 0.287 | +/- 0.018 |
| minimum output rel-L2 improvement | 0.0387 | |

Positive improvement means rank-2 has lower error than position-only.

## Per Trace Split

| Seed | Train traces | Held-out traces | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 8 | 0.1759 | 0.1348 | +0.0411 | 0.306 |
| 1 | 16 | 8 | 0.1771 | 0.1374 | +0.0397 | 0.278 |
| 2 | 16 | 8 | 0.1755 | 0.1368 | +0.0387 | 0.278 |

## Decision

This is stronger than token-level randomized splits because no text trace appears in both train and held-out sets. It still only measures Mac-local attention-output drift, not downstream quality or GPU latency.
