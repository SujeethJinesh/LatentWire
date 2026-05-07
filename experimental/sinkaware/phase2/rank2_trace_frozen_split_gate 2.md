# SinkAware All-Rank2 Trace-Level Frozen Split Gate

Status: **ALIVE but bounded; all-head rank-2 beats position-only across trace-level frozen splits.**

- model: `distilgpt2`
- traces: 48
- max length: 96
- sink tokens: 4
- train fraction: 0.67
- seeds: 0, 1, 2

This gate freezes whole traces into train and held-out sets for each seed. It fits all-head rank-2 on train traces and evaluates output drift on held-out traces. It keeps all non-sink QK scores exact and makes no GPU speed claim.

## Aggregate Across Trace Splits

| Metric | Mean improvement | 95% CI |
|---|---:|---:|
| output rel-L2 vs position | 0.0379 | +/- 0.0014 |
| sink-mass MAE vs position | 0.0220 | +/- 0.0008 |
| attention L1 vs position | 0.0461 | +/- 0.0014 |
| layer-head output win rate | 0.278 | +/- 0.016 |
| minimum output rel-L2 improvement | 0.0367 | |

Positive improvement means rank-2 has lower error than position-only.

## Per Trace Split

| Seed | Train traces | Held-out traces | Position output rel-L2 | Rank2 output rel-L2 | Improvement | Head win rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 16 | 0.1753 | 0.1360 | +0.0392 | 0.292 |
| 1 | 32 | 16 | 0.1753 | 0.1374 | +0.0378 | 0.264 |
| 2 | 32 | 16 | 0.1707 | 0.1340 | +0.0367 | 0.278 |

## Decision

This is stronger than token-level randomized splits because no text trace appears in both train and held-out sets. It still only measures Mac-local attention-output drift, not downstream quality or GPU latency.
