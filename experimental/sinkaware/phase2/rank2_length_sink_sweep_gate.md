# SinkAware All-Rank2 Length/Sink Sweep

Status: **ALIVE but bounded; all-head rank-2 beats position-only across the length/sink sweep.**

- model: `distilgpt2`
- max traces: 48
- max lengths: 64, 96
- sink tokens: 2, 4
- train fraction: 0.67
- seeds: 0, 1, 2

This sweep keeps the method fixed as all-head rank-2 and reuses the randomized token split gate for each configuration. It keeps all non-sink QK scores exact and makes no GPU speed claim.

## Aggregate Across Configurations

| Metric | Mean | 95% CI |
|---|---:|---:|
| output rel-L2 improvement vs position | 0.0366 | +/- 0.0024 |
| layer-head output win rate | 0.286 | +/- 0.010 |
| minimum config output improvement | 0.0342 | |

Positive improvement means rank-2 has lower error than position-only.

## Per Configuration

| Max length | Sink tokens | Position output rel-L2 | Rank2 output rel-L2 | Improvement | 95% CI | Head win rate | All seeds positive? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 2 | 0.1642 | 0.1287 | 0.0354 | +/- 0.0014 | 0.301 | yes |
| 64 | 4 | 0.1699 | 0.1357 | 0.0342 | +/- 0.0007 | 0.278 | yes |
| 96 | 2 | 0.1720 | 0.1320 | 0.0400 | +/- 0.0008 | 0.282 | yes |
| 96 | 4 | 0.1726 | 0.1358 | 0.0368 | +/- 0.0006 | 0.282 | yes |

## Decision

The exact static branch remains killed, and simple validation head selection remains ruled out. This sweep only decides whether all-head rank-2 quality is stable enough to justify interpreter/GPU work.
