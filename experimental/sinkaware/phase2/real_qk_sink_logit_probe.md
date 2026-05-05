# SinkAware Real Q/K Sink-Logit Probe

Status: **ALIVE as approximate QK-sink predictor; next gate is full per-head error and kernel cost model.**

- model: `distilgpt2`
- traces: 24
- token samples: 9450
- sink tokens: 4

This computes GPT-style Q/K sink logits on CPU and tests a query-side low-rank approximation.
It is not an exact-kernel result and not a GPU timing result.

| Static R2 | Position R2 | Rank-1 hidden R2 | Rank-2 hidden R2 | Rank-4 hidden R2 | Rank-8 hidden R2 | Best hidden+pos R2 |
|---:|---:|---:|---:|---:|---:|---:|
| -0.002 | 0.153 | 0.195 | 0.361 | 0.488 | 0.701 | 0.712 |

## Per Layer

| Layer | Samples | Static R2 | Position R2 | Rank-8 hidden R2 | Rank-8 hidden+pos R2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1575 | -0.000 | 0.531 | 0.962 | 0.967 |
| 1 | 1575 | -0.003 | 0.170 | 0.557 | 0.569 |
| 2 | 1575 | -0.007 | 0.065 | 0.578 | 0.607 |
| 3 | 1575 | -0.002 | 0.066 | 0.750 | 0.749 |
| 4 | 1575 | -0.000 | 0.035 | 0.627 | 0.636 |
| 5 | 1575 | -0.002 | 0.048 | 0.732 | 0.742 |

## Decision

This is the strongest Mac-local SinkAware revival evidence so far if hidden+position beats position-only on real Q/K logits.
The exact static-prior branch remains killed. The live branch is approximate low-rank QK-sink prediction or a fused exact path with a cost model.
