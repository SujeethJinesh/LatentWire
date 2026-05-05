# SinkAware Real-Query Approximation Probe

Status: **ALIVE as approximate real-query predictor; next gate is real Q/K or kernel-side profiling.**

- model: `distilgpt2`
- traces: 24
- token samples: 9450
- sink tokens: 4

This probes real model attention sink mass from saved LatentWire text traces.
It is not a kernel benchmark, and it does not revive exact static sink reuse.

## Mean Across Layers

| Model | Static R2 | Position R2 | Rank-1 hidden R2 | Rank-2 hidden R2 | Rank-4 hidden R2 | Rank-8 hidden R2 | Best hidden+pos R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | -0.001 | 0.520 | 0.147 | 0.263 | 0.437 | 0.622 | 0.693 |

## Per Layer

| Layer | Samples | Static R2 | Position R2 | Rank-8 hidden R2 | Rank-8 hidden+pos R2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1575 | -0.000 | 0.710 | 0.904 | 0.939 |
| 1 | 1575 | -0.001 | 0.462 | 0.529 | 0.530 |
| 2 | 1575 | -0.002 | 0.568 | 0.642 | 0.641 |
| 3 | 1575 | -0.000 | 0.467 | 0.519 | 0.586 |
| 4 | 1575 | -0.002 | 0.426 | 0.573 | 0.713 |
| 5 | 1575 | -0.000 | 0.485 | 0.566 | 0.750 |

## Decision

The only useful pre-GPU SinkAware path is approximate. A static prior remains killed.
Advance only if real query features predict sink mass materially better than position-only structure.
