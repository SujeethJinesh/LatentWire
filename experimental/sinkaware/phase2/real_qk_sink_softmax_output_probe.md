# SinkAware Per-Head Softmax/Output Error Probe

Status: **WEAKLY ALIVE for GPU gate; aggregate rank-2 improves, but paired per-head gains are concentrated.**

- model: `distilgpt2`
- traces: 48
- held-out token samples per layer: 1038
- sink tokens: 4

This probe keeps all non-sink QK scores exact and replaces only fixed sink-token logits.
It measures softmax drift and attention-output error on held-out tokens. It is not a GPU timing result.

## Mean Across Layers

| Predictor | Sink-logit RMSE | Sink-mass MAE | Attention L1 | Output rel-L2 |
|---|---:|---:|---:|---:|
| static | 1229.5088 | 0.0954 | 0.2165 | 0.1927 |
| position | 1197.3897 | 0.0756 | 0.1750 | 0.1700 |
| rank1 | 1079.5410 | 0.0670 | 0.1573 | 0.1596 |
| rank2 | 1001.5057 | 0.0550 | 0.1350 | 0.1408 |
| rank4 | 838.3202 | 0.0448 | 0.1151 | 0.1221 |
| rank8 | 699.2322 | 0.0396 | 0.1042 | 0.1075 |

## Layer-Head Paired Improvement vs Position-Only

Layer-head cells: 72. Positive values are lower error than the position-only predictor.

| Predictor | Output rel-L2 improvement | 95% CI | Output win rate | Sink-mass MAE improvement | Attention L1 improvement |
|---|---:|---:|---:|---:|---:|
| static | -0.0463 | +/- 0.0338 | 0.056 | -0.0198 | -0.0416 |
| rank1 | 0.0257 | +/- 0.0290 | 0.250 | 0.0086 | 0.0177 |
| rank2 | 0.0297 | +/- 0.0378 | 0.278 | 0.0206 | 0.0400 |
| rank4 | 0.0596 | +/- 0.0465 | 0.333 | 0.0308 | 0.0599 |
| rank8 | 0.0955 | +/- 0.0805 | 0.347 | 0.0360 | 0.0707 |

## Decision

Exact static sink reuse remains killed because sink logits are query-dependent.
Rank-2 is the only current low-rank compromise that stays below exact four-sink QK cost; its aggregate improvement and weak paired per-head gains are the Mac-local evidence for a correctness gate.
The relevant question is whether a cheap per-head low-rank approximation preserves softmax and output quality well enough to justify a native kernel gate.
