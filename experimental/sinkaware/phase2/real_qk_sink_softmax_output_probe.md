# SinkAware Per-Head Softmax/Output Error Probe

Status: **ALIVE for GPU gate; rank-2 improves output error over position-only with bounded drift.**

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

## Decision

Exact static sink reuse remains killed because sink logits are query-dependent.
The relevant question is whether a cheap per-head low-rank approximation preserves softmax and output quality well enough to justify a GPU kernel gate.
