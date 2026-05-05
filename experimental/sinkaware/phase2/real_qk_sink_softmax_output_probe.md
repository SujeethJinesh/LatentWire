# SinkAware Per-Head Softmax/Output Error Probe

Status: **ALIVE for GPU gate; rank-2 improves output error over position-only with bounded drift.**

- model: `distilgpt2`
- traces: 24
- held-out token samples per layer: 520
- sink tokens: 4

This probe keeps all non-sink QK scores exact and replaces only fixed sink-token logits.
It measures softmax drift and attention-output error on held-out tokens. It is not a GPU timing result.

## Mean Across Layers

| Predictor | Sink-logit RMSE | Sink-mass MAE | Attention L1 | Output rel-L2 |
|---|---:|---:|---:|---:|
| static | 1182.8939 | 0.1031 | 0.2187 | 0.1981 |
| position | 1141.3060 | 0.0802 | 0.1701 | 0.1731 |
| rank1 | 1027.9389 | 0.0719 | 0.1520 | 0.1651 |
| rank2 | 940.8725 | 0.0546 | 0.1174 | 0.1341 |
| rank4 | 774.7908 | 0.0437 | 0.0963 | 0.1144 |
| rank8 | 618.0497 | 0.0352 | 0.0782 | 0.0965 |

## Decision

Exact static sink reuse remains killed because sink logits are query-dependent.
The relevant question is whether a cheap per-head low-rank approximation preserves softmax and output quality well enough to justify a GPU kernel gate.
