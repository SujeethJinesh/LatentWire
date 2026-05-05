# SinkAware QK-Sink Approximation Cost Model

Status: **ALIVE at low rank; rank-2 gives useful QK-sink prediction below exact QK cost.**

- model: `distilgpt2`
- heads: 12
- head dim: 64
- sink tokens: 4

This estimates multiply-adds per token per layer for exact sink logits versus a per-head low-rank query predictor.
It is not a GPU benchmark and ignores memory layout, launch overhead, and approximation error impact on model quality.

| Rank | Approx mul-adds | Exact QK-sink mul-adds | Cost ratio | Hidden+pos R2 | Passes pre-GPU tradeoff? |
|---:|---:|---:|---:|---:|---|
| 1 | 816 | 3072 | 0.266 | 0.257 | no |
| 2 | 1632 | 3072 | 0.531 | 0.420 | yes |
| 4 | 3264 | 3072 | 1.062 | 0.512 | no |
| 8 | 6528 | 3072 | 2.125 | 0.712 | no |

## Decision

The approximate branch has a plausible pre-GPU systems wedge only at low rank.
Rank-8 is more accurate but likely too expensive relative to exact four-sink `QK_sink`; rank-2 is the strongest current compromise.
