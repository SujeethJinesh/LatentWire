# ARC Hidden/Query MLP Cache Connector Gate

- pass gate: `False`
- selected view: `query_residual`
- selected pca/hidden/weight_decay: `16` / `16` / `0.001`
- validation disagreement rows: `144`
- test disagreement rows: `473`
- frontier candidates: `36`
- test matched mean: `0.231712`
- test Qwen-substituted mean: `0.317125`
- test cached Tiny mean: `0.269345`
- test delta vs Qwen-sub: `-0.085412`
- test CI95 low vs Qwen-sub: `-0.154334`
- candidate-roll control mean: `0.261311`
- content-rotation control mean: `0.255391`
- spectral-permutation control mean: `0.235941`

## Lay Explanation

This experiment tries a tiny learned translator instead of another hand-built geometric map. TinyLlama looks at the question and answer choices, the translator compresses each candidate's cached hidden/query vector into the same public packet coordinates, and only a 12-byte packet reaches the receiver. The hard comparison is whether that learned packet beats simply using Qwen's own packet on the same disagreement rows.

## Interpretation

This is a bounded Mac-local proxy for a query-bottleneck connector, not a full target-LM soft-prefix claim. A pass would promote the learned cache-to-packet branch for larger training and cross-family tests. A failure weakens the claim that the current TinyLlama hidden/query means contain a low-data connector into the ARC public packet basis, while leaving open tokenwise query/KV connectors that need new extraction infrastructure or NVIDIA runs.
