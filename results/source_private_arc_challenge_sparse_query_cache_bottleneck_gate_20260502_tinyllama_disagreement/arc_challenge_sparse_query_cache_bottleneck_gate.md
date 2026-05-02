# ARC Sparse-Query Cache-Bottleneck Gate

- pass gate: `False`
- selected view: `hidden_query_residual`
- selected pca/rff/active/gamma/ridge: `16` / `32` / `16` / `1.0` / `1000.0`
- validation disagreement rows: `144`
- test disagreement rows: `473`
- test matched mean: `0.248203`
- test Qwen-substituted mean: `0.317125`
- test cached Tiny mean: `0.269345`
- test delta vs Qwen-sub: `-0.068922`
- test CI95 low vs Qwen-sub: `-0.138531`
- candidate-roll control mean: `0.260465`
- content-rotation control mean: `0.281607`
- spectral-permutation control mean: `0.251163`

## Lay Explanation

This run gives TinyLlama a small learned nonlinear bottleneck before it sends the same 12-byte ARC hint. The bottleneck acts like a set of sparse queries over TinyLlama's hidden/query state, then translates those query activations into the public packet coordinate system. The hidden/query vectors themselves are not sent.

## Interpretation

A pass would revive the ARC hidden/query branch with a real nonlinear source-private connector. A failure kills another Mac-local TinyLlama hidden/query connector family and leaves a stronger true cross-family source or larger learned query/cache connector on NVIDIA as the next live branch.
