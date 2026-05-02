# ARC Sparse-Query Cache-Bottleneck Gate

- pass gate: `False`
- selected view: `query_residual`
- selected pca/rff/active/gamma/ridge: `16` / `32` / `8` / `0.5` / `10.0`
- validation disagreement rows: `32`
- test disagreement rows: `64`
- test matched mean: `0.265625`
- test Qwen-substituted mean: `0.223958`
- test cached Tiny mean: `0.385417`
- test delta vs Qwen-sub: `0.041667`
- test CI95 low vs Qwen-sub: `-0.164453`
- candidate-roll control mean: `0.255208`
- content-rotation control mean: `0.250000`
- spectral-permutation control mean: `0.197917`

## Lay Explanation

This run gives qwen2.5_1.5b a small learned nonlinear bottleneck before it sends the same 8-byte ARC hint. The bottleneck acts like a set of sparse queries over the source model's hidden/query state, then translates those query activations into the public packet coordinate system. The hidden/query vectors themselves are not sent.

## Interpretation

A pass would revive the ARC hidden/query branch with a real nonlinear source-private connector. A failure weakens another Mac-local hidden/query connector family and leaves a stronger true cross-family source or larger learned query/cache connector on NVIDIA as the next live branch.
