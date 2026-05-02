# ARC Transport Common-Basis Gate

- pass gate: `False`
- selected method: `whitened_procrustes`
- selected view: `query_residual`
- selected transform: `raw`
- selected parameter: `dim=32`
- validation disagreement rows: `144`
- test disagreement rows: `473`
- test matched mean: `0.228753`
- test Qwen-substituted mean: `0.317125`
- test cached Tiny packet mean: `0.269345`
- test delta vs Qwen-sub: `-0.088372`
- test CI95 low vs Qwen-sub min: `-0.160677`
- candidate-roll control mean: `0.251163`
- spectral-permutation control mean: `0.257082`

## Lay Explanation

This run keeps the same tiny ARC packet, but changes how TinyLlama's hidden/query vectors are translated into the public receiver coordinate system. Instead of a direct PCA/ridge fit, it tries local nearest-neighbor transport, sign-projected transport, and orthogonal Procrustes alignment. The key comparison is still against the stronger Qwen-substituted packet on the same disagreement rows.

## Interpretation

A pass would revive the TinyLlama hidden/query connector with a more principled common-basis map. A failure rules out another shallow Mac-local connector family and pushes the positive method gate back to a stronger true cross-family source or a trainable query/cache connector on NVIDIA.
