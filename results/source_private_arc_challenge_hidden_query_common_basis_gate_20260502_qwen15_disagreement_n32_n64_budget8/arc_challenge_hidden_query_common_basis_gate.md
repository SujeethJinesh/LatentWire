# ARC Hidden/Query Common-Basis Gate

- pass gate: `False`
- selected view: `query_residual`
- selected pca/ridge: `32` / `10.0`
- validation disagreement rows: `32`
- test disagreement rows: `64`
- test matched mean: `0.234375`
- test Qwen-substituted mean: `0.223958`
- test cached source packet mean: `0.385417`
- test delta vs Qwen-sub: `0.010417`
- test min CI95 low vs Qwen-sub: `-0.187500`
- candidate-roll control mean: `0.302083`
- spectral-permutation control mean: `0.286458`

## Lay Explanation

We take only the examples where qwen2.5_1.5b and Qwen chose different answers.  The source model's internal hidden/query vectors are compressed into the same small public coordinate system used by the ARC packet method, then the receiver tries to answer from a 8-byte packet.  The key comparison is not against doing nothing; it is against simply using the stronger Qwen packet on those same examples.

## Interpretation

This is a strict source-family common-basis gate.  A pass would mean source hidden/query state can be translated into useful fixed-byte public-basis evidence beyond both the cached source packet and the Qwen-substituted packet.  A failure weakens this ARC branch and says the current mapping is not yet a real cross-family latent language, even though it may remain useful as a falsification and systems accounting artifact.
