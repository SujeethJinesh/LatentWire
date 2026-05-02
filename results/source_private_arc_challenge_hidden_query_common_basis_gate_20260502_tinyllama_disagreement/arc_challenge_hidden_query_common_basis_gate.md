# ARC Hidden/Query Common-Basis Gate

- pass gate: `False`
- selected view: `hidden_residual`
- selected pca/ridge: `32` / `100.0`
- validation disagreement rows: `144`
- test disagreement rows: `473`
- test matched mean: `0.229598`
- test Qwen-substituted mean: `0.317125`
- test cached Tiny packet mean: `0.269345`
- test delta vs Qwen-sub: `-0.087526`
- test min CI95 low vs Qwen-sub: `-0.159672`
- candidate-roll control mean: `0.256660`
- spectral-permutation control mean: `0.251163`

## Lay Explanation

We take only the examples where TinyLlama and Qwen chose different answers.  TinyLlama's internal hidden/query vectors are compressed into the same small public coordinate system used by the ARC packet method, then the receiver tries to answer from a 12-byte packet.  The key comparison is not against doing nothing; it is against simply using the stronger Qwen packet on those same examples.

## Interpretation

This is a strict source-family common-basis gate.  A pass would mean TinyLlama hidden/query state can be translated into useful fixed-byte public-basis evidence beyond both the cached Tiny packet and the Qwen-substituted packet.  A failure weakens this ARC branch and says the current mapping is not yet a real cross-family latent language, even though it may remain useful as a falsification and systems accounting artifact.
