# HellaSwag Anchor Feature Variant Scout

- scout pass: `False`
- eval rows: `1024`
- best variant: `cosine_full`
- best accuracy: `0.512695`
- best label-copy accuracy: `0.500000`
- delta vs best label-copy: `0.012695`
- CI95 vs best label-copy: `[-0.002930, 0.025391]`
- score-only bagged control: `0.497070`
- dense hidden-innovation reference: `0.503125`
- packet: `2B` raw / `5B` framed

## Interpretation

This is a bounded rescue scout for the failed anchor-relative common-basis branch. It tests whether local-neighborhood, RBF, spectral/Fourier-like, or QJL sign-sketch coordinates can recover the dense hidden-innovation signal from existing cached source hiddens. Because variants are compared on the eval slice, any success only promotes a predeclared all-slice gate; a failure weakens anchor-feature rescues and pushes the next branch toward learned sparse/crosscoder-style bases.
