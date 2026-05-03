# HellaSwag Hidden-Innovation Multi-Slice Stress

- pass gate: `True`
- slice count: `9`
- total eval rows: `9216`
- contiguous validation prefix: `True`
- weighted selected accuracy: `0.525499`
- weighted best label-copy accuracy: `0.483941`
- min delta vs best label-copy: `0.034180`
- min CI95 low vs best label-copy: `0.011719`
- min delta vs source-rank/index-only bagged: `0.037109`
- min source-rank/index-only CI95 low: `0.018555`
- min delta vs score-only bagged: `0.037109`
- min score-only CI95 low: `0.018555`
- min delta vs zero-hidden: `0.037109`
- strict rank/score-channel control slices: `9/9`
- all corrupted-hidden controls below label-copy: `True`
- jackknife slices passing: `9/9`
- packet: `2B` raw / `5B` framed

## Interpretation

This aggregate gate is the Mac-feasible bridge between a single heldout HellaSwag slice and full-validation evidence. It turns the current headline-candidate into a stricter multi-slice claim only if every contiguous validation block preserves the source-private advantage and all rank, score, and hidden-corruption controls still collapse.
