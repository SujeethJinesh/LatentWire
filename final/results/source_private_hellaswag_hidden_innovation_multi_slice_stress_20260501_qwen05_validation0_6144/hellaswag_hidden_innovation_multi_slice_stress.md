# HellaSwag Hidden-Innovation Multi-Slice Stress

- pass gate: `True`
- slice count: `6`
- total eval rows: `6144`
- contiguous validation prefix: `True`
- weighted selected accuracy: `0.511882`
- weighted best label-copy accuracy: `0.470052`
- min delta vs best label-copy: `0.034180`
- min CI95 low vs best label-copy: `0.011719`
- min delta vs score-only bagged: `0.041016`
- min score-only CI95 low: `0.024414`
- min delta vs zero-hidden: `0.041016`
- all corrupted-hidden controls below label-copy: `True`
- jackknife slices passing: `6/6`
- packet: `2B` raw / `5B` framed

## Interpretation

This aggregate gate is the Mac-feasible bridge between a single heldout HellaSwag slice and full-validation evidence. It turns the current headline-candidate into a stricter multi-slice claim only if every contiguous validation block preserves the source-private advantage and all leakage controls still collapse.
