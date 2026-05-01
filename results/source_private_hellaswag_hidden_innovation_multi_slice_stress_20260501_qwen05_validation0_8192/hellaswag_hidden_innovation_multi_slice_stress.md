# HellaSwag Hidden-Innovation Multi-Slice Stress

- pass gate: `True`
- slice count: `8`
- total eval rows: `8192`
- contiguous validation prefix: `True`
- weighted selected accuracy: `0.522949`
- weighted best label-copy accuracy: `0.482056`
- min delta vs best label-copy: `0.034180`
- min CI95 low vs best label-copy: `0.011719`
- min delta vs score-only bagged: `0.037109`
- min score-only CI95 low: `0.017065`
- min delta vs zero-hidden: `0.037109`
- all corrupted-hidden controls below label-copy: `True`
- jackknife slices passing: `8/8`
- packet: `2B` raw / `5B` framed

## Interpretation

This aggregate gate is the Mac-feasible bridge between a single heldout HellaSwag slice and full-validation evidence. It turns the current headline-candidate into a stricter multi-slice claim only if every contiguous validation block preserves the source-private advantage and all leakage controls still collapse.
