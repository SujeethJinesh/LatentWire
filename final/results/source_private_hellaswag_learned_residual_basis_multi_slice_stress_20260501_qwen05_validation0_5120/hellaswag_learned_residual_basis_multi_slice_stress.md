# HellaSwag Learned Residual Basis Multi-Slice Stress

- pass gate: `False`
- slice count: `5`
- strict pass slices: `2/5`
- total eval rows: `5120`
- contiguous validation prefix: `True`
- weighted selected accuracy: `0.476172`
- weighted best label-copy accuracy: `0.461523`
- weighted score-only accuracy: `0.456445`
- weighted delta vs best label-copy: `0.014648`
- min delta vs best label-copy: `-0.000977`
- min CI95 low vs best label-copy: `-0.017578`
- min delta vs score-only: `0.003906`
- packet: `2B` raw / `5B` framed

## Interpretation

The learned residual basis is a promising bridge on the strongest scout slice, but it does not survive the all-slice promotion gate. It should be treated as alive only as evidence that learned basis dimension matters; PCA is not enough for the ICLR common-basis claim. The next branch should use a richer sparse/crosscoder objective or focus the paper on dense hidden innovation plus systems.
