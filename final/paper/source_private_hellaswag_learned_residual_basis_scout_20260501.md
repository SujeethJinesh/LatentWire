# HellaSwag Learned Residual Basis Scout

## Status

The learned residual basis branch is alive but not promoted. A train-only
`pca256` basis passes the strongest single-slice scout, but it fails the
predeclared five-slice promotion gate.

## Why This Gate Was Run

Dense hidden innovation works, while anchor-relative and random sign-sketch
common bases failed. This gate tested whether a learned residual coordinate
system, fit only on train rows, can preserve the source's useful hidden signal
while keeping the communication boundary fixed at `2B` raw / `5B` framed.

In lay terms: instead of using random rulers, we learned a small set of useful
directions from training examples and asked whether the sender's private hunch
can be expressed in those directions.

## Scout Result

Artifact:
`results/source_private_hellaswag_learned_residual_basis_scout_20260501_qwen05_validation4096_5120/hellaswag_learned_residual_basis_scout.json`

Decision slice: HellaSwag validation rows `4096:5120`.

Best variant: `pca256_top2_gold`.

- accuracy: `0.528320`
- best label-copy: `0.500000`
- delta vs label-copy: `+0.028320`
- CI95 low vs label-copy: `+0.010742`
- score-only bagged: `0.497070`
- delta vs score-only: `+0.031250`
- zero-hidden: `0.497070`
- wrong-example hidden: `0.473633`
- candidate-roll hidden: `0.458984`
- basis-roll: `0.492188`
- sign-flip: `0.430664`
- random-basis: `0.498047`

This passes the scout rule and is the first shared-coordinate-style branch to
beat the anchor and sign-sketch variants on the strongest slice.

## Five-Slice Promotion Result

Artifact:
`results/source_private_hellaswag_learned_residual_basis_multi_slice_stress_20260501_qwen05_validation0_5120/hellaswag_learned_residual_basis_multi_slice_stress.json`

Result: `pass_gate=false`.

- strict pass slices: `2/5`
- total rows: `5120`
- weighted selected accuracy: `0.476172`
- weighted best label-copy: `0.461523`
- weighted score-only: `0.456445`
- weighted delta vs label-copy: `+0.014648`
- min delta vs label-copy: `-0.000977`
- min CI95 low vs label-copy: `-0.017578`
- weighted delta vs score-only: `+0.019727`
- packet: `2B` raw / `5B` framed

## Interpretation

The learned basis result is scientifically useful but not yet paper-ready as a
positive common-basis method. It shows that learned residual coordinates can
recover more source signal than anchors or random sign sketches, but PCA is not
stable enough across the frozen validation prefix.

This promotes a more serious learned sparse/crosscoder-style branch, not another
round of hand-designed anchors or random projections. The next method should
learn a sparse dictionary with explicit reconstruction and decision objectives,
then rerun the same all-slice gate.

## Contribution Accounting

Alive:

1. Dense hidden-innovation packet remains the only fully promoted positive
   method.
2. Learned residual basis is a promising bridge diagnostic.
3. The evaluation ladder now cleanly separates dense hidden signal, anchor
   common-basis failure, random-sketch failure, and learned-basis partial
   recovery.

Still blocked:

- ICLR common-basis claim.
- Strict cross-family falsification pair.
- Native GPU systems rows.
