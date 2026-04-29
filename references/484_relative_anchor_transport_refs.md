# Relative-Anchor Transport References

Date: 2026-04-29

## Blocker Addressed

The live scalar packet is strong on fixed slot codebooks but weaker under
candidate-order remapping. This memo records primary sources motivating a
relative-anchor packet: encode the source posterior in coordinates defined by
the public candidate anchors, then transmit a tiny score vector instead of an
absolute latent.

## Sources

### Relative Representations

Primary source: https://arxiv.org/abs/2209.15430

- Blocker: absolute latent coordinates are brittle under model/codebook
  transformations.
- Mechanism: represent examples by similarities to anchors so coordinates are
  more invariant to isometries/rescalings.
- Experiment impact: motivated `relative_scores`, a packet of quantized
  candidate-anchor scores.
- Role: method inspiration and paper framing.

### CKA

Primary source: https://arxiv.org/abs/1905.00414

- Blocker: reviewers will ask whether packet variants really change the
  representation geometry.
- Mechanism: compare representations with centered kernel alignment rather than
  raw coordinate agreement.
- Experiment impact: supports future diagnostics for scalar vs relative packet
  geometry; does not change the immediate gate.
- Role: diagnostic support.

### SVCCA

Primary source: https://arxiv.org/abs/1706.05806

- Blocker: cross-model latent-transfer claims need affine-invariant
  representation comparisons.
- Mechanism: use canonical correlations over subspaces to compare learned
  representations.
- Experiment impact: future analysis for model-family transfer; not required
  for the current CPU gate.
- Role: diagnostic support.

### Model Stitching

Primary source: https://arxiv.org/abs/2106.07682

- Blocker: source and target representations may differ by a small stitch
  layer, not a full retraining problem.
- Mechanism: learn a small transformation between frozen network parts.
- Experiment impact: motivates a future Procrustes/stitch layer before
  relative scoring.
- Role: method inspiration.

### Gromov-Wasserstein Alignment

Primary source: https://proceedings.mlr.press/v97/xu19b.html

- Blocker: anchor sets may not have one-to-one coordinate alignment across
  models or candidate views.
- Mechanism: align relational structures rather than raw coordinates.
- Experiment impact: future anchor-bank matching branch; too heavy for the
  immediate strict-small CPU gate.
- Role: future method inspiration.

### Crosscoders / Shared Dictionaries

Primary source: https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html

- Blocker: an interpretable cross-model packet may need shared sparse atoms
  rather than dense latent vectors.
- Mechanism: learn sparse cross-model features that expose shared and
  model-specific components.
- Experiment impact: future interpretable packet branch; no immediate code
  change.
- Role: inspiration and interpretability framing.

## Design Consequence

The RASP/relative-score packet should be framed as a source-private posterior
expressed in public candidate-anchor coordinates. It is not a universal latent
translator. Its value is systems-oriented: fewer bytes, cleaner candidate-order
equivariance controls, and modest remap robustness over equal-byte scalar
packets.
