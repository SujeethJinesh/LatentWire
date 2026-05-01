# HellaSwag Hidden-Innovation Multi-Slice Stress

- pass gate: `True`
- validation slices: `0:1024`, `1024:2048`, `2048:3072`
- total eval rows: `3072`
- weighted selected accuracy: `0.482096`
- weighted best label-copy accuracy: `0.441081`
- weighted score-only bagged control accuracy: `0.434896`
- weighted zero-hidden control accuracy: `0.434896`
- min delta vs best label-copy: `0.034180`
- min CI95 low vs best label-copy: `0.011719`
- min delta vs score-only bagged: `0.044922`
- min score-only CI95 low: `0.027344`
- min delta vs zero-hidden: `0.044922`
- max wrong-example hidden control accuracy: `0.437500`
- max candidate-roll hidden control accuracy: `0.389648`
- jackknife slices passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This is the strongest Mac-local HellaSwag evidence so far. The same
three-train-sample bagged hidden-innovation packet now clears three contiguous
HellaSwag validation blocks without changing the packet format, train samples,
split seeds, ridge family, or aggregation rule. The target still receives only
a fixed candidate/confidence record; source text, source KV, raw hidden
vectors, and raw source scores are not transmitted.

The result directly addresses the most important reviewer risk from the
single-slice gate: a positive result on validation-first1024 or validation
`1024:2048` could have been slice luck. Across `3072` rows, every slice beats
best label-copy, score-only bagging, and zero-hidden controls by at least the
predeclared `0.02` margin with positive paired uncertainty. Wrong-example and
candidate-roll hidden controls remain below best label-copy, and every
two-train-sample jackknife subbag passes inside each slice.

## Reviewer Boundary

This upgrades HellaSwag to a stronger headline candidate, not a finished ICLR
claim. The remaining ICLR gates are: remaining-slice or full-validation stress,
one strict cross-family falsification pair, and native NVIDIA/vLLM/SGLang
systems rows. The result should be framed as fixed-byte source-private hidden
innovation transfer with public candidate side information, not as universal
latent-language translation.

## Lay Summary

We took a model's private internal hint, squeezed it into a tiny `2B` message,
and checked whether another model answered HellaSwag questions better. The
important part is that the tiny message still helped across three separate
chunks of the benchmark, while fake or broken messages stopped helping.
