# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `5120:6144`
- eval rows: `1024`
- selected accuracy: `0.555664`
- best label-copy accuracy: `0.512695`
- delta vs best label-copy: `0.042969`
- CI95 vs best label-copy: `[0.018555, 0.066431]`
- source-rank/index-only bagged control: `0.501953`
- score-only bagged control: `0.501953`
- zero-hidden control: `0.501953`
- wrong-example hidden control: `0.478516`
- candidate-roll hidden control: `0.454102`
- score-channel-roll hidden control: `0.261719`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
