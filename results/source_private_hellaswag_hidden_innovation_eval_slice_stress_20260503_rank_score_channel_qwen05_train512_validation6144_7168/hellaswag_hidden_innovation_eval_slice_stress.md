# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `6144:7168`
- eval rows: `1024`
- selected accuracy: `0.555664`
- best label-copy accuracy: `0.515625`
- delta vs best label-copy: `0.040039`
- CI95 vs best label-copy: `[0.015625, 0.066406]`
- source-rank/index-only bagged control: `0.511719`
- score-only bagged control: `0.511719`
- zero-hidden control: `0.511719`
- wrong-example hidden control: `0.494141`
- candidate-roll hidden control: `0.421875`
- score-channel-roll hidden control: `0.249023`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
