# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `3072:4096`
- eval rows: `1024`
- selected accuracy: `0.531250`
- best label-copy accuracy: `0.484375`
- delta vs best label-copy: `0.046875`
- CI95 vs best label-copy: `[0.026367, 0.067383]`
- source-rank/index-only bagged control: `0.480469`
- score-only bagged control: `0.480469`
- zero-hidden control: `0.480469`
- wrong-example hidden control: `0.455078`
- candidate-roll hidden control: `0.405273`
- score-channel-roll hidden control: `0.241211`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
