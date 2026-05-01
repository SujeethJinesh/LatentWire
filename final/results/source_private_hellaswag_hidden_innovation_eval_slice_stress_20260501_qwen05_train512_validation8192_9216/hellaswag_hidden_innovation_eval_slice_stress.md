# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `8192:9216`
- eval rows: `1024`
- selected accuracy: `0.545898`
- best label-copy accuracy: `0.499023`
- delta vs best label-copy: `0.046875`
- CI95 vs best label-copy: `[0.027808, 0.065942]`
- score-only bagged control: `0.499023`
- zero-hidden control: `0.499023`
- wrong-example hidden control: `0.450195`
- candidate-roll hidden control: `0.423828`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
