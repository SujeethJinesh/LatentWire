# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `False`
- eval slice: `0:10042`
- eval rows: `10042`
- selected accuracy: `0.619199`
- best label-copy accuracy: `0.558753`
- delta vs best label-copy: `0.060446`
- CI95 vs best label-copy: `[0.053423, 0.067721]`
- score-only bagged control: `0.558753`
- zero-hidden control: `0.558753`
- wrong-example hidden control: `0.456682`
- candidate-roll hidden control: `0.366162`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
