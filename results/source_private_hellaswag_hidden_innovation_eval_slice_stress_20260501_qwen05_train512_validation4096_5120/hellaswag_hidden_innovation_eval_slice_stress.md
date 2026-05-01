# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `4096:5120`
- eval rows: `1024`
- selected accuracy: `0.538086`
- best label-copy accuracy: `0.500000`
- delta vs best label-copy: `0.038086`
- CI95 vs best label-copy: `[0.018555, 0.056641]`
- score-only bagged control: `0.497070`
- zero-hidden control: `0.497070`
- wrong-example hidden control: `0.452148`
- candidate-roll hidden control: `0.429688`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
