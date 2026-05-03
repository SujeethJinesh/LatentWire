# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `1024:2048`
- eval rows: `1024`
- selected accuracy: `0.454102`
- best label-copy accuracy: `0.414062`
- delta vs best label-copy: `0.040039`
- CI95 vs best label-copy: `[0.018555, 0.061523]`
- source-rank/index-only bagged control: `0.409180`
- score-only bagged control: `0.409180`
- zero-hidden control: `0.409180`
- wrong-example hidden control: `0.386719`
- candidate-roll hidden control: `0.372070`
- score-channel-roll hidden control: `0.272461`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
