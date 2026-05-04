# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `False`
- eval slice: `0:1024`
- eval rows: `1024`
- selected accuracy: `0.463867`
- best label-copy accuracy: `0.416992`
- delta vs best label-copy: `0.046875`
- CI95 vs best label-copy: `[0.024390, 0.066431]`
- source-rank/index-only bagged control: `0.409180`
- score-only bagged control: `0.409180`
- zero-hidden control: `0.409180`
- wrong-example hidden control: `0.382812`
- candidate-roll hidden control: `0.353516`
- score-channel-roll hidden control: `0.264648`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
