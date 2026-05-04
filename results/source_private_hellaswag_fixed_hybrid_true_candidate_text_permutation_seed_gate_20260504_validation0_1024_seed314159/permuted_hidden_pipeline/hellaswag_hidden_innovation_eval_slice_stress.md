# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `False`
- eval slice: `0:1024`
- eval rows: `1024`
- selected accuracy: `0.523438`
- best label-copy accuracy: `0.461914`
- delta vs best label-copy: `0.061523`
- CI95 vs best label-copy: `[0.041943, 0.083008]`
- source-rank/index-only bagged control: `0.461914`
- score-only bagged control: `0.461914`
- zero-hidden control: `0.461914`
- wrong-example hidden control: `0.419922`
- candidate-roll hidden control: `0.403320`
- score-channel-roll hidden control: `0.245117`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
