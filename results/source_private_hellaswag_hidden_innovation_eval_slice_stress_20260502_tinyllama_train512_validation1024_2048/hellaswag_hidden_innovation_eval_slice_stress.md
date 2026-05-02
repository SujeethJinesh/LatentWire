# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `1024:2048`
- eval rows: `1024`
- selected accuracy: `0.501953`
- best label-copy accuracy: `0.450195`
- delta vs best label-copy: `0.051758`
- CI95 vs best label-copy: `[0.025391, 0.075708]`
- score-only bagged control: `0.446289`
- zero-hidden control: `0.446289`
- wrong-example hidden control: `0.403320`
- candidate-roll hidden control: `0.358398`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
