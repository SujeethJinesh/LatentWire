# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `False`
- eval slice: `9216:10042`
- eval rows: `826`
- selected accuracy: `0.547215`
- best label-copy accuracy: `0.498789`
- delta vs best label-copy: `0.048426`
- CI95 vs best label-copy: `[0.020581, 0.076271]`
- score-only bagged control: `0.497579`
- zero-hidden control: `0.497579`
- wrong-example hidden control: `0.469734`
- candidate-roll hidden control: `0.414044`
- jackknife subbags passing: `2/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
