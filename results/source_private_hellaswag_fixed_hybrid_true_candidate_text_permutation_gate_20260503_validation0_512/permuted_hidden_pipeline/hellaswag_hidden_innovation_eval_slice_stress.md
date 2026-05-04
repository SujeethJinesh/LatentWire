# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `False`
- eval slice: `0:512`
- eval rows: `512`
- selected accuracy: `0.531250`
- best label-copy accuracy: `0.482422`
- delta vs best label-copy: `0.048828`
- CI95 vs best label-copy: `[0.021484, 0.076221]`
- source-rank/index-only bagged control: `0.482422`
- score-only bagged control: `0.482422`
- zero-hidden control: `0.482422`
- wrong-example hidden control: `0.414062`
- candidate-roll hidden control: `0.400391`
- score-channel-roll hidden control: `0.246094`
- jackknife subbags passing: `2/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
