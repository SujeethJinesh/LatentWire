# HellaSwag Hidden-Innovation Heldout Eval-Slice Stress

- pass gate: `True`
- eval slice: `7168:8192`
- eval rows: `1024`
- selected accuracy: `0.556641`
- best label-copy accuracy: `0.520508`
- delta vs best label-copy: `0.036133`
- CI95 vs best label-copy: `[0.015625, 0.056641]`
- source-rank/index-only bagged control: `0.519531`
- score-only bagged control: `0.519531`
- zero-hidden control: `0.519531`
- wrong-example hidden control: `0.477539`
- candidate-roll hidden control: `0.454102`
- score-channel-roll hidden control: `0.230469`
- jackknife subbags passing: `3/3`
- packet: `2B` raw / `5B` framed

## Interpretation

This gate freezes the three-sample bagged hidden-innovation method and moves the evaluation off the repeatedly inspected validation-first1024 slice. It is the cheapest Mac-local falsification of slice overfitting before spending full-validation or NVIDIA systems compute.
