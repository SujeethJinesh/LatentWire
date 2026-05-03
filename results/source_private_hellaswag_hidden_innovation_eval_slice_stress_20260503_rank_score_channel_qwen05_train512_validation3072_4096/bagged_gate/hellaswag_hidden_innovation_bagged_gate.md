# HellaSwag Hidden-Innovation Bagged Gate

- pass gate: `True`
- aggregation policy: `mean_zscore`
- component models: `9`
- train sample seeds: `3`
- new train sample seeds: `2`
- eval accuracy: `0.531250`
- best label-copy accuracy: `0.484375`
- delta vs best label-copy: `0.046875`
- CI95 vs best label-copy: `[0.026367, 0.067383]`
- source-rank/index-only bagged control accuracy: `0.480469`
- score-only bagged control accuracy: `0.480469`
- zero-hidden control accuracy: `0.480469`
- wrong-example hidden control accuracy: `0.455078`
- candidate-roll hidden control accuracy: `0.405273`
- score-channel-roll hidden control accuracy: `0.241211`
- jackknife subbags passing: `3/3`
- jackknife min delta vs best label-copy: `0.039062`
- jackknife min CI95 low vs best label-copy: `0.016602`
- jackknife min delta vs source-rank/index-only: `0.042969`

## Interpretation

The previous single-denoiser HellaSwag branch was support-sensitive: one fresh 512-row train sample failed a split row. This gate keeps the same 2B raw / 5B framed source-private packet but treats the source-side denoiser as a small predeclared model bank. Bagging over train-row samples and split seeds is a stability method, not a larger communication channel: the receiver still sees only the selected candidate packet, while score-only and hidden-corruption controls test whether the win requires real source hidden innovation.
