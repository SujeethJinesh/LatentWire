# HellaSwag Hidden-Innovation Bagged Gate

- pass gate: `True`
- aggregation policy: `mean_zscore`
- component models: `9`
- train sample seeds: `3`
- new train sample seeds: `2`
- eval accuracy: `0.555664`
- best label-copy accuracy: `0.515625`
- delta vs best label-copy: `0.040039`
- CI95 vs best label-copy: `[0.015625, 0.066406]`
- source-rank/index-only bagged control accuracy: `0.511719`
- score-only bagged control accuracy: `0.511719`
- zero-hidden control accuracy: `0.511719`
- wrong-example hidden control accuracy: `0.494141`
- candidate-roll hidden control accuracy: `0.421875`
- score-channel-roll hidden control accuracy: `0.249023`
- jackknife subbags passing: `3/3`
- jackknife min delta vs best label-copy: `0.030273`
- jackknife min CI95 low vs best label-copy: `0.003906`
- jackknife min delta vs source-rank/index-only: `0.034180`

## Interpretation

The previous single-denoiser HellaSwag branch was support-sensitive: one fresh 512-row train sample failed a split row. This gate keeps the same 2B raw / 5B framed source-private packet but treats the source-side denoiser as a small predeclared model bank. Bagging over train-row samples and split seeds is a stability method, not a larger communication channel: the receiver still sees only the selected candidate packet, while score-only and hidden-corruption controls test whether the win requires real source hidden innovation.
