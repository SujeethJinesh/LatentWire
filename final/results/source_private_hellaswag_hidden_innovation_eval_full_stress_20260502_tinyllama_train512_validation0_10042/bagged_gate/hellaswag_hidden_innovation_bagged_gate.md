# HellaSwag Hidden-Innovation Bagged Gate

- pass gate: `True`
- aggregation policy: `mean_zscore`
- component models: `9`
- train sample seeds: `3`
- new train sample seeds: `3`
- eval accuracy: `0.619199`
- best label-copy accuracy: `0.558753`
- delta vs best label-copy: `0.060446`
- CI95 vs best label-copy: `[0.053423, 0.067721]`
- score-only bagged control accuracy: `0.558753`
- zero-hidden control accuracy: `0.558753`
- wrong-example hidden control accuracy: `0.456682`
- candidate-roll hidden control accuracy: `0.366162`
- jackknife subbags passing: `3/3`
- jackknife min delta vs best label-copy: `0.056961`
- jackknife min CI95 low vs best label-copy: `0.049639`

## Interpretation

The previous single-denoiser HellaSwag branch was support-sensitive: one fresh 512-row train sample failed a split row. This gate keeps the same 2B raw / 5B framed source-private packet but treats the source-side denoiser as a small predeclared model bank. Bagging over train-row samples and split seeds is a stability method, not a larger communication channel: the receiver still sees only the selected candidate packet, while score-only and hidden-corruption controls test whether the win requires real source hidden innovation.
