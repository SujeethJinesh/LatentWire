# HellaSwag Hidden-Innovation Bagged Gate

- pass gate: `False`
- aggregation policy: `mean_zscore`
- component models: `9`
- train sample seeds: `3`
- new train sample seeds: `2`
- eval accuracy: `0.539952`
- best label-copy accuracy: `0.498789`
- delta vs best label-copy: `0.041162`
- CI95 vs best label-copy: `[0.014528, 0.069007]`
- source-rank/index-only bagged control accuracy: `0.497579`
- score-only bagged control accuracy: `0.497579`
- zero-hidden control accuracy: `0.497579`
- wrong-example hidden control accuracy: `0.484262`
- candidate-roll hidden control accuracy: `0.433414`
- score-channel-roll hidden control accuracy: `0.266344`
- jackknife subbags passing: `1/3`
- jackknife min delta vs best label-copy: `0.020581`
- jackknife min CI95 low vs best label-copy: `-0.007264`
- jackknife min delta vs source-rank/index-only: `0.021792`

## Interpretation

The previous single-denoiser HellaSwag branch was support-sensitive: one fresh 512-row train sample failed a split row. This gate keeps the same 2B raw / 5B framed source-private packet but treats the source-side denoiser as a small predeclared model bank. Bagging over train-row samples and split seeds is a stability method, not a larger communication channel: the receiver still sees only the selected candidate packet, while score-only and hidden-corruption controls test whether the win requires real source hidden innovation.
