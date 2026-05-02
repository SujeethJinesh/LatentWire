# HellaSwag Hidden-Innovation Bagged Gate

- pass gate: `False`
- aggregation policy: `mean_zscore_vote_on_score_agreement`
- component models: `12`
- train sample seeds: `4`
- new train sample seeds: `3`
- eval accuracy: `0.535109`
- best label-copy accuracy: `0.498789`
- delta vs best label-copy: `0.036320`
- CI95 vs best label-copy: `[0.009685, 0.064165]`
- score-only bagged control accuracy: `0.497579`
- zero-hidden control accuracy: `0.497579`
- wrong-example hidden control accuracy: `0.475787`
- candidate-roll hidden control accuracy: `0.417676`
- jackknife subbags passing: `2/4`
- jackknife min delta vs best label-copy: `0.019370`
- jackknife min CI95 low vs best label-copy: `-0.003632`

## Interpretation

The previous single-denoiser HellaSwag branch was support-sensitive: one fresh 512-row train sample failed a split row. This gate keeps the same 2B raw / 5B framed source-private packet but treats the source-side denoiser as a small predeclared model bank. Bagging over train-row samples and split seeds is a stability method, not a larger communication channel: the receiver still sees only the selected candidate packet, while score-only and hidden-corruption controls test whether the win requires real source hidden innovation.
