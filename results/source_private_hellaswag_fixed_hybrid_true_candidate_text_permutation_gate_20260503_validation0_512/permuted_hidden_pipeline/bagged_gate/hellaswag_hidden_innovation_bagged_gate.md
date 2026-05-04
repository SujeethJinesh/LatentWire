# HellaSwag Hidden-Innovation Bagged Gate

- pass gate: `False`
- aggregation policy: `mean_zscore_vote_on_score_agreement`
- component models: `9`
- train sample seeds: `3`
- new train sample seeds: `2`
- eval accuracy: `0.531250`
- best label-copy accuracy: `0.482422`
- delta vs best label-copy: `0.048828`
- CI95 vs best label-copy: `[0.021484, 0.076221]`
- source-rank/index-only bagged control accuracy: `0.482422`
- score-only bagged control accuracy: `0.482422`
- zero-hidden control accuracy: `0.482422`
- wrong-example hidden control accuracy: `0.414062`
- candidate-roll hidden control accuracy: `0.400391`
- score-channel-roll hidden control accuracy: `0.246094`
- jackknife subbags passing: `2/3`
- jackknife min delta vs best label-copy: `0.019531`
- jackknife min CI95 low vs best label-copy: `-0.005859`
- jackknife min delta vs source-rank/index-only: `0.019531`

## Interpretation

The previous single-denoiser HellaSwag branch was support-sensitive: one fresh 512-row train sample failed a split row. This gate keeps the same 2B raw / 5B framed source-private packet but treats the source-side denoiser as a small predeclared model bank. Bagging over train-row samples and split seeds is a stability method, not a larger communication channel: the receiver still sees only the selected candidate packet, while score-only and hidden-corruption controls test whether the win requires real source hidden innovation.
