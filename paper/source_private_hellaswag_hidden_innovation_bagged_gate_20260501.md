# HellaSwag Hidden-Innovation Bagged Gate

- pass gate: `True`
- aggregation policy: `mean_zscore`
- component models: `9`
- train sample seeds: `3`
- new train sample seeds: `2`
- eval accuracy: `0.512695`
- best label-copy accuracy: `0.463867`
- delta vs best label-copy: `0.048828`
- CI95 vs best label-copy: `[0.028320, 0.070825]`
- score-only bagged control accuracy: `0.461914`
- zero-hidden control accuracy: `0.461914`
- wrong-example hidden control accuracy: `0.437500`
- candidate-roll hidden control accuracy: `0.389648`
- jackknife subbags passing: `3/3`
- jackknife min delta vs best label-copy: `0.032227`
- jackknife min CI95 low vs best label-copy: `0.010229`
- jackknife min delta vs score-only bagged: `0.033203`
- packet: `2B` raw / `5B` framed

## Interpretation

The previous single-denoiser HellaSwag branch was support-sensitive: one fresh
512-row train sample failed a split row. This gate keeps the same fixed-byte
source-private packet but treats the source-side denoiser as a small
predeclared model bank. Bagging over train-row samples and split seeds is a
stability method, not a larger communication channel: the receiver still sees
only the selected candidate packet, while score-only and hidden-corruption
controls test whether the win requires real source hidden innovation.

## Reviewer Boundary

This is a positive robustness rescue, not a final ICLR headline. It directly
addresses the failed 2027 train-sample stress without increasing bytes, but it
now clears a third train sample (`2039`) and all `2-of-3` train-sample
jackknife subbags. The remaining HellaSwag promotion gate is therefore no
longer fresh-support stability; it is frozen larger/full-validation stress.
The next systems blocker also remains unchanged: native vLLM/SGLang
measurements on NVIDIA hardware are required before claiming throughput,
memory, or latency superiority over internal-state communication baselines.
