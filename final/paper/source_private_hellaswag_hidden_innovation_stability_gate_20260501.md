# HellaSwag Hidden-Innovation Stability Gate

- pass gate: `True`
- split seeds passing: `5/5`
- eval accuracy mean/min/max: `0.502930` / `0.494141` / `0.528320`
- delta vs best label-copy mean/min: `0.040820` / `0.032227`
- min CI95 low vs best label-copy: `0.006836`
- min delta vs zero-hidden: `0.032227`
- selected view counts: `{'score_hidden_residual': 5}`
- unrestricted selector diagnostic pass count: `3/5`
- unrestricted selector min delta vs best label-copy: `-0.090820`

## Interpretation

This gate repeats model selection over independent train/dev splits of the cached 512-row HellaSwag train-hidden slice while evaluating once on the frozen 1024-row validation slice. The promoted method is anchored to the score+hidden-residual view because the unrestricted selector can drift into score-only or hidden-only shortcuts. It does not yet test new train-hidden row samples or full validation.
