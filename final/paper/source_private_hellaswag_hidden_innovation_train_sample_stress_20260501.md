# HellaSwag Hidden-Innovation Train-Sample Stress

- pass gate: `False`
- train sample seeds: `2`
- new train sample seeds: `1`
- split rows passing: `5/6`
- sample pass map: `{'1729': True, '2027': False}`
- eval accuracy mean/min/max: `0.483561` / `0.416992` / `0.500000`
- delta vs best label-copy mean/min: `0.021484` / `-0.044922`
- min CI95 low vs best label-copy: `-0.071802`
- min delta vs zero-hidden: `-0.044922`
- selected view counts: `{'score_hidden_residual': 6}`

## Interpretation

This gate redraws the HellaSwag train rows used to fit the anchored source-side hidden-innovation denoiser, while keeping the frozen validation-first1024 readout fixed. It tests whether the score+hidden-residual packet survives beyond the original cached 512-row train-hidden slice. It is still a Mac-local train-sample stress, not a full-validation or NVIDIA serving result.

## Stable-Ridge Diagnostic

A follow-up run constrained the ridge grid to the previously stable family
`{1000, 10000, 100000}`:

- artifact: `results/source_private_hellaswag_hidden_innovation_train_sample_stress_ridge_stable_20260501_qwen05_train512_validation1024/`
- pass gate: `False`
- split rows passing: `5/6`
- sample pass map: `{'1729': True, '2027': False}`
- eval accuracy mean/min/max: `0.488932` / `0.449219` / `0.500000`
- delta vs best label-copy mean/min: `0.026855` / `-0.012695`
- min CI95 low vs best label-copy: `-0.035156`

This reduces the severity of the failing row relative to the wider ridge grid,
but it does not rescue the train-sample gate. The current dense
score+hidden-residual denoiser should therefore be treated as weakened until a
new aggregation rule, larger train sample, or sparse/common-basis packet clears
the same stress.
