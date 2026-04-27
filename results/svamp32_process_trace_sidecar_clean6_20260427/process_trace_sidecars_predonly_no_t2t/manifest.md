# SVAMP Process-Trace Sidecars

- date: `2026-04-27`
- status: `process_trace_sidecars_materialized`
- git commit: `5d3c2d4103ca2ec9a9eca1cdb4cd74163cedecb6`
- target set: `results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json`
- n: `6`
- sidecar bytes: `32`
- max ngram: `2`
- exclude labels: `['t2t']`
- prediction only: `True`
- top label counts: `{'target_sample_s0': 1, 'target_sample_s1': 3, 'target_sample_s2': 2}`
- top value in unmasked source numbers: `5`
- zero margin rows: `1`
- margin mean: `0.059867`

## Collapse Telemetry

- feature count: `259`
- std min: `0.006054`
- std mean: `0.020084`
- effective rank: `31.270460`
- zero vectors: `0`

## Command

```bash
scripts/materialize_svamp_process_trace_sidecars.py --target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json --output-dir results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars_predonly_no_t2t --sidecar-bits 256 --max-ngram 2 --exclude-label t2t --prediction-only --date 2026-04-27
```
