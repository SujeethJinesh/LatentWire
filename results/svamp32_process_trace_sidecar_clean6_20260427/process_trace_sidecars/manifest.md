# SVAMP Process-Trace Sidecars

- date: `2026-04-27`
- status: `process_trace_sidecars_materialized`
- git commit: `5d3c2d4103ca2ec9a9eca1cdb4cd74163cedecb6`
- target set: `results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json`
- n: `6`
- sidecar bytes: `32`
- max ngram: `2`
- top label counts: `{'target': 6}`
- top value in unmasked source numbers: `4`
- zero margin rows: `6`
- margin mean: `0.000000`

## Collapse Telemetry

- feature count: `259`
- std min: `0.006250`
- std mean: `0.021025`
- effective rank: `31.782150`
- zero vectors: `0`

## Command

```bash
scripts/materialize_svamp_process_trace_sidecars.py --target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json --output-dir results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars --sidecar-bits 256 --max-ngram 2 --date 2026-04-27
```
