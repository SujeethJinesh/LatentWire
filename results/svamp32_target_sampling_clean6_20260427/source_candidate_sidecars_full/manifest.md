# SVAMP Source Candidate Sidecars

- date: `2026-04-27`
- status: `source_candidate_sidecars_materialized`
- git commit: `614f2a2d5482b3d5007b005b4f7df86c23c5479d`

This CPU-only materializer emits source-derived candidate-score sidecars
over target-side candidate values only. It does not add source-only
answers to the decoder pool.

## Surfaces

### live

- target set: `results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json`
- n: `6`
- sidecar bytes: `1`
- profile mode: `full`
- source final in target pool: `6`
- source-mentioned target-pool hits: `6`
- top label counts: `{'target': 6}`

## Artifacts

- live sidecar: `results/svamp32_target_sampling_clean6_20260427/source_candidate_sidecars_full/live_candidate_sidecars.jsonl`
- holdout sidecar: ``

## Command

```bash
scripts/materialize_svamp_source_candidate_sidecars.py --live-target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json --output-dir results/svamp32_target_sampling_clean6_20260427/source_candidate_sidecars_full --sidecar-bits 8 --label-prior 0.0 --profile-mode full --date 2026-04-27
```
