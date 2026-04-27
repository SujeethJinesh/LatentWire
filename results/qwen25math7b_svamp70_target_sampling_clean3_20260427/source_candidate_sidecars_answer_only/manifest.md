# SVAMP Source Candidate Sidecars

- date: `2026-04-27`
- status: `source_candidate_sidecars_materialized`
- git commit: `6a09be6d62e3804af408cad178b8274b965b7da6`

This CPU-only materializer emits source-derived candidate-score sidecars
over target-side candidate values only. It does not add source-only
answers to the decoder pool.

## Surfaces

### live

- target set: `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_target_set.json`
- n: `3`
- sidecar bytes: `1`
- profile mode: `answer_only`
- source final in target pool: `2`
- source-mentioned target-pool hits: `2`
- top label counts: `{'target': 1, 'target_sample_s3': 1, 'text': 1}`

## Artifacts

- live sidecar: `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/source_candidate_sidecars_answer_only/live_candidate_sidecars.jsonl`
- holdout sidecar: ``

## Command

```bash
scripts/materialize_svamp_source_candidate_sidecars.py --live-target-set results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_target_set.json --output-dir results/qwen25math7b_svamp70_target_sampling_clean3_20260427/source_candidate_sidecars_answer_only --sidecar-bits 8 --label-prior 0.0 --profile-mode answer_only --date 2026-04-27
```
