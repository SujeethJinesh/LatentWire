# SVAMP Source Candidate Sidecars

- date: `2026-04-27`
- status: `source_candidate_sidecars_materialized`
- git commit: `ef4886ed050f7ee21c8746a42e673a86c5fc1fe1`

This CPU-only materializer emits source-derived candidate-score sidecars
over target-side candidate values only. It does not add source-only
answers to the decoder pool.

## Surfaces

### live

- target set: `results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json`
- n: `32`
- sidecar bytes: `1`
- profile mode: `answer_masked`
- source final in target pool: `0`
- source-mentioned target-pool hits: `25`
- top label counts: `{'t2t': 1, 'target': 30, 'target_sample_s2': 1}`

## Artifacts

- live sidecar: `results/svamp32_source_sample_selector_newclean2_20260427/sidecars_answer_masked/live_candidate_sidecars.jsonl`
- holdout sidecar: ``

## Command

```bash
scripts/materialize_svamp_source_candidate_sidecars.py --live-target-set results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json --output-dir results/svamp32_source_sample_selector_newclean2_20260427/sidecars_answer_masked --sidecar-bits 8 --label-prior 0.0 --profile-mode answer_masked --date 2026-04-27
```
