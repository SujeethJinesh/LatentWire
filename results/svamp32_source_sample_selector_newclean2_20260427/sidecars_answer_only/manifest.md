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
- profile mode: `answer_only`
- source final in target pool: `21`
- source-mentioned target-pool hits: `22`
- top label counts: `{'source_sample_s1': 1, 't2t': 3, 'target': 25, 'target_sample_s2': 1, 'target_sample_s4': 1, 'target_sample_s5': 1}`

## Artifacts

- live sidecar: `results/svamp32_source_sample_selector_newclean2_20260427/sidecars_answer_only/live_candidate_sidecars.jsonl`
- holdout sidecar: ``

## Command

```bash
scripts/materialize_svamp_source_candidate_sidecars.py --live-target-set results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json --output-dir results/svamp32_source_sample_selector_newclean2_20260427/sidecars_answer_only --sidecar-bits 8 --label-prior 0.0 --profile-mode answer_only --date 2026-04-27
```
