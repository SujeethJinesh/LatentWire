# SVAMP Source Candidate Sidecars

- date: `2026-04-27`
- status: `source_candidate_sidecars_materialized`
- git commit: `afbf022b1e7e1e96c1bd76f72c28e6f538f73abf`

This CPU-only materializer emits source-derived candidate-score sidecars
over target-side candidate values only. It does not add source-only
answers to the decoder pool.

## Surfaces

### live

- target set: `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json`
- n: `3`
- sidecar bytes: `1`
- source final in target pool: `1`
- source-mentioned target-pool hits: `3`
- top label counts: `{'target': 1, 'target_sample_s1': 1, 'target_self_repair': 1}`

## Artifacts

- live sidecar: `results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl`
- holdout sidecar: ``

## Command

```bash
scripts/materialize_svamp_source_candidate_sidecars.py --live-target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json --output-dir results/target_only_sampling_clean3_20260427/source_candidate_sidecars --sidecar-bits 8 --label-prior 0.0 --date 2026-04-27
```

## 2026-04-27 Source-Answer Ablation

The clean3 top-sidecar smoke is killed as a positive-method branch. Full
sidecar selection recovers `14bfbfc94f2c2e7b`, but masking source-final and
verified-answer numeric values removes the win, while a source-final-only
sidecar recovers the same ID.

Counterfactual artifacts are scratch-only under
`.debug/clean3_sidecar_counterfactuals_20260427/`.

- source-answer-masked sidecar sha256:
  `d8fcde23ea05c1f989974925467c769565f11cdfa5fda61c3de852667fb4a7a2`
- source-final-only sidecar sha256:
  `e1fe88457fdd74defd94008fd22178aa355a2d54278fac44671d5e9817b655c3`
