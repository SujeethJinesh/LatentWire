# Math-7B SVAMP70 Clean3 Target-Only Sampling

- date: `2026-04-27`
- status: `candidate_pool_generator_positive_selector_answer_leak`
- scale rung: micro smoke
- source surface:
  `results/qwen25math7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json`
- residual clean IDs: `14bfbfc94f2c2e7b`, `a07cd6cc8f1c832e`,
  `d64f6e35083ffe8c`
- target sampler: `Qwen/Qwen3-0.6B`
- samples per example: `16`
- temperature: `0.9`
- top-p: `0.95`

## Candidate-Pool Result

- target-only samples oracle: `1/3`
- combined sampled target-side oracle: `2/3`
- clean IDs in combined target-side pool: `14bfbfc94f2c2e7b`,
  `a07cd6cc8f1c832e`

This is useful as target/no-source candidate-pool generation, not as
communication evidence.

## Selector Controls

| Profile Mode | Status | Matched Clean Correct | Source-Necessary Clean IDs |
|---|---|---:|---|
| `full` | `top_sidecar_selector_passes_smoke` | 2/3 | `14bfbfc94f2c2e7b`, `a07cd6cc8f1c832e` |
| `answer_only` | `top_sidecar_selector_passes_smoke` | 2/3 | `14bfbfc94f2c2e7b`, `a07cd6cc8f1c832e` |
| `answer_masked` | `top_sidecar_selector_fails_smoke` | 0/3 | none |

Decision: prune the current source-candidate score selector as a communication
method. The apparent source signal is fully explained by final/verified source
answer values.

## Artifacts

- `clean_source_only_eval.jsonl`
- `target_only_samples.jsonl`
- `target_only_samples.md`
- `sampled_clean3_target_set.json`
- `sampled_clean3_target_set.md`
- `sampled_clean3_headroom.json`
- `sampled_clean3_headroom.md`
- `source_candidate_sidecars_full/manifest.md`
- `source_candidate_sidecars_answer_only/manifest.md`
- `source_candidate_sidecars_answer_masked/manifest.md`
- `top_selector_full.md`
- `top_selector_answer_only.md`
- `top_selector_answer_masked.md`
