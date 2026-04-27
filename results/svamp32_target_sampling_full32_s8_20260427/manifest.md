# SVAMP32 Full32 Target Sampling S8

- date: `2026-04-27`
- status: `target_reachability_pass_selector_surface_not_expanded`
- git commit at run time: `5feb0e05568c65cc44ff1aadec4973f52b0ccf82`
- scale rung: `strict small gate`
- model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- samples per ID: `8`
- sample rows: `256`
- sample SHA256: `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5`

## Metrics

- target baseline: `8/32`
- raw sample candidate oracle: `14/32`
- raw sample oracle gain vs target: `7`
- merged target-side oracle with text relay plus samples: `18/32`
- merged oracle gain: `10`
- C2C clean residual in pool: `2/6`
- C2C teacher-only in pool: `4/9`
- source-contrastive clean in pool: `2/4`
- mean unique sampled answers per ID: `3.344`
- duplicate nonempty row fraction: `0.582`

## Decision

The target/no-source receiver pool is substantially larger, but it does not
expand the C2C-clean residual surface beyond the two clean IDs already reached
by the prior clean6 gate. Treat this as target-prior headroom, not source
communication evidence.

## Artifacts

- `target_only_samples.jsonl`
- `target_only_samples.json`
- `target_only_samples.md`
- `reachability.json`
- `reachability.md`
- `headroom.json`
- `headroom.md`
- `no_source_surface/source_contrastive_target_set.json`
- `no_source_surface/source_contrastive_target_set.md`
- `no_source_surface/manifest.json`
- `no_source_surface/manifest.md`
