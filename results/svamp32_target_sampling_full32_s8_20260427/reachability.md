# Target Sampling Reachability Audit

- date: `2026-04-27`
- status: `target_sampling_reachability_audited`
- git commit: `5feb0e05568c65cc44ff1aadec4973f52b0ccf82`
- samples: `256` rows over `32/32` IDs
- sample rows SHA256: `650c19fb283d908b1e11af54f19dd0ec8a48e204d38bdd359120d6e4bbd354e5`

## Summary

- target baseline correct: `8/32`
- sample candidate oracle: `14/32`
- sample oracle gain vs target: `7`
- source-contrastive clean in pool: `2/4`
- C2C clean residual in pool: `2/6`
- C2C teacher-only in pool: `4/9`
- mean unique sampled answers per ID: `3.344`
- duplicate nonempty row fraction: `0.582`

## Reachable IDs

- sample oracle IDs: `013133cdef4f637c`, `0d2b4f7681973e19`, `14bfbfc94f2c2e7b`, `2f80204fc9e0896f`, `316b525a4fcbfb89`, `3e8a5691f5443495`, `4d780f825bb8541c`, `575d7e83d84c1e67`, `a85be5ec651dac24`, `aee922049c757331`, `de2a795ab37694af`, `e26bdbcab1449bbe`, `e3ab8666238a289e`, `f367349ce0604296`
- oracle gain IDs: `14bfbfc94f2c2e7b`, `3e8a5691f5443495`, `4d780f825bb8541c`, `575d7e83d84c1e67`, `aee922049c757331`, `e26bdbcab1449bbe`, `e3ab8666238a289e`
- C2C clean residual IDs in pool: `3e8a5691f5443495`, `575d7e83d84c1e67`
- source-contrastive clean IDs in pool: `14bfbfc94f2c2e7b`, `4d780f825bb8541c`

## Decision

Pass: C2C clean residual target reachability matches the clean6 floor; next gate should test a source-derived selector only on reachable clean IDs.

## Command

```bash
scripts/analyze_target_sampling_reachability.py --samples-jsonl results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json --date 2026-04-27 --output-json results/svamp32_target_sampling_full32_s8_20260427/reachability.json --output-md results/svamp32_target_sampling_full32_s8_20260427/reachability.md
```
