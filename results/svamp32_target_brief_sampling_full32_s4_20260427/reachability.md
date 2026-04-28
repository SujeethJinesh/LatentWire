# Target Sampling Reachability Audit

- date: `2026-04-27`
- status: `target_sampling_reachability_audited`
- git commit: `4aad04649fd186a985f59fb072e1492a47931d10`
- samples: `128` rows over `32/32` IDs
- sample rows SHA256: `0f2b6eacb6f53bf7850528f713d142a2beb5e8fb355a850756659c9dea489830`

## Summary

- target baseline correct: `8/32`
- sample candidate oracle: `18/32`
- sample oracle gain vs target: `12`
- source-contrastive clean in pool: `2/4`
- C2C clean residual in pool: `4/6`
- C2C teacher-only in pool: `5/9`
- mean unique sampled answers per ID: `2.250`
- duplicate nonempty row fraction: `0.438`

## Reachable IDs

- sample oracle IDs: `0d2b4f7681973e19`, `1d50b408c8f5cd2c`, `2f80204fc9e0896f`, `316b525a4fcbfb89`, `33836927fc9f1a8a`, `41cce6c6e6bb0058`, `47464cc0b064f172`, `4c84ebf42812703b`, `4d780f825bb8541c`, `4ef979ea2bf931df`, `6e9745b37ab6fc45`, `aee922049c757331`, `c042f0a2949ff8e6`, `de1bf4d142544e5b`, `de2a795ab37694af`, `e26bdbcab1449bbe`, `e3ab8666238a289e`, `f367349ce0604296`
- oracle gain IDs: `1d50b408c8f5cd2c`, `33836927fc9f1a8a`, `41cce6c6e6bb0058`, `47464cc0b064f172`, `4c84ebf42812703b`, `4d780f825bb8541c`, `4ef979ea2bf931df`, `6e9745b37ab6fc45`, `aee922049c757331`, `de1bf4d142544e5b`, `e26bdbcab1449bbe`, `e3ab8666238a289e`
- C2C clean residual IDs in pool: `1d50b408c8f5cd2c`, `47464cc0b064f172`, `6e9745b37ab6fc45`, `de1bf4d142544e5b`
- source-contrastive clean IDs in pool: `41cce6c6e6bb0058`, `4d780f825bb8541c`

## Decision

Strong pass: C2C clean residual target reachability is large enough to justify a strict source-derived selector or connector gate.

## Command

```bash
scripts/analyze_target_sampling_reachability.py --samples-jsonl results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.jsonl --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json --date 2026-04-27 --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.md
```
