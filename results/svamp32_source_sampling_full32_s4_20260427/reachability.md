# Target Sampling Reachability Audit

- date: `2026-04-27`
- status: `target_sampling_reachability_audited`
- git commit: `7fa3f2b35a3dacd1bb789f7a3b2c563e5bb6d45a`
- samples: `128` rows over `32/32` IDs
- sample rows SHA256: `3520951745cd92b53ff8bbe01d3b4b9d47d5985027ab3b5abc4e9d0b247fb18b`

## Summary

- target baseline correct: `8/32`
- sample candidate oracle: `10/32`
- sample oracle gain vs target: `7`
- source-contrastive clean in pool: `1/4`
- C2C clean residual in pool: `3/6`
- C2C teacher-only in pool: `4/9`
- mean unique sampled answers per ID: `3.406`
- duplicate nonempty row fraction: `0.148`

## Reachable IDs

- sample oracle IDs: `14bfbfc94f2c2e7b`, `2f80204fc9e0896f`, `3e8a5691f5443495`, `4c84ebf42812703b`, `6e9745b37ab6fc45`, `aee922049c757331`, `b1200c32546a34a5`, `c042f0a2949ff8e6`, `de1bf4d142544e5b`, `f367349ce0604296`
- oracle gain IDs: `14bfbfc94f2c2e7b`, `3e8a5691f5443495`, `4c84ebf42812703b`, `6e9745b37ab6fc45`, `aee922049c757331`, `b1200c32546a34a5`, `de1bf4d142544e5b`
- C2C clean residual IDs in pool: `3e8a5691f5443495`, `6e9745b37ab6fc45`, `de1bf4d142544e5b`
- source-contrastive clean IDs in pool: `14bfbfc94f2c2e7b`

## Decision

Strong pass: C2C clean residual target reachability is large enough to justify a strict source-derived selector or connector gate.

## Command

```bash
scripts/analyze_target_sampling_reachability.py --samples-jsonl results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json --date 2026-04-27 --output-json results/svamp32_source_sampling_full32_s4_20260427/reachability.json --output-md results/svamp32_source_sampling_full32_s4_20260427/reachability.md
```
