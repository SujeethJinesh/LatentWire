# Reachability Union Summary

- date: `2026-04-27`
- status: `reachability_union_summarized`
- git commit: `4aad04649fd186a985f59fb072e1492a47931d10`
- reference rows: `32`

## Inputs

- `target_direct_s8`: `results/svamp32_target_sampling_full32_s8_20260427/reachability.json`
- `target_brief_s4`: `results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json`

## Summary

- union oracle: `23/32`
- C2C clean residual in union: `6/6`
- C2C teacher-only in union: `8/9`
- source-contrastive clean in union: `3/4`

## IDs

- oracle IDs: `013133cdef4f637c`, `0d2b4f7681973e19`, `14bfbfc94f2c2e7b`, `1d50b408c8f5cd2c`, `2f80204fc9e0896f`, `316b525a4fcbfb89`, `33836927fc9f1a8a`, `3e8a5691f5443495`, `41cce6c6e6bb0058`, `47464cc0b064f172`, `4c84ebf42812703b`, `4d780f825bb8541c`, `4ef979ea2bf931df`, `575d7e83d84c1e67`, `6e9745b37ab6fc45`, `a85be5ec651dac24`, `aee922049c757331`, `c042f0a2949ff8e6`, `de1bf4d142544e5b`, `de2a795ab37694af`, `e26bdbcab1449bbe`, `e3ab8666238a289e`, `f367349ce0604296`
- C2C clean residual IDs: `1d50b408c8f5cd2c`, `3e8a5691f5443495`, `47464cc0b064f172`, `575d7e83d84c1e67`, `6e9745b37ab6fc45`, `de1bf4d142544e5b`
- C2C teacher-only IDs: `14bfbfc94f2c2e7b`, `1d50b408c8f5cd2c`, `3e8a5691f5443495`, `47464cc0b064f172`, `4d780f825bb8541c`, `575d7e83d84c1e67`, `6e9745b37ab6fc45`, `de1bf4d142544e5b`

## Command

```bash
scripts/summarize_reachability_union.py --reachability target_direct_s8=results/svamp32_target_sampling_full32_s8_20260427/reachability.json --reachability target_brief_s4=results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json --date 2026-04-27 --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.json --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.md
```
