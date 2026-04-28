# Candidate Pool Reachability Comparison

- date: `2026-04-27`
- status: `candidate_pool_reachability_compared`
- git commit: `4aad04649fd186a985f59fb072e1492a47931d10`
- baseline: `results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json`
- candidate: `results/svamp32_source_sampling_full32_s4_20260427/reachability.json`

## Summary

- baseline oracle: `18/32`
- candidate oracle: `10/32`
- candidate minus baseline oracle: `-8`
- new candidate oracle IDs: `3`
- lost baseline oracle IDs: `11`
- new C2C clean residual IDs: `1`
- candidate C2C clean residual in pool: `3/6`

## IDs

- new oracle IDs: `14bfbfc94f2c2e7b`, `3e8a5691f5443495`, `b1200c32546a34a5`
- lost oracle IDs: `0d2b4f7681973e19`, `1d50b408c8f5cd2c`, `316b525a4fcbfb89`, `33836927fc9f1a8a`, `41cce6c6e6bb0058`, `47464cc0b064f172`, `4d780f825bb8541c`, `4ef979ea2bf931df`, `de2a795ab37694af`, `e26bdbcab1449bbe`, `e3ab8666238a289e`
- new C2C clean residual IDs: `3e8a5691f5443495`

## Decision

Pass: candidate pool adds C2C-clean residual reachability beyond the baseline pool.

## Command

```bash
scripts/compare_candidate_pool_reachability.py --baseline-reachability results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json --candidate-reachability results/svamp32_source_sampling_full32_s4_20260427/reachability.json --date 2026-04-27 --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/source_vs_target_brief_reachability.json --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/source_vs_target_brief_reachability.md
```
