# Candidate Pool Reachability Comparison

- date: `2026-04-27`
- status: `candidate_pool_reachability_compared`
- git commit: `4aad04649fd186a985f59fb072e1492a47931d10`
- baseline: `results/svamp32_target_sampling_full32_s8_20260427/reachability.json`
- candidate: `results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json`

## Summary

- baseline oracle: `14/32`
- candidate oracle: `18/32`
- candidate minus baseline oracle: `4`
- new candidate oracle IDs: `9`
- lost baseline oracle IDs: `5`
- new C2C clean residual IDs: `4`
- candidate C2C clean residual in pool: `4/6`

## IDs

- new oracle IDs: `1d50b408c8f5cd2c`, `33836927fc9f1a8a`, `41cce6c6e6bb0058`, `47464cc0b064f172`, `4c84ebf42812703b`, `4ef979ea2bf931df`, `6e9745b37ab6fc45`, `c042f0a2949ff8e6`, `de1bf4d142544e5b`
- lost oracle IDs: `013133cdef4f637c`, `14bfbfc94f2c2e7b`, `3e8a5691f5443495`, `575d7e83d84c1e67`, `a85be5ec651dac24`
- new C2C clean residual IDs: `1d50b408c8f5cd2c`, `47464cc0b064f172`, `6e9745b37ab6fc45`, `de1bf4d142544e5b`

## Decision

Pass: candidate pool adds C2C-clean residual reachability beyond the baseline pool.

## Command

```bash
scripts/compare_candidate_pool_reachability.py --baseline-reachability results/svamp32_target_sampling_full32_s8_20260427/reachability.json --candidate-reachability results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json --date 2026-04-27 --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_vs_target_direct_reachability.json --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_vs_target_direct_reachability.md
```
