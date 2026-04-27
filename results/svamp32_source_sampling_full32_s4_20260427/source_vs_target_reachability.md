# Candidate Pool Reachability Comparison

- date: `2026-04-27`
- status: `candidate_pool_reachability_compared`
- git commit: `7fa3f2b35a3dacd1bb789f7a3b2c563e5bb6d45a`
- baseline: `results/svamp32_target_sampling_full32_s8_20260427/reachability.json`
- candidate: `results/svamp32_source_sampling_full32_s4_20260427/reachability.json`

## Summary

- baseline oracle: `14/32`
- candidate oracle: `10/32`
- candidate minus baseline oracle: `-4`
- new candidate oracle IDs: `5`
- lost baseline oracle IDs: `9`
- new C2C clean residual IDs: `2`
- candidate C2C clean residual in pool: `3/6`

## IDs

- new oracle IDs: `4c84ebf42812703b`, `6e9745b37ab6fc45`, `b1200c32546a34a5`, `c042f0a2949ff8e6`, `de1bf4d142544e5b`
- lost oracle IDs: `013133cdef4f637c`, `0d2b4f7681973e19`, `316b525a4fcbfb89`, `4d780f825bb8541c`, `575d7e83d84c1e67`, `a85be5ec651dac24`, `de2a795ab37694af`, `e26bdbcab1449bbe`, `e3ab8666238a289e`
- new C2C clean residual IDs: `6e9745b37ab6fc45`, `de1bf4d142544e5b`

## Decision

Pass: candidate pool adds C2C-clean residual reachability beyond the baseline pool.

## Command

```bash
scripts/compare_candidate_pool_reachability.py --baseline-reachability results/svamp32_target_sampling_full32_s8_20260427/reachability.json --candidate-reachability results/svamp32_source_sampling_full32_s4_20260427/reachability.json --date 2026-04-27 --output-json results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.json --output-md results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.md
```
