# Candidate Pool Reachability Comparison

- date: `2026-04-27`
- status: `candidate_pool_reachability_compared`
- git commit: `4aad04649fd186a985f59fb072e1492a47931d10`
- baseline: `results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.json`
- candidate: `results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_plus_source_union_reachability.json`

## Summary

- baseline oracle: `23/32`
- candidate oracle: `24/32`
- candidate minus baseline oracle: `1`
- new candidate oracle IDs: `1`
- lost baseline oracle IDs: `0`
- new C2C clean residual IDs: `0`
- candidate C2C clean residual in pool: `6/6`

## IDs

- new oracle IDs: `b1200c32546a34a5`
- lost oracle IDs: none
- new C2C clean residual IDs: none

## Decision

Weak pass: candidate pool improves total oracle but not C2C-clean residual reachability.

## Command

```bash
scripts/compare_candidate_pool_reachability.py --baseline-reachability results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.json --candidate-reachability results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_plus_source_union_reachability.json --date 2026-04-27 --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/source_addition_vs_target_prior_union.json --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/source_addition_vs_target_prior_union.md
```
