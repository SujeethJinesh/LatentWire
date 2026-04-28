# Candidate Pool Reachability Comparison

- date: `2026-04-27`
- status: `candidate_pool_reachability_compared`
- git commit: `f17c315e2170b872fcfed6402f94f2d39b03a01a`
- baseline: `results/gsm_source_residual_prompt_control_clean2_20260427/target_prompt_union_reachability.json`
- candidate: `results/gsm_source_residual_prompt_control_clean2_20260427/source_brief_reachability.json`

## Summary

- baseline oracle: `1/70`
- candidate oracle: `1/70`
- candidate minus baseline oracle: `0`
- new candidate oracle IDs: `0`
- lost baseline oracle IDs: `0`
- new C2C clean residual IDs: `0`
- candidate C2C clean residual in pool: `0/0`

## IDs

- new oracle IDs: none
- lost oracle IDs: none
- new C2C clean residual IDs: none

## Decision

Fail: candidate pool does not improve oracle reachability over the baseline pool.

## Command

```bash
scripts/compare_candidate_pool_reachability.py --baseline-reachability results/gsm_source_residual_prompt_control_clean2_20260427/target_prompt_union_reachability.json --candidate-reachability results/gsm_source_residual_prompt_control_clean2_20260427/source_brief_reachability.json --date 2026-04-27 --output-json results/gsm_source_residual_prompt_control_clean2_20260427/source_vs_target_prompt_union.json --output-md results/gsm_source_residual_prompt_control_clean2_20260427/source_vs_target_prompt_union.md
```
