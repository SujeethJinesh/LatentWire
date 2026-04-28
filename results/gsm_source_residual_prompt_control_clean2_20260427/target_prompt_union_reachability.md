# Reachability Union Summary

- date: `2026-04-27`
- status: `reachability_union_summarized`
- git commit: `f17c315e2170b872fcfed6402f94f2d39b03a01a`
- reference rows: `70`

## Inputs

- `target_direct`: `results/gsm_source_residual_prompt_control_clean2_20260427/target_direct_reachability.json`
- `target_brief`: `results/gsm_source_residual_prompt_control_clean2_20260427/target_brief_reachability.json`

## Summary

- union oracle: `1/70`
- C2C clean residual in union: `0/0`
- C2C teacher-only in union: `0/0`
- source-contrastive clean in union: `1/2`

## IDs

- oracle IDs: `1deed634dcd7d229`
- C2C clean residual IDs: none
- C2C teacher-only IDs: none

## Command

```bash
scripts/summarize_reachability_union.py --reachability target_direct=results/gsm_source_residual_prompt_control_clean2_20260427/target_direct_reachability.json --reachability target_brief=results/gsm_source_residual_prompt_control_clean2_20260427/target_brief_reachability.json --date 2026-04-27 --output-json results/gsm_source_residual_prompt_control_clean2_20260427/target_prompt_union_reachability.json --output-md results/gsm_source_residual_prompt_control_clean2_20260427/target_prompt_union_reachability.md
```
