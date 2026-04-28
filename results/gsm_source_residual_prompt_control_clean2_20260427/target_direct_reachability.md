# Target Sampling Reachability Audit

- date: `2026-04-27`
- status: `target_sampling_reachability_audited`
- git commit: `f17c315e2170b872fcfed6402f94f2d39b03a01a`
- samples: `32` rows over `2/70` IDs
- sample rows SHA256: `a4dc2321a9d4295f6c8470f75a5a680a8e3e646bfe88524aeb6d0d04f0484d94`

## Summary

- target baseline correct: `4/70`
- sample candidate oracle: `1/70`
- sample oracle gain vs target: `1`
- source-contrastive clean in pool: `1/2`
- C2C clean residual in pool: `0/0`
- C2C teacher-only in pool: `0/0`
- mean unique sampled answers per ID: `9.000`
- duplicate nonempty row fraction: `0.438`

## Reachable IDs

- sample oracle IDs: `1deed634dcd7d229`
- oracle gain IDs: `1deed634dcd7d229`
- C2C clean residual IDs in pool: none
- source-contrastive clean IDs in pool: `1deed634dcd7d229`

## Decision

Fail: target/no-source sampling did not expose enough C2C clean residual reachability; switch candidate-surface generator instead of training a selector.

## Command

```bash
scripts/analyze_target_sampling_reachability.py --samples-jsonl results/gsm_source_residual_prompt_control_clean2_20260427/target_direct_samples.jsonl --base-target-set results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json --date 2026-04-27 --output-json results/gsm_source_residual_prompt_control_clean2_20260427/target_direct_reachability.json --output-md results/gsm_source_residual_prompt_control_clean2_20260427/target_direct_reachability.md
```
