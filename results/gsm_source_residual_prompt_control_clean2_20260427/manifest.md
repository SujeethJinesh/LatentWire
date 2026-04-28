# GSM Clean2 Source Residual Prompt-Control Gate

- date: `2026-04-27`
- status: `fails_target_prompt_union_subtraction`
- scale rung: `strict small gate`
- source surface: `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json`
- eval slice: `clean_source_only`
- eval rows: `2`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- source samples per ID: `8`
- target direct samples per ID: `16`
- target brief samples per ID: `16`

## Sample Hashes

- source brief samples SHA256: `30ecb00f7afd76d6c9dc60ece4dd2e30329b0e4e6c44ac656788bc8ae8b0a174`
- target direct samples SHA256: `a4dc2321a9d4295f6c8470f75a5a680a8e3e646bfe88524aeb6d0d04f0484d94`
- target brief samples SHA256: `3c5d2e3a12cf655001585df9d6aaece8aafebe151aec56a8fdef89707f893dea`

## Metrics

- clean source-only IDs: `2`
- source brief S8 oracle: `1/2`
- target direct S16 oracle: `1/2`
- target brief-wrapper S16 oracle: `1/2`
- target direct plus target brief-wrapper union oracle: `1/2`
- source addition beyond target prompt union: `0`
- shared reachable ID: `1deed634dcd7d229`
- unreached ID across all sampled pools: `bc004c17bc99562d`

## Decision

Fail the GSM clean2 source-residual gate. The only source-reached clean ID is
also reached by both target direct sampling and target brief-wrapper sampling.
The source pool adds `0` residual IDs beyond the target prompt union, so this is
not a live source-communication surface.

## Artifacts

- `gsm_clean2_eval.jsonl`
- `gsm_clean2_eval.meta.json`
- `source_brief_samples.jsonl`
- `target_direct_samples.jsonl`
- `target_brief_samples.jsonl`
- `source_brief_reachability.{json,md}`
- `target_direct_reachability.{json,md}`
- `target_brief_reachability.{json,md}`
- `target_prompt_union_reachability.{json,md}`
- `source_vs_target_prompt_union.{json,md}`
