# SVAMP32 Target Brief Wrapper Sampling Full32 S4

- date: `2026-04-27`
- status: `target_prompt_wrapper_baseline_promoted`
- scale rung: `smoke`
- model: `Qwen/Qwen3-0.6B`
- prompt mode: `source_reasoning`
- source reasoning mode: `brief_analysis`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- samples per ID: `4`
- sample rows: `128`
- sample SHA256: `0f2b6eacb6f53bf7850528f713d142a2beb5e8fb355a850756659c9dea489830`

## Metrics

- target brief-wrapper oracle: `18/32`
- target direct S8 oracle: `14/32`
- source brief S4 oracle: `10/32`
- target direct plus target brief-wrapper union oracle: `23/32`
- target direct plus target brief-wrapper C2C-clean residual union: `6/6`
- target direct plus target brief-wrapper plus source brief union oracle: `24/32`
- source addition beyond target-prior union: `+1` oracle ID
- source addition beyond target-prior union C2C-clean residual: `0`

## Decision

Promote target brief-wrapper sampling to a mandatory target-prior baseline. The
target prompt wrapper alone reaches all C2C-clean residual IDs when unioned with
target direct sampling, so the prior source-sampling branch has no C2C-clean
source-specific residual surface left.

## Artifacts

- `target_brief_samples.jsonl`
- `target_brief_samples.json`
- `target_brief_samples.md`
- `reachability.json`
- `reachability.md`
- `target_brief_vs_target_direct_reachability.json`
- `target_brief_vs_target_direct_reachability.md`
- `source_vs_target_brief_reachability.json`
- `source_vs_target_brief_reachability.md`
- `target_prior_union_reachability.json`
- `target_prior_union_reachability.md`
- `target_prior_plus_source_union_reachability.json`
- `target_prior_plus_source_union_reachability.md`
- `source_addition_vs_target_prior_union.json`
- `source_addition_vs_target_prior_union.md`
