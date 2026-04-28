# Qwen2.5-7B SVAMP70 Clean7 Target Brief S8 Control

- date: `2026-04-27`
- status: `partial_target_prompt_prior_explanation`
- scale rung: `strict small gate`
- source surface: `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json`
- eval slice: `clean_source_only`
- eval rows: `7`
- target model: `Qwen/Qwen3-0.6B`
- samples per ID: `8`
- prompt mode: `source_reasoning`
- source reasoning mode: `brief_analysis`
- sample SHA256: `d4ee408a152d2499300e48d9958ee6a5739276eefd1fe8920714d5c3f19c05bf`

## Metrics

- prior clean source-only IDs: `7`
- target brief-wrapper S8 oracle: `4/7`
- numeric coverage: `56/56`
- target-prior explained IDs:
  - `3c5aeb08941dbb6d`
  - `ce08a3a269bf0151`
  - `de1bf4d142544e5b`
  - `e099e405e8d1a66b`
- target-prior-unexplained IDs:
  - `33836927fc9f1a8a`
  - `4c84ebf42812703b`
  - `d64f6e35083ffe8c`

## Decision

This surface is not promotable as a source-communication branch yet. A no-source
target prompt-wrapper recovers `4/7` clean source-only IDs, so those IDs are
target-prior reachability rather than source communication. The remaining `3/7`
IDs are only residual candidates and still need answer-masked, answer-only,
zero-source, shuffled-source, random sidecar, and matched-budget target controls
before any connector training.

## Artifacts

- `clean7_eval.jsonl`
- `clean7_eval.meta.json`
- `target_brief_samples.jsonl`
- `target_brief_samples.json`
- `target_brief_samples.md`
- `reachability.json`
- `reachability.md`
