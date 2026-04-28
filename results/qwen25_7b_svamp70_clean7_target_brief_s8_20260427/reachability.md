# Target Sampling Reachability Audit

- date: `2026-04-27`
- status: `target_sampling_reachability_audited`
- git commit: `f17c315e2170b872fcfed6402f94f2d39b03a01a`
- samples: `56` rows over `7/70` IDs
- sample rows SHA256: `d4ee408a152d2499300e48d9958ee6a5739276eefd1fe8920714d5c3f19c05bf`

## Summary

- target baseline correct: `21/70`
- sample candidate oracle: `4/70`
- sample oracle gain vs target: `4`
- source-contrastive clean in pool: `4/7`
- C2C clean residual in pool: `0/0`
- C2C teacher-only in pool: `0/0`
- mean unique sampled answers per ID: `2.857`
- duplicate nonempty row fraction: `0.643`

## Reachable IDs

- sample oracle IDs: `3c5aeb08941dbb6d`, `ce08a3a269bf0151`, `de1bf4d142544e5b`, `e099e405e8d1a66b`
- oracle gain IDs: `3c5aeb08941dbb6d`, `ce08a3a269bf0151`, `de1bf4d142544e5b`, `e099e405e8d1a66b`
- C2C clean residual IDs in pool: none
- source-contrastive clean IDs in pool: `3c5aeb08941dbb6d`, `ce08a3a269bf0151`, `de1bf4d142544e5b`, `e099e405e8d1a66b`

## Decision

Fail: target/no-source sampling did not expose enough C2C clean residual reachability; switch candidate-surface generator instead of training a selector.

## Command

```bash
scripts/analyze_target_sampling_reachability.py --samples-jsonl results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.jsonl --base-target-set results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json --date 2026-04-27 --output-json results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/reachability.json --output-md results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/reachability.md
```
