# SVAMP32 Target-Only Clean6 Sampling Gate Manifest

- date: `2026-04-27`
- status: `generator_pass_selector_fail`
- scale rung: `strict small gate`
- source surface: `Qwen/Qwen2.5-Math-1.5B -> Qwen/Qwen3-0.6B`
- target/no-source model: `Qwen/Qwen3-0.6B`
- clean slice source: `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`
- decoder target set: `results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json`

## Metrics

- clean residual IDs: `6`
- target-only sampled candidates: `16` per ID
- target-only sample numeric coverage: `96/96`
- target-only candidate oracle: `2/6`
- reachable clean IDs: `3e8a5691f5443495`, `575d7e83d84c1e67`
- selector `full`: matched correct `0/6`, source-necessary clean `0`
- selector `answer_only`: matched correct `0/6`, source-necessary clean `0`
- selector `answer_masked`: matched correct `0/6`, accepted `0`, source-necessary clean `0`

## Decision

The target/no-source generator passes the headroom floor, but the current
source-candidate sidecar fails the communication gate. Treat this as reusable
candidate-pool evidence only. The next source-side method must use a non-answer
process/latent signal with collapse telemetry and the same source-destroying
controls.

## Key Artifacts

- `clean6_eval.jsonl`
- `clean6_eval.meta.json`
- `target_only_samples.jsonl`
- `target_only_samples.json`
- `target_only_samples.md`
- `sampled_clean6_target_set.json`
- `sampled_clean6_target_set.md`
- `extend_manifest.json`
- `source_candidate_sidecars_full/manifest.md`
- `source_candidate_sidecars_answer_only/manifest.md`
- `source_candidate_sidecars_answer_masked/manifest.md`
- `top_selector_full.md`
- `top_selector_answer_only.md`
- `top_selector_answer_masked.md`
