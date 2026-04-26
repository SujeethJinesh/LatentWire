# Qwen2.5-Math -> Qwen3 SVAMP32 Source Soft-Prefix Logprob Manifest

- date: `2026-04-26`
- status: `source_soft_prefix_logprob_fails_gate`
- git commit at run time: `09aed7128f76449ceb377ed0f83e3cf7969b6770`
- scale-up rung: `micro smoke / strict-small pre-generation diagnostic`

## Files

- `source_only_numeric_meanlogprob_smoke.json`
  - sha256: `f89c0a8759a94574de9e5a52eb50af800fab352c13550efdf1660f85d33778c9`
- `source_only_numeric_meanlogprob_smoke.md`
  - sha256: `cd67391b2c87a449a2096fb04942a8232cc7f8035bcbb3aa989b4e1aeae94169`
- `sha256.txt`

## Inputs

- eval file:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- target JSONL:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl`
- C2C teacher JSONL:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl`
- target-set JSON:
  `results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json`

## Decision

Kill the pooled-summary soft-prefix branch before generation. It recovers only
`1/6` clean matched-only IDs and has `4/6` clean control leaks under the
calibrated source-only numeric mean-logprob smoke.
