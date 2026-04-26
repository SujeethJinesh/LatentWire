# Source Generation Diagnostics

- date: `2026-04-26`
- status: `source_generation_diagnostics_collected`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- prompt mode: `direct`
- eval file: `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl`
- output JSONL: `results/qwen25math_svamp70_source_generation_diagnostics_20260426/holdout_source_diagnostics.jsonl`
- output JSONL sha256: `2fc5226940ea4fc743324534bb51c938829910810619040f78afea2c905ecb0e`
- correct: `8/70`

This sidecar artifact records source-only greedy generation confidence signals: per-token chosen logprob, entropy, top-1 probability, and top-1/top-2 logit margin.
