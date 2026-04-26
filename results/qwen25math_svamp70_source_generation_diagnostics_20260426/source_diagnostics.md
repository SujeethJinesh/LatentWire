# Source Generation Diagnostics

- date: `2026-04-26`
- status: `source_generation_diagnostics_collected`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- prompt mode: `direct`
- eval file: `results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl`
- output JSONL: `results/qwen25math_svamp70_source_generation_diagnostics_20260426/source_diagnostics.jsonl`
- output JSONL sha256: `b17755be3db764f6130830cc516b18b6e4fadce7a78de36d20f10dd8c84c69b2`
- correct: `13/70`

This sidecar artifact records source-only greedy generation confidence signals: per-token chosen logprob, entropy, top-1 probability, and top-1/top-2 logit margin.
