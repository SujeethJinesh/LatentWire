# Qwen2.5-Math SVAMP32 Source-Control Contrastive Cross-Attention

- date: `2026-04-26`
- status: `source_cross_attention_logprob_fails_gate`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval file:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- target JSONL:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl`
- teacher JSONL:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl`
- target set:
  `results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json`

## Result

- matched-only clean IDs: `0/6`
- clean control leaks: `4/6`
- mean matched-minus-control clean margin: `-0.382854`

## Files

| File | SHA256 |
|---|---|
| `smoke.json` | `2f6e2a38f6b1685b7a571f30f53dd1587fa03532560aa7fe04f4f515a15cb4a1` |
| `smoke.md` | `e009da82d298ade4e65aa0c76709f67f77066654485170fe05a27ca3e9918637` |

## Command

See `paper/qwen25math_svamp32_source_cross_attention_contrastive_20260426.md`.
