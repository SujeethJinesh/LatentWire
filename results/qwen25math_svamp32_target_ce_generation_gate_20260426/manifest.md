# Qwen2.5-Math SVAMP32 Target-CE Cross-Attention Generation Gate

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

- training objective: target continuation next-token CE
- heldout logprob clean IDs: `0/6` matched-only, `4/6` clean control leaks
- mean matched-minus-control clean margin: `-0.194783`
- 64-token generation on clean IDs:
  - matched: `1/6`
  - zero-source: `2/6`
  - shuffled-source: `2/6`
  - target-only prefix: `2/6`
  - slots-only prefix: `2/6`
- decision: fail and prune this low-capacity prefix-emitter family; target-CE
  training does not convert the source cross-attention prefix into source-
  necessary communication.

## Files

| File | SHA256 |
|---|---|
| `smoke.json` | `a8fda429e29ba9ed1ee06285706dc3f0fa95609be2f8829a508b379e689c9517` |
| `smoke.md` | `20be9a45f3be23453c75cd3b15d9a42953c5ae75cc59299e28ab2f19be3e22d9` |
| `generations.jsonl` | `f6cff21a9f981f6482f04b53ef9902cad6ea9ea079b05e5e57c88a03f998acad` |

## Command

See `paper/qwen25math_svamp32_target_ce_generation_gate_20260426.md`.
