# ThoughtFlow-FP8 Real-Trace Retention Gate

Status: **WEAKENED on real generated traces; do not advance without a better protected-token signal.**

This gate reuses saved generation traces already in the LatentWire repo.
It is not model accuracy evidence, not KV-cache telemetry, and not a GPU systems result.

- traces: 74
- keep fraction: 0.20

## Overall

| Policy | Traces | Avg tokens | Keep rate | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|---:|---:|---:|
| longflow_like | 74 | 44.5 | 0.211 | 1.000 | 0.941 | 0.484 |
| rkv_like | 74 | 44.5 | 0.211 | 1.000 | 0.142 | 0.433 |
| thin_kv_like | 74 | 44.5 | 0.211 | 1.000 | 0.529 | 0.557 |
| thoughtflow | 74 | 44.5 | 0.211 | 1.000 | 0.941 | 0.484 |

## By Dataset

### `prompt_control_20260419/qwen_gsm10_target_alone_chat_thinking_false`

| Policy | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|
| longflow_like | 1.000 | 1.000 | 0.281 |
| rkv_like | 1.000 | 0.353 | 0.224 |
| thin_kv_like | 1.000 | 0.353 | 0.408 |
| thoughtflow | 1.000 | 1.000 | 0.281 |

### `surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_alone`

| Policy | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|
| longflow_like | 1.000 | 0.942 | 0.729 |
| rkv_like | 1.000 | 0.050 | 0.758 |
| thin_kv_like | 1.000 | 0.730 | 0.841 |
| thoughtflow | 1.000 | 0.942 | 0.729 |

### `surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/text_to_text`

| Policy | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|
| longflow_like | 1.000 | 0.922 | 0.302 |
| rkv_like | 1.000 | 0.168 | 0.172 |
| thin_kv_like | 1.000 | 0.383 | 0.319 |
| thoughtflow | 1.000 | 0.922 | 0.302 |

## Decision

The branch remains useful only if the protected-token policy keeps phase/control
markers under fixed nominal-budget accounting on real traces.
If advanced, the next gate must move from text heuristics to real KV/cache telemetry and accuracy or perplexity impact.
