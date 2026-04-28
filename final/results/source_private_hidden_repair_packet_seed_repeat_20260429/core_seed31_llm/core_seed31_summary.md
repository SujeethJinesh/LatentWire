# Source-Private Hidden-Repair Core Seed 31 Summary

- gate: `source_private_hidden_repair_packet_seed_repeat_core_seed31_20260429`
- pass gate: `True`
- examples: `500`
- bootstrap samples: `2000`

| Run | Model | Mode | Pass | Matched | Target | Best control | Valid | Mean bytes | p95 latency ms | Delta target 95% CI | Delta control 95% CI |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.808 | 0.250 | 0.256 | 0.776 | 1.55 | 513.19 | [0.516, 0.602] | [0.506, 0.594] |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.256 | 1.000 | 2.00 | 586.90 | [0.710, 0.786] | [0.704, 0.780] |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.256 | 0.000 | 0.00 | 369.82 | [0.000, 0.000] | [-0.014, 0.000] |

Pass rule: Qwen3 and Phi-3 trace_no_hint rows pass, raw_log_no_trace fails, and paired bootstrap lower bounds remain comfortably above +0.15 for primary rows.
