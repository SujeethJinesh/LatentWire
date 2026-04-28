# Source-Private Hidden-Repair Holdout Seed 32 Summary

- gate: `source_private_hidden_repair_packet_seed_repeat_holdout_seed32_20260429`
- pass gate: `True`
- examples: `500`
- bootstrap samples: `2000`

| Run | Model | Mode | Pass | Matched | Target | Best control | Valid | Mean bytes | p95 latency ms | Delta target 95% CI | Delta control 95% CI |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.924 | 0.250 | 0.252 | 0.860 | 1.72 | 388.70 | [0.634, 0.716] | [0.632, 0.712] |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.252 | 1.000 | 2.00 | 558.23 | [0.710, 0.786] | [0.710, 0.786] |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.252 | 0.000 | 0.00 | 643.89 | [0.000, 0.000] | [-0.006, 0.000] |

Pass rule: Qwen3 and Phi-3 trace_no_hint rows pass, raw_log_no_trace fails, and paired bootstrap lower bounds remain comfortably above +0.15 for primary rows.
