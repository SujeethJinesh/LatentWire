# Source-Private Hidden-Repair Holdout-Families Summary

- gate: `source_private_hidden_repair_packet_holdout_families_20260429`
- pass gate: `True`
- examples: `500`
- bootstrap samples: `2000`

| Run | Model | Mode | Pass | Matched | Target | Best control | Valid | Mean bytes | p95 latency ms | Delta target 95% CI | Delta control 95% CI |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.922 | 0.250 | 0.258 | 0.864 | 1.73 | 429.15 | [0.632, 0.712] | [0.622, 0.706] |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.258 | 1.000 | 2.00 | 602.46 | [0.710, 0.788] | [0.702, 0.778] |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.258 | 0.000 | 0.00 | 620.12 | [0.000, 0.000] | [-0.016, -0.002] |

Pass rule: Qwen3 and Phi-3 trace_no_hint rows pass, raw_log_no_trace fails, and paired bootstrap lower bounds remain comfortably above +0.15 for primary rows.
