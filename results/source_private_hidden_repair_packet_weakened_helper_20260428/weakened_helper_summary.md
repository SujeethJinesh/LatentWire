# Source-Private Hidden-Repair Weakened-Helper Summary

- gate: `source_private_hidden_repair_packet_weakened_helper_20260428`
- pass gate: `True`

| Run | Model | Prompt mode | Pass | Matched | Target-only | Best control | Valid packets | Mean bytes | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_log_only | Qwen/Qwen3-0.6B | log_only | `true` | 0.984 | 0.250 | 0.250 | 0.984 | 1.97 | 366.37 |
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.781 | 0.250 | 0.250 | 0.734 | 1.47 | 293.33 |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.250 | 0.000 | 0.00 | 277.67 |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 2.00 | 426.51 |

Pass rule: At least two trace_no_hint model rows pass while raw_log_no_trace fails as a source-signal destruction control.
