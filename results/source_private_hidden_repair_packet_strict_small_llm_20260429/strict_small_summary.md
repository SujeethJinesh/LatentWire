# Source-Private Hidden-Repair Strict-Small Summary

- gate: `source_private_hidden_repair_packet_strict_small_20260429`
- pass gate: `True`
- examples: `160`

| Run | Model | Prompt mode | Pass | Matched | Target-only | Best control | Valid packets | Mean bytes | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.794 | 0.250 | 0.256 | 0.762 | 1.52 | 379.06 |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.256 | 1.000 | 2.00 | 431.40 |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.256 | 0.000 | 0.00 | 286.20 |

Pass rule: Qwen3 and Phi-3 trace_no_hint rows pass on 160 frozen examples, raw_log_no_trace fails, and all source-destroying controls remain within +0.02 of target-only.
