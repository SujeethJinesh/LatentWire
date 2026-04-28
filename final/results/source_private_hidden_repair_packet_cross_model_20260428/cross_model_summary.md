# Source-Private Hidden-Repair Cross-Model Summary

- gate: `source_private_hidden_repair_packet_cross_model_20260428`
- pass gate: `True`
- passing models: `3/4`
- non-Qwen passing models: `1`

| Run | Model | Family | Pass | Matched | Target-only | Best control | Valid packets | Mean bytes | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen25_0_5b_helper | Qwen/Qwen2.5-0.5B-Instruct | qwen2.5 | `true` | 0.984 | 0.250 | 0.250 | 0.984 | 1.97 | 330.85 |
| qwen3_0_6b_helper | Qwen/Qwen3-0.6B | qwen3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 2.00 | 312.52 |
| phi3_mini_helper | microsoft/Phi-3-mini-4k-instruct | phi3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 2.00 | 511.35 |
| tinyllama_1_1b_helper | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama | `false` | 0.250 | 0.250 | 0.250 | 0.000 | 0.00 | 561.73 |

Pass rule: At least two capable instruction models pass, at least one non-Qwen model passes, and source-destroying controls remain within +0.02 of target-only.
