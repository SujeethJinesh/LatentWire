# Source-Private Systems Summary

- gate: `source_private_systems_summary_20260428`
- status: systems evidence from existing artifacts
- claim: At the far-left rate point, 2-byte source-private packets recover hidden diagnostic evidence while matched-byte text relays stay at target floor, using about 183-187x fewer bytes than full hidden-log relay.

## Headline

- packet accuracy minimum: `1.000`
- matched-byte text accuracy maximum: `0.250`
- full hidden-log relay accuracy minimum: `1.000`
- packet bytes: `2.00`
- full hidden-log bytes: `366.45-373.50`
- compression vs full hidden-log relay: `183.2x-186.7x`
- compression vs full diagnostic text: `7.0x`

## Deterministic Rate Rows

| Surface | Interface | Kind | Accuracy | Bytes | Tokens | p50 latency ms | Compression vs full log |
|---|---|---|---:|---:|---:|---:|---:|
| core seed29 | target-only | no source | 0.250 | 0.00 | 0.00 | 0.001 | - |
| core seed29 | 2-byte diagnostic packet | method | 1.000 | 2.00 | 1.00 | 0.002 | 183.2x |
| core seed29 | 2-byte hidden-log truncation | matched-byte text | 0.250 | 2.00 | 1.00 | 0.002 | 183.2x |
| core seed29 | 2-byte JSON relay | matched-byte text | 0.250 | 2.00 | 1.00 | 0.004 | 183.2x |
| core seed29 | 2-byte free-text relay | matched-byte text | 0.250 | 2.00 | 1.00 | 0.002 | 183.2x |
| core seed29 | full hidden-log relay | oracle text relay | 1.000 | 366.45 | 33.87 | 0.006 | 1.0x |
| core seed29 | full diagnostic text | oracle diagnostic text | 1.000 | 14.00 | 1.00 | 0.002 | 26.2x |
| holdout seed30 | target-only | no source | 0.250 | 0.00 | 0.00 | 0.001 | - |
| holdout seed30 | 2-byte diagnostic packet | method | 1.000 | 2.00 | 1.00 | 0.002 | 186.7x |
| holdout seed30 | 2-byte hidden-log truncation | matched-byte text | 0.250 | 2.00 | 1.00 | 0.002 | 186.7x |
| holdout seed30 | 2-byte JSON relay | matched-byte text | 0.250 | 2.00 | 1.00 | 0.004 | 186.7x |
| holdout seed30 | 2-byte free-text relay | matched-byte text | 0.250 | 2.00 | 1.00 | 0.002 | 186.7x |
| holdout seed30 | full hidden-log relay | oracle text relay | 1.000 | 373.50 | 34.75 | 0.006 | 1.0x |
| holdout seed30 | full diagnostic text | oracle diagnostic text | 1.000 | 14.00 | 1.00 | 0.002 | 26.7x |

## Model Packet Rows

| Model | Run | Accuracy | Target | Best control | Valid | Bytes | Tokens | p50 ms | p95 ms | Packets/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | qwen3_trace_no_hint | 0.808 | 0.250 | 0.252 | 0.776 | 1.55 | 2.86 | 293.4 | 522.9 | 3.41 |
| microsoft/Phi-3-mini-4k-instruct | phi3_trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | 2.00 | 3.00 | 421.9 | 516.4 | 2.37 |

## Target Decoder Rows

| Surface | Accuracy | Target | Best control | Valid | Packet bytes | p50 ms | Generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| core seed29 qwen3 n16 mps | 0.688 | 0.250 | 0.250 | 1.000 | 2.00 | 1266.5 | 13.00 |
| holdout seed30 qwen3 n32 mps | 0.750 | 0.250 | 0.281 | 1.000 | 2.00 | 1314.7 | 13.00 |
| core seed29 qwen3 n64 cpu | 0.656 | 0.250 | 0.250 | 1.000 | 2.00 | 2182.0 | 13.00 |
| holdout seed30 qwen3 n64 cpu | 0.719 | 0.250 | 0.266 | 1.000 | 2.00 | 2237.0 | 13.00 |

## Caveat

Local p50 latency is single-request wall-clock timing, not server throughput or TTFT. Future endpoint runs should add TTFT and streaming decode timing.
