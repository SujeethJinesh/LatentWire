# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.500`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `182.10 ms`
- full-log p50 E2E delta vs packet: `209.69 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 193.4 | 536.93 | 2058.90 |
| matched_packet | 0.750 | 1.000 | 2.0 | 193.4 | 455.86 | 775.27 |
| matched_byte_text_2 | 0.250 | 1.000 | 2.0 | 192.4 | 432.15 | 2017.36 |
| query_aware_diag_span | 0.812 | 1.000 | 14.0 | 196.8 | 442.71 | 871.62 |
| structured_json_diag | 0.812 | 1.000 | 21.0 | 198.4 | 479.47 | 726.50 |
| structured_free_text_diag | 0.750 | 1.000 | 17.0 | 196.4 | 442.03 | 743.62 |
| full_hidden_log | 0.312 | 0.312 | 366.5 | 280.9 | 637.97 | 984.96 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
