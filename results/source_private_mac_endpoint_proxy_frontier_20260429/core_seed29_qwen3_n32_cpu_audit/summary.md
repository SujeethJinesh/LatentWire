# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.469`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `163.45 ms`
- full-log p50 E2E delta vs packet: `225.84 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 193.3 | 475.74 | 2014.03 |
| matched_packet | 0.719 | 1.000 | 2.0 | 193.3 | 483.99 | 815.54 |
| matched_byte_text_2 | 0.281 | 1.000 | 2.0 | 192.3 | 483.05 | 1992.72 |
| query_aware_diag_span | 0.781 | 1.000 | 14.0 | 196.7 | 523.94 | 889.18 |
| structured_json_diag | 0.781 | 1.000 | 21.0 | 198.3 | 476.89 | 843.79 |
| structured_free_text_diag | 0.812 | 1.000 | 17.0 | 196.3 | 474.88 | 830.36 |
| full_hidden_log | 0.406 | 0.406 | 366.5 | 280.8 | 647.44 | 1041.37 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
