# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- pass gate: `True`
- packet minus target accuracy: `0.438`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `165.38 ms`
- full-log p50 E2E delta vs packet: `-636.29 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 218.4 | 494.57 | 2095.74 |
| matched_packet | 0.688 | 1.000 | 2.0 | 216.4 | 525.69 | 2057.43 |
| matched_byte_text_2 | 0.250 | 1.000 | 2.0 | 215.4 | 544.85 | 2138.97 |
| query_aware_diag_span | 0.812 | 1.000 | 14.0 | 219.8 | 454.93 | 2075.83 |
| structured_json_diag | 1.000 | 1.000 | 21.0 | 221.4 | 502.76 | 1544.93 |
| structured_free_text_diag | 1.000 | 1.000 | 17.0 | 219.4 | 566.71 | 825.41 |
| full_hidden_log | 1.000 | 1.000 | 366.5 | 303.9 | 691.07 | 1421.13 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
