# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- pass gate: `True`
- packet minus target accuracy: `0.438`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `190.66 ms`
- full-log p50 E2E delta vs packet: `-623.02 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 218.4 | 502.89 | 1978.80 |
| matched_packet | 0.688 | 1.000 | 2.0 | 216.4 | 454.12 | 1989.76 |
| matched_byte_text_2 | 0.250 | 1.000 | 2.0 | 215.4 | 446.69 | 2018.19 |
| query_aware_diag_span | 0.750 | 1.000 | 14.0 | 219.9 | 474.14 | 1993.11 |
| structured_json_diag | 0.938 | 1.000 | 21.0 | 221.4 | 532.22 | 855.90 |
| structured_free_text_diag | 0.938 | 1.000 | 17.0 | 219.4 | 467.56 | 864.95 |
| full_hidden_log | 1.000 | 1.000 | 373.5 | 306.8 | 644.78 | 1366.74 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
