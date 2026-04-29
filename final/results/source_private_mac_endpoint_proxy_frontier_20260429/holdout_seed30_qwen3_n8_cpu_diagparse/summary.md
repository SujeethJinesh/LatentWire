# Source-Private Mac Endpoint-Proxy Frontier

- examples: `8`
- pass gate: `True`
- packet minus target accuracy: `0.500`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `279.45 ms`
- full-log p50 E2E delta vs packet: `5.91 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 218.5 | 532.78 | 2109.60 |
| matched_packet | 0.750 | 1.000 | 2.0 | 216.5 | 481.03 | 1434.94 |
| matched_byte_text_2 | 0.250 | 1.000 | 2.0 | 215.5 | 493.89 | 2038.79 |
| query_aware_diag_span | 0.625 | 1.000 | 14.0 | 220.0 | 558.65 | 2038.39 |
| structured_json_diag | 0.875 | 1.000 | 21.0 | 221.5 | 532.49 | 861.35 |
| structured_free_text_diag | 0.875 | 1.000 | 17.0 | 219.5 | 498.28 | 933.20 |
| full_hidden_log | 1.000 | 1.000 | 373.5 | 306.9 | 760.49 | 1440.84 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
