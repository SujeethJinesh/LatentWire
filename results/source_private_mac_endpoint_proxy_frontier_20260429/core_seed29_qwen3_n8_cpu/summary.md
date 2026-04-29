# Source-Private Mac Endpoint-Proxy Frontier

- examples: `8`
- pass gate: `False`
- packet minus target accuracy: `0.000`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `104.95 ms`
- full-log p50 E2E delta vs packet: `290.06 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 218.4 | 963.93 | 3214.48 |
| matched_packet | 0.250 | 0.500 | 2.0 | 216.4 | 992.03 | 1925.97 |
| matched_byte_text_2 | 0.250 | 1.000 | 2.0 | 215.4 | 887.42 | 3187.73 |
| query_aware_diag_span | 0.500 | 0.750 | 14.0 | 219.8 | 876.72 | 3178.83 |
| structured_json_diag | 0.500 | 0.500 | 21.0 | 221.4 | 933.43 | 2233.36 |
| structured_free_text_diag | 0.250 | 0.250 | 17.0 | 219.4 | 935.50 | 1257.49 |
| full_hidden_log | 0.000 | 0.000 | 366.5 | 303.9 | 1096.98 | 2216.04 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
