# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- prompt style: `terse`
- pass gate: `False`
- packet minus target accuracy: `0.000`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `283.42 ms`
- full-log p50 E2E delta vs packet: `-1551.16 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 169.4 | 533.04 | 2966.23 |
| matched_packet | 0.250 | 1.000 | 2.0 | 169.4 | 643.72 | 2898.26 |
| matched_byte_text_2 | 0.250 | 1.000 | 2.0 | 168.4 | 643.53 | 2848.02 |
| query_aware_diag_span | 0.500 | 0.875 | 14.0 | 172.8 | 587.02 | 2880.28 |
| structured_json_diag | 0.625 | 1.000 | 21.0 | 174.4 | 536.30 | 2868.30 |
| structured_free_text_diag | 0.688 | 1.000 | 17.0 | 172.4 | 652.91 | 2863.45 |
| full_hidden_log | 0.375 | 1.000 | 366.5 | 256.9 | 927.14 | 1347.10 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
