# Source-Private Mac Endpoint-Proxy Frontier

- examples: `8`
- pass gate: `True`
- packet minus target accuracy: `0.500`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `181.14 ms`
- full-log p50 E2E delta vs packet: `-38.86 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 1.000 | 0.0 | 218.4 | 494.56 | 2056.19 |
| matched_packet | 0.750 | 1.000 | 2.0 | 216.4 | 488.07 | 1412.60 |
| matched_byte_text_2 | 0.250 | 1.000 | 2.0 | 215.4 | 560.62 | 2120.74 |
| query_aware_diag_span | 0.750 | 1.000 | 14.0 | 219.8 | 509.98 | 2075.62 |
| structured_json_diag | 1.000 | 1.000 | 21.0 | 221.4 | 569.65 | 1400.49 |
| structured_free_text_diag | 1.000 | 1.000 | 17.0 | 219.4 | 544.88 | 860.78 |
| full_hidden_log | 1.000 | 1.000 | 366.5 | 303.9 | 669.21 | 1373.74 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
