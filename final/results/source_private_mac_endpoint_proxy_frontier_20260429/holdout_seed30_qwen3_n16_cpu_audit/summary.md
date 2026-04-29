# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.562`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `150.18 ms`
- full-log p50 E2E delta vs packet: `310.41 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.312 | 1.000 | 0.0 | 193.4 | 442.41 | 2037.13 |
| matched_packet | 0.875 | 1.000 | 2.0 | 193.4 | 438.21 | 761.64 |
| matched_byte_text_2 | 0.312 | 1.000 | 2.0 | 192.4 | 411.69 | 2017.14 |
| query_aware_diag_span | 0.812 | 1.000 | 14.0 | 196.9 | 454.38 | 844.19 |
| structured_json_diag | 0.750 | 1.000 | 21.0 | 198.4 | 444.77 | 815.54 |
| structured_free_text_diag | 0.875 | 1.000 | 17.0 | 196.4 | 441.04 | 749.77 |
| full_hidden_log | 0.375 | 0.375 | 373.5 | 283.8 | 588.39 | 1072.05 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
