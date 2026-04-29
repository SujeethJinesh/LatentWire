# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.531`
- packet strict-label accuracy: `0.219`
- best source-destroying control accuracy: `0.312`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `185.83 ms`
- full-log p50 E2E delta vs packet: `367.21 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.312 | 0.250 | 1.000 | 0.0 | 193.3 | 469.91 | 2037.74 |
| matched_packet | 0.844 | 0.219 | 1.000 | 2.0 | 193.3 | 461.16 | 735.95 |
| matched_byte_text_2 | 0.312 | 0.188 | 1.000 | 2.0 | 192.3 | 447.83 | 2000.58 |
| random_same_byte_packet | 0.094 | 0.000 | 1.000 | 2.0 | 193.3 | 464.65 | 1441.50 |
| deranged_candidate_diag_table | 0.000 | 0.000 | 1.000 | 2.0 | 193.3 | 463.79 | 857.37 |
| query_aware_diag_span | 0.812 | 0.312 | 1.000 | 14.0 | 196.7 | 484.34 | 866.51 |
| structured_json_diag | 0.750 | 0.125 | 1.000 | 21.0 | 198.3 | 485.50 | 829.15 |
| structured_free_text_diag | 0.875 | 0.156 | 1.000 | 17.0 | 196.3 | 473.65 | 838.50 |
| full_hidden_log | 0.344 | 0.000 | 0.344 | 373.5 | 283.6 | 646.99 | 1103.16 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
