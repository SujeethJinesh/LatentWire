# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.500`
- packet strict-label accuracy: `0.062`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `278.22 ms`
- full-log p50 E2E delta vs packet: `430.06 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 193.4 | 753.78 | 2917.72 |
| matched_packet | 0.750 | 0.062 | 1.000 | 2.0 | 193.4 | 727.75 | 1067.07 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 192.4 | 809.82 | 2917.24 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 193.4 | 649.15 | 1761.50 |
| deranged_candidate_diag_table | 0.000 | 0.000 | 1.000 | 2.0 | 193.4 | 822.51 | 1278.86 |
| query_aware_diag_span | 0.812 | 0.375 | 1.000 | 14.0 | 196.8 | 703.48 | 1303.56 |
| structured_json_diag | 0.812 | 0.188 | 1.000 | 21.0 | 198.4 | 785.58 | 1058.23 |
| structured_free_text_diag | 0.750 | 0.062 | 1.000 | 17.0 | 196.4 | 794.62 | 1110.23 |
| full_hidden_log | 0.312 | 0.000 | 0.312 | 366.5 | 280.9 | 1005.97 | 1497.13 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
