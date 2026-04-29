# Source-Private Mac Endpoint-Proxy Frontier

- examples: `64`
- prompt style: `audit`
- pass gate: `False`
- packet minus target accuracy: `0.500`
- packet strict-label accuracy: `0.172`
- best source-destroying control accuracy: `0.203`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `260.17 ms`
- full-log p50 E2E delta vs packet: `227.44 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 193.3 | 493.80 | 2210.86 |
| matched_packet | 0.750 | 0.172 | 0.781 | 2.0 | 193.3 | 466.19 | 905.63 |
| matched_byte_text_2 | 0.203 | 0.203 | 0.828 | 2.0 | 192.3 | 466.84 | 2174.12 |
| random_same_byte_packet | 0.000 | 0.000 | 0.953 | 2.0 | 193.3 | 471.93 | 2057.21 |
| deranged_candidate_diag_table | 0.000 | 0.000 | 0.781 | 2.0 | 193.3 | 500.95 | 905.21 |
| query_aware_diag_span | 0.891 | 0.406 | 0.938 | 14.0 | 196.7 | 471.45 | 1145.21 |
| structured_json_diag | 0.812 | 0.188 | 0.859 | 21.0 | 198.3 | 489.34 | 875.63 |
| structured_free_text_diag | 0.844 | 0.094 | 0.844 | 17.0 | 196.3 | 483.06 | 837.21 |
| full_hidden_log | 0.375 | 0.000 | 0.375 | 366.5 | 280.8 | 726.36 | 1133.07 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
