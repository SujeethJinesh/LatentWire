# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.469`
- packet strict-label accuracy: `0.156`
- best source-destroying control accuracy: `0.281`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `159.17 ms`
- full-log p50 E2E delta vs packet: `231.16 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 193.3 | 468.55 | 2051.07 |
| matched_packet | 0.719 | 0.156 | 1.000 | 2.0 | 193.3 | 460.84 | 789.13 |
| matched_byte_text_2 | 0.281 | 0.219 | 1.000 | 2.0 | 192.3 | 451.30 | 1949.95 |
| random_same_byte_packet | 0.031 | 0.000 | 1.000 | 2.0 | 193.3 | 493.24 | 1928.08 |
| deranged_candidate_diag_table | 0.000 | 0.000 | 1.000 | 2.0 | 193.3 | 447.48 | 856.13 |
| query_aware_diag_span | 0.781 | 0.375 | 1.000 | 14.0 | 196.7 | 473.53 | 879.38 |
| structured_json_diag | 0.781 | 0.188 | 1.000 | 21.0 | 198.3 | 443.77 | 846.43 |
| structured_free_text_diag | 0.812 | 0.094 | 1.000 | 17.0 | 196.3 | 474.85 | 714.28 |
| full_hidden_log | 0.406 | 0.000 | 0.406 | 366.5 | 280.8 | 620.01 | 1020.29 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
