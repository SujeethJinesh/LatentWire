# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.438`
- packet strict-label accuracy: `0.688`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `151.86 ms`
- full-log p50 E2E delta vs packet: `202.49 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.4 | 481.24 | 2166.18 |
| matched_packet | 0.688 | 0.688 | 1.000 | 2.0 | 218.4 | 508.79 | 2130.91 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.4 | 479.91 | 2194.86 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.4 | 480.65 | 2152.65 |
| deranged_candidate_diag_table | 0.188 | 0.188 | 1.000 | 2.0 | 218.4 | 478.39 | 2178.95 |
| query_aware_diag_span | 0.625 | 0.625 | 1.000 | 14.0 | 221.8 | 489.36 | 2197.11 |
| structured_json_diag | 0.500 | 0.500 | 1.000 | 21.0 | 223.4 | 484.96 | 2213.58 |
| structured_free_text_diag | 0.688 | 0.688 | 1.000 | 17.0 | 221.4 | 486.91 | 2179.94 |
| full_hidden_log | 0.438 | 0.438 | 1.000 | 366.5 | 305.9 | 660.65 | 2333.40 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
