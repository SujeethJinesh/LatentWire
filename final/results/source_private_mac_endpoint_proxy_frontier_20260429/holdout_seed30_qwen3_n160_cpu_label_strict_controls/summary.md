# Source-Private Mac Endpoint-Proxy Frontier

- examples: `160`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.438`
- packet strict-label accuracy: `0.675`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `183.54 ms`
- full-log p50 E2E delta vs packet: `249.72 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.3 | 556.26 | 2106.41 |
| matched_packet | 0.688 | 0.675 | 1.000 | 2.0 | 218.3 | 547.21 | 2110.92 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.3 | 545.18 | 2107.43 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.3 | 547.07 | 2090.60 |
| deranged_candidate_diag_table | 0.244 | 0.244 | 1.000 | 2.0 | 218.3 | 549.36 | 2108.00 |
| query_aware_diag_span | 0.688 | 0.688 | 1.000 | 14.0 | 221.6 | 558.93 | 2117.99 |
| structured_json_diag | 0.594 | 0.594 | 1.000 | 21.0 | 223.3 | 552.50 | 2123.32 |
| structured_free_text_diag | 0.719 | 0.719 | 1.000 | 17.0 | 221.3 | 557.69 | 2127.65 |
| full_hidden_log | 0.531 | 0.525 | 1.000 | 373.5 | 308.5 | 730.76 | 2360.64 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
