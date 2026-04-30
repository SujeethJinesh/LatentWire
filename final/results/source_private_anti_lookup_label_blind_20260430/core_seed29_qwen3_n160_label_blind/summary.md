# Source-Private Mac Endpoint-Proxy Frontier

- examples: `160`
- prompt style: `label_strict`
- candidate view: `label_blind`
- pass gate: `False`
- packet minus target accuracy: `0.000`
- packet strict-label accuracy: `0.250`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `156.67 ms`
- full-log p50 E2E delta vs packet: `168.58 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 209.6 | 502.83 | 746.01 |
| matched_packet | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 489.97 | 736.98 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 208.6 | 500.83 | 736.14 |
| random_same_byte_packet | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 494.93 | 749.93 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 485.43 | 739.89 |
| query_aware_diag_span | 0.250 | 0.250 | 1.000 | 14.0 | 212.9 | 510.46 | 746.61 |
| structured_json_diag | 0.250 | 0.250 | 1.000 | 21.0 | 214.6 | 507.15 | 745.47 |
| structured_free_text_diag | 0.250 | 0.250 | 1.000 | 17.0 | 212.6 | 498.07 | 745.43 |
| full_hidden_log | 0.250 | 0.250 | 1.000 | 366.5 | 297.1 | 646.64 | 905.55 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
