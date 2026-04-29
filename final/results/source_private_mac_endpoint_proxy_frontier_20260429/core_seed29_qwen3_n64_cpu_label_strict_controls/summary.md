# Source-Private Mac Endpoint-Proxy Frontier

- examples: `64`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.453`
- packet strict-label accuracy: `0.672`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `217.23 ms`
- full-log p50 E2E delta vs packet: `294.22 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.3 | 533.61 | 2319.72 |
| matched_packet | 0.703 | 0.672 | 1.000 | 2.0 | 218.3 | 569.90 | 2275.79 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.3 | 537.43 | 2332.80 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.3 | 563.53 | 2323.51 |
| deranged_candidate_diag_table | 0.234 | 0.234 | 1.000 | 2.0 | 218.3 | 565.51 | 2268.85 |
| query_aware_diag_span | 0.703 | 0.703 | 1.000 | 14.0 | 221.7 | 524.99 | 2345.95 |
| structured_json_diag | 0.594 | 0.594 | 1.000 | 21.0 | 223.3 | 571.75 | 2323.53 |
| structured_free_text_diag | 0.719 | 0.719 | 1.000 | 17.0 | 221.3 | 523.92 | 2338.83 |
| full_hidden_log | 0.484 | 0.469 | 1.000 | 366.5 | 305.8 | 787.12 | 2570.01 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
