# Source-Private Mac Endpoint-Proxy Frontier

- examples: `160`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.425`
- packet strict-label accuracy: `0.662`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `164.27 ms`
- full-log p50 E2E delta vs packet: `304.75 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.3 | 482.65 | 2141.50 |
| matched_packet | 0.675 | 0.662 | 1.000 | 2.0 | 218.3 | 471.64 | 2126.43 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.3 | 473.02 | 2145.53 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.3 | 459.33 | 2134.41 |
| deranged_candidate_diag_table | 0.244 | 0.244 | 1.000 | 2.0 | 218.3 | 465.86 | 2141.50 |
| query_aware_diag_span | 0.694 | 0.694 | 1.000 | 14.0 | 221.6 | 452.03 | 2145.95 |
| structured_json_diag | 0.575 | 0.575 | 1.000 | 21.0 | 223.3 | 478.58 | 2134.54 |
| structured_free_text_diag | 0.713 | 0.713 | 1.000 | 17.0 | 221.3 | 474.18 | 2151.35 |
| full_hidden_log | 0.463 | 0.456 | 1.000 | 366.5 | 305.7 | 635.91 | 2431.18 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
