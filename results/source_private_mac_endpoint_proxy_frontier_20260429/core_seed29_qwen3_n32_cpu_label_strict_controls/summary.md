# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.438`
- packet strict-label accuracy: `0.656`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `164.84 ms`
- full-log p50 E2E delta vs packet: `232.30 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.3 | 519.93 | 2083.72 |
| matched_packet | 0.688 | 0.656 | 1.000 | 2.0 | 218.3 | 495.90 | 2101.81 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.3 | 497.27 | 2090.84 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.3 | 514.60 | 2079.67 |
| deranged_candidate_diag_table | 0.219 | 0.219 | 1.000 | 2.0 | 218.3 | 466.44 | 2141.00 |
| query_aware_diag_span | 0.688 | 0.688 | 1.000 | 14.0 | 221.7 | 492.02 | 2078.23 |
| structured_json_diag | 0.531 | 0.531 | 1.000 | 21.0 | 223.3 | 525.71 | 2077.96 |
| structured_free_text_diag | 0.688 | 0.688 | 1.000 | 17.0 | 221.3 | 468.63 | 2114.30 |
| full_hidden_log | 0.406 | 0.406 | 1.000 | 366.5 | 305.8 | 660.74 | 2334.11 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
