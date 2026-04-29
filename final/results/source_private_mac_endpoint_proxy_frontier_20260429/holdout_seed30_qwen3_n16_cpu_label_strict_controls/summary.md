# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.375`
- packet strict-label accuracy: `0.562`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `190.35 ms`
- full-log p50 E2E delta vs packet: `352.80 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.4 | 497.84 | 2231.17 |
| matched_packet | 0.625 | 0.562 | 1.000 | 2.0 | 218.4 | 509.36 | 2089.39 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.4 | 603.87 | 2118.61 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.4 | 477.33 | 2187.80 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 218.4 | 503.51 | 2095.87 |
| query_aware_diag_span | 0.688 | 0.688 | 1.000 | 14.0 | 221.9 | 502.24 | 2139.07 |
| structured_json_diag | 0.625 | 0.625 | 1.000 | 21.0 | 223.4 | 471.20 | 2157.31 |
| structured_free_text_diag | 0.688 | 0.688 | 1.000 | 17.0 | 221.4 | 561.77 | 2088.57 |
| full_hidden_log | 0.438 | 0.438 | 1.000 | 373.5 | 308.8 | 699.71 | 2442.19 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
