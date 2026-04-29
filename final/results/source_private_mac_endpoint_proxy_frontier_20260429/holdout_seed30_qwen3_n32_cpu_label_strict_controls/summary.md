# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.406`
- packet strict-label accuracy: `0.625`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `167.06 ms`
- full-log p50 E2E delta vs packet: `335.45 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.3 | 484.02 | 2156.38 |
| matched_packet | 0.656 | 0.625 | 1.000 | 2.0 | 218.3 | 505.49 | 2131.37 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.3 | 483.96 | 2157.98 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.3 | 477.48 | 2124.07 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 218.3 | 492.16 | 2145.02 |
| query_aware_diag_span | 0.656 | 0.656 | 1.000 | 14.0 | 221.7 | 500.55 | 2167.79 |
| structured_json_diag | 0.594 | 0.594 | 1.000 | 21.0 | 223.3 | 495.62 | 2137.49 |
| structured_free_text_diag | 0.688 | 0.688 | 1.000 | 17.0 | 221.3 | 496.35 | 2159.78 |
| full_hidden_log | 0.438 | 0.438 | 1.000 | 373.5 | 308.6 | 672.55 | 2466.82 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
