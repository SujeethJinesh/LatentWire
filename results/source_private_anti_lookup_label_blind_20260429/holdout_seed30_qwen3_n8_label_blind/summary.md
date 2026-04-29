# Source-Private Mac Endpoint-Proxy Frontier

- examples: `8`
- prompt style: `label_strict`
- candidate view: `label_blind`
- pass gate: `False`
- packet minus target accuracy: `0.000`
- packet strict-label accuracy: `0.250`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `209.05 ms`
- full-log p50 E2E delta vs packet: `135.61 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 213.0 | 799.05 | 1066.98 |
| matched_packet | 0.250 | 0.250 | 1.000 | 2.0 | 213.0 | 812.96 | 1113.85 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 212.0 | 709.57 | 1149.69 |
| random_same_byte_packet | 0.250 | 0.250 | 1.000 | 2.0 | 213.0 | 711.25 | 1081.75 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 213.0 | 729.18 | 1022.16 |
| query_aware_diag_span | 0.250 | 0.250 | 1.000 | 14.0 | 216.5 | 800.67 | 1085.35 |
| structured_json_diag | 0.250 | 0.250 | 1.000 | 21.0 | 218.0 | 709.81 | 1046.63 |
| structured_free_text_diag | 0.250 | 0.250 | 1.000 | 17.0 | 216.0 | 608.84 | 1097.72 |
| full_hidden_log | 0.250 | 0.250 | 1.000 | 373.5 | 303.4 | 1022.01 | 1249.46 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
