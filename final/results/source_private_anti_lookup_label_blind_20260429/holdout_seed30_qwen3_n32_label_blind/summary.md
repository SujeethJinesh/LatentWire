# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `label_strict`
- candidate view: `label_blind`
- pass gate: `False`
- packet minus target accuracy: `0.000`
- packet strict-label accuracy: `0.250`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `287.23 ms`
- full-log p50 E2E delta vs packet: `232.64 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 213.0 | 859.10 | 1203.43 |
| matched_packet | 0.250 | 0.250 | 1.000 | 2.0 | 213.0 | 805.31 | 1198.89 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 212.0 | 867.16 | 1175.02 |
| random_same_byte_packet | 0.250 | 0.250 | 1.000 | 2.0 | 213.0 | 869.93 | 1228.05 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 213.0 | 828.58 | 1151.91 |
| query_aware_diag_span | 0.250 | 0.250 | 1.000 | 14.0 | 216.3 | 943.17 | 1224.61 |
| structured_json_diag | 0.250 | 0.250 | 1.000 | 21.0 | 218.0 | 901.26 | 1177.45 |
| structured_free_text_diag | 0.250 | 0.250 | 1.000 | 17.0 | 216.0 | 902.43 | 1208.42 |
| full_hidden_log | 0.250 | 0.250 | 1.000 | 373.5 | 303.2 | 1092.54 | 1431.53 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
