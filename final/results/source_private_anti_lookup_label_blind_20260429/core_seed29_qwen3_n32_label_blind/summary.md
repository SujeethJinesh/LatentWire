# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `label_strict`
- candidate view: `label_blind`
- pass gate: `False`
- packet minus target accuracy: `0.000`
- packet strict-label accuracy: `0.250`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `117.71 ms`
- full-log p50 E2E delta vs packet: `221.93 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 209.6 | 828.67 | 1179.84 |
| matched_packet | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 922.96 | 1165.05 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 208.6 | 864.38 | 1179.37 |
| random_same_byte_packet | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 874.32 | 1181.55 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 775.67 | 1106.51 |
| query_aware_diag_span | 0.250 | 0.250 | 1.000 | 14.0 | 213.0 | 902.67 | 1203.66 |
| structured_json_diag | 0.250 | 0.250 | 1.000 | 21.0 | 214.6 | 579.08 | 1186.50 |
| structured_free_text_diag | 0.250 | 0.250 | 1.000 | 17.0 | 212.6 | 947.02 | 1139.77 |
| full_hidden_log | 0.250 | 0.250 | 1.000 | 366.5 | 297.1 | 1040.67 | 1386.98 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
