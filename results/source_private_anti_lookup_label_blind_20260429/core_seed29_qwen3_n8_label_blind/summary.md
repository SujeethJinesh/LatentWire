# Source-Private Mac Endpoint-Proxy Frontier

- examples: `8`
- prompt style: `label_strict`
- candidate view: `label_blind`
- pass gate: `False`
- packet minus target accuracy: `0.000`
- packet strict-label accuracy: `0.250`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `183.2x`
- full-log p50 TTFT delta vs packet: `211.32 ms`
- full-log p50 E2E delta vs packet: `231.49 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 209.6 | 821.86 | 1209.15 |
| matched_packet | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 704.87 | 1208.11 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 208.6 | 921.13 | 1162.88 |
| random_same_byte_packet | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 817.11 | 1195.29 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 209.6 | 896.61 | 1153.06 |
| query_aware_diag_span | 0.250 | 0.250 | 1.000 | 14.0 | 213.0 | 890.39 | 1200.52 |
| structured_json_diag | 0.250 | 0.250 | 1.000 | 21.0 | 214.6 | 719.94 | 1177.54 |
| structured_free_text_diag | 0.250 | 0.250 | 1.000 | 17.0 | 212.6 | 928.05 | 1199.08 |
| full_hidden_log | 0.250 | 0.250 | 1.000 | 366.5 | 297.1 | 916.18 | 1439.60 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
