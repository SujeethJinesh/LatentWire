# Source-Private Mac Endpoint-Proxy Frontier

- examples: `160`
- prompt style: `label_strict`
- candidate view: `label_blind`
- pass gate: `False`
- packet minus target accuracy: `-0.006`
- packet strict-label accuracy: `0.244`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `154.85 ms`
- full-log p50 E2E delta vs packet: `162.14 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 213.0 | 412.97 | 638.20 |
| matched_packet | 0.244 | 0.244 | 1.000 | 2.0 | 213.0 | 412.16 | 637.53 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 212.0 | 411.64 | 636.16 |
| random_same_byte_packet | 0.250 | 0.250 | 1.000 | 2.0 | 213.0 | 411.62 | 638.32 |
| deranged_candidate_diag_table | 0.244 | 0.244 | 1.000 | 2.0 | 213.0 | 411.05 | 638.00 |
| query_aware_diag_span | 0.250 | 0.250 | 1.000 | 14.0 | 216.3 | 418.92 | 640.09 |
| structured_json_diag | 0.250 | 0.250 | 1.000 | 21.0 | 218.0 | 415.48 | 650.92 |
| structured_free_text_diag | 0.250 | 0.250 | 1.000 | 17.0 | 216.0 | 411.11 | 645.92 |
| full_hidden_log | 0.250 | 0.250 | 1.000 | 373.5 | 303.2 | 567.01 | 799.67 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
