# Source-Private Mac Endpoint-Proxy Frontier

- examples: `64`
- prompt style: `label_strict`
- pass gate: `True`
- packet minus target accuracy: `0.422`
- packet strict-label accuracy: `0.656`
- best source-destroying control accuracy: `0.250`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `192.71 ms`
- full-log p50 E2E delta vs packet: `279.03 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 1.000 | 0.0 | 218.3 | 501.40 | 2079.15 |
| matched_packet | 0.672 | 0.656 | 1.000 | 2.0 | 218.3 | 508.00 | 2071.16 |
| matched_byte_text_2 | 0.250 | 0.250 | 1.000 | 2.0 | 217.3 | 481.13 | 2097.19 |
| random_same_byte_packet | 0.000 | 0.000 | 1.000 | 2.0 | 218.3 | 485.22 | 2085.04 |
| deranged_candidate_diag_table | 0.250 | 0.250 | 1.000 | 2.0 | 218.3 | 508.96 | 2088.00 |
| query_aware_diag_span | 0.703 | 0.703 | 1.000 | 14.0 | 221.7 | 497.56 | 2092.36 |
| structured_json_diag | 0.609 | 0.609 | 1.000 | 21.0 | 223.3 | 511.42 | 2106.08 |
| structured_free_text_diag | 0.719 | 0.719 | 1.000 | 17.0 | 221.3 | 496.86 | 2077.49 |
| full_hidden_log | 0.531 | 0.516 | 1.000 | 373.5 | 308.5 | 700.72 | 2350.19 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
