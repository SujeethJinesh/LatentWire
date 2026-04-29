# Source-Private Mac Endpoint-Proxy Frontier

- examples: `16`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.562`
- packet strict-label accuracy: `0.250`
- best source-destroying control accuracy: `0.312`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `227.14 ms`
- full-log p50 E2E delta vs packet: `181.97 ms`

| Condition | Accuracy | Strict accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_only | 0.312 | 0.250 | 1.000 | 0.0 | 193.4 | 478.48 | 2046.49 |
| matched_packet | 0.875 | 0.250 | 1.000 | 2.0 | 193.4 | 461.16 | 838.02 |
| matched_byte_text_2 | 0.312 | 0.188 | 1.000 | 2.0 | 192.4 | 473.45 | 1994.45 |
| random_same_byte_packet | 0.125 | 0.000 | 1.000 | 2.0 | 193.4 | 458.53 | 1454.28 |
| deranged_candidate_diag_table | 0.000 | 0.000 | 1.000 | 2.0 | 193.4 | 450.01 | 884.76 |
| query_aware_diag_span | 0.812 | 0.312 | 1.000 | 14.0 | 196.9 | 503.99 | 841.79 |
| structured_json_diag | 0.750 | 0.125 | 1.000 | 21.0 | 198.4 | 472.48 | 826.56 |
| structured_free_text_diag | 0.875 | 0.125 | 1.000 | 17.0 | 196.4 | 466.48 | 790.05 |
| full_hidden_log | 0.375 | 0.000 | 0.375 | 373.5 | 283.8 | 688.29 | 1020.00 |

Pass rule: Packet must beat target by >=0.15, all included source-destroying controls must stay within target+0.05, matched packet valid rate must be >=0.95, exact ID parity must hold, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
