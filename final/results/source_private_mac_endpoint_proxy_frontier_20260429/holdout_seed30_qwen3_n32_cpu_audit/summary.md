# Source-Private Mac Endpoint-Proxy Frontier

- examples: `32`
- prompt style: `audit`
- pass gate: `True`
- packet minus target accuracy: `0.531`
- packet vs query-aware payload compression: `7.0x`
- packet vs full-log payload compression: `186.8x`
- full-log p50 TTFT delta vs packet: `157.40 ms`
- full-log p50 E2E delta vs packet: `179.57 ms`

| Condition | Accuracy | Valid | Payload bytes | Prompt tokens | p50 TTFT ms | p50 E2E ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0.312 | 1.000 | 0.0 | 193.3 | 455.70 | 1985.75 |
| matched_packet | 0.844 | 1.000 | 2.0 | 193.3 | 452.12 | 821.82 |
| matched_byte_text_2 | 0.312 | 1.000 | 2.0 | 192.3 | 434.75 | 1954.54 |
| query_aware_diag_span | 0.812 | 1.000 | 14.0 | 196.7 | 480.23 | 831.52 |
| structured_json_diag | 0.750 | 1.000 | 21.0 | 198.3 | 475.41 | 824.46 |
| structured_free_text_diag | 0.875 | 1.000 | 17.0 | 196.3 | 466.57 | 765.31 |
| full_hidden_log | 0.344 | 0.344 | 373.5 | 283.6 | 609.53 | 1001.39 |

Pass rule: Packet must beat target by >=0.15, matched-byte text must stay within target+0.05, and query-aware/full-log payloads must be larger than the packet. Latency is reported, not gated.
