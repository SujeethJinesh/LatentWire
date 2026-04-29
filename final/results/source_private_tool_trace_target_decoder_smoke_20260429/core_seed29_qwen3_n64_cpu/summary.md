# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `64`
- pass gate: `True`
- matched minus target: `0.406`
- matched minus best control: `0.406`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 16/64 | 0.250 | 1.000 | 0.00 | 13.00 | 2163.30 |
| matched_packet | 42/64 | 0.656 | 1.000 | 2.00 | 13.00 | 2181.99 |
| shuffled_packet | 16/64 | 0.250 | 1.000 | 2.00 | 13.00 | 2163.92 |
| random_same_byte | 16/64 | 0.250 | 1.000 | 2.00 | 13.00 | 2168.60 |
| structured_json_2byte | 16/64 | 0.250 | 1.000 | 2.00 | 13.00 | 2159.84 |
| structured_free_text_2byte | 16/64 | 0.250 | 1.000 | 2.00 | 13.00 | 2180.77 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
