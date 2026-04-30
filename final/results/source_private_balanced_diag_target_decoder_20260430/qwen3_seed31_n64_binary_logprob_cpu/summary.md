# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `64`
- pass gate: `True`
- matched minus target: `0.750`
- matched minus best control: `0.734`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 16/64 | 0.250 | 1.000 | 0.00 | 0.00 | 0.00 |
| matched_packet | 64/64 | 1.000 | 1.000 | 2.00 | 4.00 | 1119.06 |
| shuffled_packet | 16/64 | 0.250 | 1.000 | 2.00 | 4.00 | 1105.92 |
| random_same_byte | 17/64 | 0.266 | 1.000 | 2.00 | 4.00 | 1100.88 |
| structured_json_2byte | 16/64 | 0.250 | 1.000 | 2.00 | 0.00 | 0.00 |
| structured_free_text_2byte | 16/64 | 0.250 | 1.000 | 2.00 | 0.00 | 0.00 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
