# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `32`
- pass gate: `True`
- matched minus target: `0.500`
- matched minus best control: `0.469`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 8/32 | 0.250 | 1.000 | 0.00 | 13.00 | 1284.34 |
| matched_packet | 24/32 | 0.750 | 1.000 | 2.00 | 13.00 | 1314.67 |
| shuffled_packet | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 1255.30 |
| random_same_byte | 9/32 | 0.281 | 1.000 | 2.00 | 13.00 | 1267.98 |
| structured_json_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 1302.73 |
| structured_free_text_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 1291.94 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
