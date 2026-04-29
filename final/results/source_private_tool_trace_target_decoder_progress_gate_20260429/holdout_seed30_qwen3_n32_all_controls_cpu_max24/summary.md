# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `32`
- pass gate: `True`
- matched minus target: `0.500`
- matched minus best control: `0.469`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 8/32 | 0.250 | 1.000 | 0.00 | 13.00 | 2081.95 |
| matched_packet | 24/32 | 0.750 | 1.000 | 2.00 | 13.00 | 2123.86 |
| shuffled_packet | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 2072.03 |
| random_same_byte | 9/32 | 0.281 | 1.000 | 2.00 | 13.00 | 2133.29 |
| structured_json_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 2079.27 |
| structured_free_text_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 2116.34 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
