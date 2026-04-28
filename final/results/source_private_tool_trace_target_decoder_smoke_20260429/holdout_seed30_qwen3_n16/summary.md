# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `16`
- pass gate: `False`
- matched minus target: `0.500`
- matched minus best control: `0.438`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 4/16 | 0.250 | 1.000 | 0.00 | 13.00 | 1216.02 |
| matched_packet | 12/16 | 0.750 | 1.000 | 2.00 | 13.00 | 1239.74 |
| shuffled_packet | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 1211.01 |
| random_same_byte | 5/16 | 0.312 | 1.000 | 2.00 | 13.00 | 1181.76 |
| structured_json_2byte | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 1218.50 |
| structured_free_text_2byte | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 1195.95 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
