# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `16`
- pass gate: `True`
- matched minus target: `0.438`
- matched minus best control: `0.438`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 4/16 | 0.250 | 1.000 | 0.00 | 13.00 | 1234.44 |
| matched_packet | 11/16 | 0.688 | 1.000 | 2.00 | 13.00 | 1266.54 |
| shuffled_packet | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 1206.56 |
| random_same_byte | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 1208.90 |
| structured_json_2byte | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 1230.88 |
| structured_free_text_2byte | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 1258.71 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
