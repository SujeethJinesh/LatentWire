# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `160`
- pass gate: `True`
- matched minus target: `0.444`
- matched minus best control: `0.444`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 40/160 | 0.250 | 1.000 | 0.00 | 13.00 | 2721.09 |
| matched_packet | 111/160 | 0.694 | 1.000 | 2.00 | 13.00 | 2670.31 |
| shuffled_packet | 40/160 | 0.250 | 1.000 | 2.00 | 13.00 | 2673.13 |
| random_same_byte | 40/160 | 0.250 | 1.000 | 2.00 | 13.00 | 2681.52 |
| structured_json_2byte | 40/160 | 0.250 | 1.000 | 2.00 | 13.00 | 2666.22 |
| structured_free_text_2byte | 40/160 | 0.250 | 1.000 | 2.00 | 13.00 | 2716.56 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
