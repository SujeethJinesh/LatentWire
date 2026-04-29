# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `64`
- pass gate: `True`
- matched minus target: `0.469`
- matched minus best control: `0.453`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 16/64 | 0.250 | 1.000 | 0.00 | 13.00 | 2236.04 |
| matched_packet | 46/64 | 0.719 | 1.000 | 2.00 | 13.00 | 2236.99 |
| shuffled_packet | 16/64 | 0.250 | 1.000 | 2.00 | 13.00 | 2239.18 |
| random_same_byte | 17/64 | 0.266 | 1.000 | 2.00 | 13.00 | 2225.52 |
| structured_json_2byte | 16/64 | 0.250 | 1.000 | 2.00 | 13.00 | 2251.32 |
| structured_free_text_2byte | 16/64 | 0.250 | 1.000 | 2.00 | 13.00 | 2272.07 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
