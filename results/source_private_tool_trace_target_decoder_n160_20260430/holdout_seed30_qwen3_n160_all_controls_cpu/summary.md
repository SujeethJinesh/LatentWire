# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `160`
- pass gate: `True`
- matched minus target: `0.469`
- matched minus best control: `0.456`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 40/160 | 0.250 | 1.000 | 0.00 | 13.00 | 2442.63 |
| matched_packet | 115/160 | 0.719 | 1.000 | 2.00 | 13.00 | 2451.11 |
| shuffled_packet | 41/160 | 0.256 | 1.000 | 2.00 | 13.00 | 2447.00 |
| random_same_byte | 42/160 | 0.263 | 1.000 | 2.00 | 13.00 | 2432.70 |
| structured_json_2byte | 40/160 | 0.250 | 1.000 | 2.00 | 13.00 | 2405.01 |
| structured_free_text_2byte | 40/160 | 0.250 | 1.000 | 2.00 | 13.00 | 2462.44 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
