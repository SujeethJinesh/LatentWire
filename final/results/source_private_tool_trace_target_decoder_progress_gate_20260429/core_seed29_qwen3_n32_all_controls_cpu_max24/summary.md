# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `32`
- pass gate: `True`
- matched minus target: `0.438`
- matched minus best control: `0.438`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 8/32 | 0.250 | 1.000 | 0.00 | 13.00 | 2174.86 |
| matched_packet | 22/32 | 0.688 | 1.000 | 2.00 | 13.00 | 2117.03 |
| shuffled_packet | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 2200.52 |
| random_same_byte | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 2083.94 |
| structured_json_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 2187.98 |
| structured_free_text_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 13.00 | 2090.66 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
