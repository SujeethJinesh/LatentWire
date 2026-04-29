# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `16`
- pass gate: `False`
- matched minus target: `0.500`
- matched minus best control: `0.438`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 4/16 | 0.250 | 1.000 | 0.00 | 13.00 | 2100.70 |
| matched_packet | 12/16 | 0.750 | 1.000 | 2.00 | 13.00 | 2151.83 |
| shuffled_packet | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 2155.34 |
| random_same_byte | 5/16 | 0.312 | 1.000 | 2.00 | 13.00 | 2149.21 |
| structured_json_2byte | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 2102.32 |
| structured_free_text_2byte | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 2138.26 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
