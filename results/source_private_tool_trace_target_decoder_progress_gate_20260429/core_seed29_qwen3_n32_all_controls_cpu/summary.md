# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `32`
- pass gate: `False`
- matched minus target: `0.000`
- matched minus best control: `0.000`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 0/32 | 0.000 | 0.000 | 0.00 | 8.00 | 1707.28 |
| matched_packet | 0/32 | 0.000 | 0.000 | 2.00 | 8.00 | 1689.95 |
| shuffled_packet | 0/32 | 0.000 | 0.000 | 2.00 | 8.00 | 1653.85 |
| random_same_byte | 0/32 | 0.000 | 0.000 | 2.00 | 8.00 | 1705.34 |
| structured_json_2byte | 0/32 | 0.000 | 0.000 | 2.00 | 8.00 | 1760.53 |
| structured_free_text_2byte | 0/32 | 0.000 | 0.000 | 2.00 | 8.00 | 1611.11 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
