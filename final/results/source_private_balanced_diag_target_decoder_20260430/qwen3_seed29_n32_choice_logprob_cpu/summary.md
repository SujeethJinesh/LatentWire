# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `32`
- pass gate: `False`
- matched minus target: `0.000`
- matched minus best control: `0.000`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 8/32 | 0.250 | 1.000 | 0.00 | 1.00 | 841.44 |
| matched_packet | 8/32 | 0.250 | 1.000 | 2.00 | 1.00 | 738.43 |
| shuffled_packet | 8/32 | 0.250 | 1.000 | 2.00 | 1.00 | 945.44 |
| random_same_byte | 8/32 | 0.250 | 1.000 | 2.00 | 1.00 | 768.04 |
| structured_json_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 1.00 | 837.81 |
| structured_free_text_2byte | 8/32 | 0.250 | 1.000 | 2.00 | 1.00 | 875.14 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
