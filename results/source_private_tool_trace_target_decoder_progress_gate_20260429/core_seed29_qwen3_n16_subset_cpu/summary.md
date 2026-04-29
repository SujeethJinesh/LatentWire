# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `16`
- pass gate: `True`
- matched minus target: `0.438`
- matched minus best control: `0.438`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 4/16 | 0.250 | 1.000 | 0.00 | 13.00 | 4112.49 |
| matched_packet | 11/16 | 0.688 | 1.000 | 2.00 | 13.00 | 4190.55 |
| shuffled_packet | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 4209.44 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
