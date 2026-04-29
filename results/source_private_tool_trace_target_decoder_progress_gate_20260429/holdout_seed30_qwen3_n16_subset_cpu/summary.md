# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `16`
- pass gate: `True`
- matched minus target: `0.500`
- matched minus best control: `0.500`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 4/16 | 0.250 | 1.000 | 0.00 | 13.00 | 4057.42 |
| matched_packet | 12/16 | 0.750 | 1.000 | 2.00 | 13.00 | 4059.46 |
| shuffled_packet | 4/16 | 0.250 | 1.000 | 2.00 | 13.00 | 4059.14 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
