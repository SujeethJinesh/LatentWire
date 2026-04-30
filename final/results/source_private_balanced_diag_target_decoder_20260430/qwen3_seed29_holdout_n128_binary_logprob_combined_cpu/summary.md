# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `128`
- pass gate: `True`
- matched minus target: `0.750`
- matched minus best control: `0.750`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 32/128 | 0.250 | 1.000 | 0.00 | 0.00 | 0.00 |
| matched_packet | 128/128 | 1.000 | 1.000 | 2.00 | 4.00 | 1630.75 |
| deranged_candidate_diag_table | 0/128 | 0.000 | 1.000 | 2.00 | 4.00 | 1631.26 |
| shuffled_packet | 32/128 | 0.250 | 1.000 | 2.00 | 4.00 | 1640.81 |
| random_same_byte | 32/128 | 0.250 | 1.000 | 2.00 | 4.00 | 1631.89 |
| random_noncandidate_same_byte | 32/128 | 0.250 | 1.000 | 2.00 | 4.00 | 1629.18 |
| structured_json_2byte | 32/128 | 0.250 | 1.000 | 2.00 | 0.00 | 0.01 |
| structured_free_text_2byte | 32/128 | 0.250 | 1.000 | 2.00 | 0.00 | 0.00 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
