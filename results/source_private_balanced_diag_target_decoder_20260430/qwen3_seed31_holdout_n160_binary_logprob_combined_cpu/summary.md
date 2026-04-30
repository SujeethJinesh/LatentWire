# Source-Private Tool-Trace Target-Decoder Smoke

- examples: `160`
- pass gate: `True`
- matched minus target: `0.750`
- matched minus best control: `0.750`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 40/160 | 0.250 | 1.000 | 0.00 | 0.00 | 0.00 |
| matched_packet | 160/160 | 1.000 | 1.000 | 2.00 | 4.00 | 1674.07 |
| deranged_candidate_diag_table | 0/160 | 0.000 | 1.000 | 2.00 | 4.00 | 1665.35 |
| shuffled_packet | 40/160 | 0.250 | 1.000 | 2.00 | 4.00 | 1669.80 |
| random_same_byte | 39/160 | 0.244 | 1.000 | 2.00 | 4.00 | 1667.96 |
| random_noncandidate_same_byte | 40/160 | 0.250 | 1.000 | 2.00 | 4.00 | 1668.56 |
| structured_json_2byte | 40/160 | 0.250 | 1.000 | 2.00 | 0.00 | 0.01 |
| structured_free_text_2byte | 40/160 | 0.250 | 1.000 | 2.00 | 0.00 | 0.00 |

Pass rule: matched LLM target decoder must beat target-only by >=0.15 while shuffled/random/2-byte structured relay controls stay within +0.05.
