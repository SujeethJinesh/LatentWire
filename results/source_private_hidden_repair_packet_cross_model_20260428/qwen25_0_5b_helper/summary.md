# Source-Private Hidden-Repair Model-Packet Gate

- examples: `64`
- pass gate: `True`
- packet valid rate: `0.984`
- matched minus best no-source: `0.734`
- matched minus best control: `0.734`

| Condition | Correct | Accuracy | Mean bytes | Max bytes | Mean tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 16/64 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| matched_model_packet | 63/64 | 0.984 | 1.97 | 2 | 3.08 | 330.85 |
| zero_source | 16/64 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| shuffled_model_packet | 16/64 | 0.250 | 1.97 | 2 | 0.98 | 0.00 |
| random_same_byte | 16/64 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_only | 16/64 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_masked | 16/64 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| target_derived_sidecar | 16/64 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| full_diag_oracle | 64/64 | 1.000 | 2.00 | 2 | 1.00 | 0.00 |

Pass rule: matched model-extracted hidden-repair packet must beat no-source by >=0.15 and controls stay within +0.02.
