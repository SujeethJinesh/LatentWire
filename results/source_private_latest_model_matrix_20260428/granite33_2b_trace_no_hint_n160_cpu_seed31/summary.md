# Source-Private Hidden-Repair Model-Packet Gate

- examples: `160`
- pass gate: `True`
- packet valid rate: `0.537`
- matched minus best no-source: `0.381`
- matched minus best control: `0.381`

| Condition | Correct | Accuracy | Mean bytes | Max bytes | Mean tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 40/160 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| matched_model_packet | 101/160 | 0.631 | 1.07 | 2 | 2.83 | 3690.74 |
| zero_source | 40/160 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| shuffled_model_packet | 40/160 | 0.250 | 1.07 | 2 | 0.54 | 0.00 |
| random_same_byte | 40/160 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_only | 40/160 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_masked | 40/160 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| target_derived_sidecar | 40/160 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| full_diag_oracle | 160/160 | 1.000 | 2.00 | 2 | 1.00 | 0.00 |

Pass rule: matched model-extracted hidden-repair packet must beat no-source by >=0.15 and controls stay within +0.02.
