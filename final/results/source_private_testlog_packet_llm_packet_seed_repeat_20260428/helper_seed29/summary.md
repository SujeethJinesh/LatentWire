# Source-Private Test-Log Model-Packet Gate

- examples: `160`
- pass gate: `True`
- packet valid rate: `0.919`
- matched minus best no-source: `0.688`
- matched minus best control: `0.688`

| Condition | Correct | Accuracy | Mean bytes | Max bytes | Mean tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 40/160 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| matched_model_packet | 150/160 | 0.938 | 1.84 | 2 | 2.43 | 167.09 |
| zero_source | 40/160 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| shuffled_model_packet | 40/160 | 0.250 | 1.84 | 2 | 0.92 | 0.00 |
| random_same_byte | 40/160 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_only | 40/160 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_masked | 40/160 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| target_derived_sidecar | 40/160 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| full_signature_oracle | 160/160 | 1.000 | 2.00 | 2 | 1.00 | 0.00 |

Pass rule: matched model-extracted test-log packet must beat target-only by >=0.15 and all source-destroying controls must remain within +0.02 of no-source.
