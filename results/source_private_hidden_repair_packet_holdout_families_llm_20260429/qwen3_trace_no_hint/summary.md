# Source-Private Hidden-Repair Model-Packet Gate

- examples: `500`
- pass gate: `True`
- packet valid rate: `0.864`
- matched minus best no-source: `0.672`
- matched minus best control: `0.664`

| Condition | Correct | Accuracy | Mean bytes | Max bytes | Mean tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 125/500 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| matched_model_packet | 461/500 | 0.922 | 1.73 | 2 | 2.91 | 330.81 |
| zero_source | 125/500 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| shuffled_model_packet | 125/500 | 0.250 | 1.73 | 2 | 0.86 | 0.00 |
| random_same_byte | 129/500 | 0.258 | 2.00 | 2 | 1.00 | 0.00 |
| answer_only | 125/500 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_masked | 125/500 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| target_derived_sidecar | 125/500 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| full_diag_oracle | 500/500 | 1.000 | 2.00 | 2 | 1.00 | 0.00 |

Pass rule: matched model-extracted hidden-repair packet must beat no-source by >=0.15 and controls stay within +0.02.
