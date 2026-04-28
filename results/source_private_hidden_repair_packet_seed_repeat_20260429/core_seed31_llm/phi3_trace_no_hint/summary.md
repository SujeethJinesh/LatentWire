# Source-Private Hidden-Repair Model-Packet Gate

- examples: `500`
- pass gate: `True`
- packet valid rate: `1.000`
- matched minus best no-source: `0.750`
- matched minus best control: `0.744`

| Condition | Correct | Accuracy | Mean bytes | Max bytes | Mean tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 125/500 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| matched_model_packet | 500/500 | 1.000 | 2.00 | 2 | 3.00 | 466.64 |
| zero_source | 125/500 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| shuffled_model_packet | 125/500 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| random_same_byte | 128/500 | 0.256 | 2.00 | 2 | 1.00 | 0.00 |
| answer_only | 125/500 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| answer_masked | 125/500 | 0.250 | 0.00 | 0 | 0.00 | 0.00 |
| target_derived_sidecar | 125/500 | 0.250 | 2.00 | 2 | 1.00 | 0.00 |
| full_diag_oracle | 500/500 | 1.000 | 2.00 | 2 | 1.00 | 0.00 |

Pass rule: matched model-extracted hidden-repair packet must beat no-source by >=0.15 and controls stay within +0.02.
