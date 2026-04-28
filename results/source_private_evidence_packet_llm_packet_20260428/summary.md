# Source-Private Evidence Packet Model-Packet Gate

- examples: `16`
- budget bytes: `2`
- pass gate: `False`
- packet nonempty rate: `0.562`
- matched minus best no-source: `0.000`
- matched minus best control: `0.000`
- source-final minus best no-source: `0.750`

| Condition | Correct | Accuracy | Mean bytes | Max bytes | p50 latency ms |
|---|---:|---:|---:|---:|---:|
| target_only | 4/16 | 0.250 | 0.00 | 0 | 0.00 |
| matched_model_packet | 4/16 | 0.250 | 1.12 | 2 | 1741.28 |
| zero_source | 4/16 | 0.250 | 0.00 | 0 | 0.00 |
| shuffled_model_packet | 4/16 | 0.250 | 1.12 | 2 | 0.00 |
| random_same_byte | 4/16 | 0.250 | 2.00 | 2 | 0.00 |
| answer_only | 4/16 | 0.250 | 2.00 | 2 | 0.00 |
| answer_masked | 4/16 | 0.250 | 0.00 | 0 | 0.00 |
| source_final_only | 16/16 | 1.000 | 28.00 | 32 | 0.00 |

Pass rule: matched model-produced digest packet must beat target-only by >=0.15, source-destroying controls must remain within +0.02, and source-final-only must not explain the gain.
