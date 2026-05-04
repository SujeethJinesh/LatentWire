# Conditional PQ Integrity Threshold Gate

- pass gate: `False`
- train fit/select/eval: `512/256/256`
- selected score: `negative_min_l2`
- selected threshold: `-1.5634799003601074`
- source accuracy: `0.425781`
- target-only accuracy: `0.250000`
- best control: `label_shuffled_encoder` at `0.457031`
- source minus best control: `-0.031250`
- CI95 low vs best control: `-0.097656`
- source/max corrupt accept rate: `0.773438/1.000000`
- helps/harms: `59/14`

## Condition Metrics

| Condition | Accuracy | Accept Rate | Mean Bytes |
|---|---:|---:|---:|
| `target_only` | `0.250000` | `0.000000` | `0.00` |
| `source` | `0.425781` | `0.773438` | `4.00` |
| `label_shuffled_encoder` | `0.457031` | `0.605469` | `4.00` |
| `constrained_shuffled_source` | `0.343750` | `0.722656` | `4.00` |
| `same_answer_slot_wrong_row_source` | `0.390625` | `0.746094` | `4.00` |
| `answer_masked_source` | `0.250000` | `1.000000` | `4.00` |
| `public_condition_only` | `0.250000` | `1.000000` | `4.00` |
| `permuted_codes` | `0.250000` | `0.183594` | `4.00` |
| `random_same_byte` | `0.261719` | `0.074219` | `4.00` |
| `deranged_public_basis` | `0.214844` | `0.773438` | `4.00` |
| `candidate_roll` | `0.222656` | `0.773438` | `4.00` |
| `opaque_slot_basis` | `0.269531` | `0.699219` | `4.00` |

## Interpretation

This gate tests whether an explicit scalar packet-integrity rule can preserve matched conditional-PQ packet gains while forcing corrupted packet controls to no-op. A failure rules out simple thresholded integrity on top of the existing public-zscore receiver.
