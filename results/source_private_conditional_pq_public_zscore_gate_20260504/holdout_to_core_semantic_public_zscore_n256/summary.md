# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `semantic`
- conditioning mode: `public_zscore`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.453`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.441`
- unquantized predicted accuracy: `0.254`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0017 |
| source | 0.453 | 4.00 | 1.3762 |
| label_shuffled_encoder | 0.441 | 4.00 | 1.2587 |
| constrained_shuffled_source | 0.230 | 4.00 | 1.3209 |
| same_answer_slot_wrong_row_source | 0.336 | 4.00 | 1.3272 |
| answer_masked_source | 0.250 | 4.00 | 1.2514 |
| public_condition_only | 0.250 | 4.00 | 1.2524 |
| permuted_codes | 0.129 | 4.00 | 1.2557 |
| random_same_byte | 0.359 | 4.00 | 0.7264 |
| deranged_public_basis | 0.191 | 4.00 | 1.2892 |
| opaque_slot_basis | 0.234 | 4.00 | 0.7636 |

Payload uniqueness: `{'unique_payloads': 116, 'unique_payload_ratio': 0.453125, 'max_payload_frequency': 20, 'reused_payload_examples': 188, 'reused_payload_accuracy': 0.44680851063829785}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
