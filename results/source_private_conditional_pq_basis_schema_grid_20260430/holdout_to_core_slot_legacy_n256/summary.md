# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `slot`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.246`
- target accuracy: `0.250`
- best control: `deranged_public_basis` at `0.258`
- unquantized predicted accuracy: `0.254`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0020 |
| source | 0.246 | 4.00 | 0.4381 |
| label_shuffled_encoder | 0.254 | 4.00 | 0.3376 |
| constrained_shuffled_source | 0.230 | 4.00 | 0.3609 |
| answer_masked_source | 0.254 | 4.00 | 0.3344 |
| permuted_codes | 0.250 | 4.00 | 0.3370 |
| random_same_byte | 0.254 | 4.00 | 0.1399 |
| deranged_public_basis | 0.258 | 4.00 | 0.3697 |
| opaque_slot_basis | 0.246 | 4.00 | 0.3364 |

Payload uniqueness: `{'unique_payloads': 226, 'unique_payload_ratio': 0.8828125, 'max_payload_frequency': 4, 'reused_payload_examples': 56, 'reused_payload_accuracy': 0.23214285714285715}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
