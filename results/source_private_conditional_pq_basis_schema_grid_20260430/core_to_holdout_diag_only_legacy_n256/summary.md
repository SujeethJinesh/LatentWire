# Source-Private Conditional PQ Innovation Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `diag_only`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `0.312`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.332`
- unquantized predicted accuracy: `0.469`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0021 |
| source | 0.312 | 4.00 | 0.4283 |
| label_shuffled_encoder | 0.332 | 4.00 | 0.3432 |
| constrained_shuffled_source | 0.289 | 4.00 | 0.3674 |
| answer_masked_source | 0.250 | 4.00 | 0.3428 |
| permuted_codes | 0.250 | 4.00 | 0.3424 |
| random_same_byte | 0.250 | 4.00 | 0.1401 |
| deranged_public_basis | 0.230 | 4.00 | 0.3819 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3454 |

Payload uniqueness: `{'unique_payloads': 235, 'unique_payload_ratio': 0.91796875, 'max_payload_frequency': 3, 'reused_payload_examples': 40, 'reused_payload_accuracy': 0.325}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
