# Source-Private Conditional PQ Innovation Gate

- pass gate: `True`
- examples: `256`
- train/eval ID overlap: `0`
- basis view: `shared_text`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `1.000`
- target accuracy: `0.250`
- best control: `constrained_shuffled_source` at `0.277`
- unquantized predicted accuracy: `0.832`
- target innovation oracle accuracy: `0.777`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0016 |
| source | 1.000 | 4.00 | 0.8272 |
| label_shuffled_encoder | 0.270 | 4.00 | 0.2935 |
| constrained_shuffled_source | 0.277 | 4.00 | 0.2975 |
| answer_masked_source | 0.250 | 4.00 | 0.2866 |
| permuted_codes | 0.250 | 4.00 | 0.2886 |
| random_same_byte | 0.254 | 4.00 | 0.0974 |
| deranged_public_basis | 0.000 | 4.00 | 0.3122 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3350 |

Payload uniqueness: `{'unique_payloads': 229, 'unique_payload_ratio': 0.89453125, 'max_payload_frequency': 5, 'reused_payload_examples': 48, 'reused_payload_accuracy': 1.0}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
