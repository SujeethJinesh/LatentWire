# Source-Private Conditional PQ Innovation Gate

- pass gate: `True`
- examples: `500`
- train/eval ID overlap: `0`
- basis view: `shared_text`
- variant: `utility_protected_hadamard`
- budget bytes: `4`
- source accuracy: `1.000`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.274`
- unquantized predicted accuracy: `0.826`
- target innovation oracle accuracy: `0.808`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.0018 |
| source | 1.000 | 4.00 | 0.8577 |
| label_shuffled_encoder | 0.274 | 4.00 | 0.3445 |
| constrained_shuffled_source | 0.262 | 4.00 | 0.4022 |
| answer_masked_source | 0.250 | 4.00 | 0.3368 |
| permuted_codes | 0.256 | 4.00 | 0.3386 |
| random_same_byte | 0.250 | 4.00 | 0.1018 |
| deranged_public_basis | 0.000 | 4.00 | 0.3754 |
| opaque_slot_basis | 0.250 | 4.00 | 0.3899 |

Payload uniqueness: `{'unique_payloads': 436, 'unique_payload_ratio': 0.872, 'max_payload_frequency': 7, 'reused_payload_examples': 112, 'reused_payload_accuracy': 1.0}`

Pass rule: Pass requires disjoint train/eval IDs, exact ID parity, source >= target+0.15, source >= best destructive control+0.15, all controls <= target+0.06, paired CI95 low vs best control > +0.10, and payload uniqueness below lookup-risk or reused-payload accuracy >= target+0.10.
