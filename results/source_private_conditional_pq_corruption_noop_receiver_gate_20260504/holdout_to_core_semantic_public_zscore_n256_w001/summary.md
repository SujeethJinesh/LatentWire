# Conditional PQ Corruption-to-Noop Receiver Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- train/eval families: `holdout->core`
- basis view: `semantic`
- conditioning mode: `public_zscore`
- budget bytes: `4`
- source accuracy: `0.301`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.352`
- source minus best control: `-0.051`
- CI95 low vs best control: `-0.106`
- helps/harms: `13/0`
- unquantized predicted accuracy: `0.254`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.7354 |
| source | 0.301 | 4.00 | 1.3722 |
| label_shuffled_encoder | 0.352 | 4.00 | 1.2592 |
| constrained_shuffled_source | 0.312 | 4.00 | 1.3260 |
| same_answer_slot_wrong_row_source | 0.281 | 4.00 | 1.3272 |
| answer_masked_source | 0.250 | 4.00 | 1.2552 |
| public_condition_only | 0.250 | 4.00 | 1.2523 |
| permuted_codes | 0.074 | 4.00 | 1.2572 |
| random_same_byte | 0.293 | 4.00 | 0.7293 |
| deranged_public_basis | 0.176 | 4.00 | 1.2899 |
| candidate_roll | 0.164 | 4.00 | 1.2712 |
| opaque_slot_basis | 0.273 | 4.00 | 0.7667 |

Payload uniqueness: `{'unique_payloads': 116, 'unique_payload_ratio': 0.453125, 'max_payload_frequency': 20, 'reused_payload_examples': 188, 'reused_payload_accuracy': 0.30851063829787234}`

Pass rule: Pass requires disjoint train/eval IDs, source >= target+0.05, source >= best destructive/shortcut control+0.10, and paired CI95 low versus best control > 0.
