# Conditional PQ Corruption-to-Noop Receiver Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- train/eval families: `holdout->core`
- basis view: `semantic`
- conditioning mode: `public_zscore`
- budget bytes: `4`
- source accuracy: `0.250`
- target accuracy: `0.250`
- best control: `label_shuffled_encoder` at `0.250`
- source minus best control: `0.000`
- CI95 low vs best control: `0.000`
- helps/harms: `0/0`
- unquantized predicted accuracy: `0.254`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.7164 |
| source | 0.250 | 4.00 | 1.3584 |
| label_shuffled_encoder | 0.250 | 4.00 | 1.2510 |
| constrained_shuffled_source | 0.250 | 4.00 | 1.3086 |
| same_answer_slot_wrong_row_source | 0.250 | 4.00 | 1.3041 |
| answer_masked_source | 0.250 | 4.00 | 1.2485 |
| public_condition_only | 0.250 | 4.00 | 1.2486 |
| permuted_codes | 0.250 | 4.00 | 1.2511 |
| random_same_byte | 0.250 | 4.00 | 0.7129 |
| deranged_public_basis | 0.250 | 4.00 | 1.2842 |
| candidate_roll | 0.250 | 4.00 | 1.2628 |
| opaque_slot_basis | 0.250 | 4.00 | 0.7690 |

Payload uniqueness: `{'unique_payloads': 116, 'unique_payload_ratio': 0.453125, 'max_payload_frequency': 20, 'reused_payload_examples': 188, 'reused_payload_accuracy': 0.2765957446808511}`

Pass rule: Pass requires disjoint train/eval IDs, source >= target+0.05, source >= best destructive/shortcut control+0.10, and paired CI95 low versus best control > 0.
