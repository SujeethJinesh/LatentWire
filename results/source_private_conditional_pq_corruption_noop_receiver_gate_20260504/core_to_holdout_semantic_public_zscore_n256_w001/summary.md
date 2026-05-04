# Conditional PQ Corruption-to-Noop Receiver Gate

- pass gate: `False`
- examples: `256`
- train/eval ID overlap: `0`
- train/eval families: `core->holdout`
- basis view: `semantic`
- conditioning mode: `public_zscore`
- budget bytes: `4`
- source accuracy: `0.285`
- target accuracy: `0.250`
- best control: `random_same_byte` at `0.367`
- source minus best control: `-0.082`
- CI95 low vs best control: `-0.149`
- helps/harms: `30/21`
- unquantized predicted accuracy: `0.500`
- target innovation oracle accuracy: `1.000`

| Condition | Accuracy | Mean bytes | p50 ms |
|---|---:|---:|---:|
| target_only | 0.250 | 0.00 | 0.7399 |
| source | 0.285 | 4.00 | 1.4211 |
| label_shuffled_encoder | 0.328 | 4.00 | 1.2738 |
| constrained_shuffled_source | 0.309 | 4.00 | 1.3072 |
| same_answer_slot_wrong_row_source | 0.340 | 4.00 | 1.3014 |
| answer_masked_source | 0.250 | 4.00 | 1.2664 |
| public_condition_only | 0.250 | 4.00 | 1.2655 |
| permuted_codes | 0.363 | 4.00 | 1.2690 |
| random_same_byte | 0.367 | 4.00 | 0.7337 |
| deranged_public_basis | 0.211 | 4.00 | 1.3027 |
| candidate_roll | 0.230 | 4.00 | 1.2863 |
| opaque_slot_basis | 0.223 | 4.00 | 0.7714 |

Payload uniqueness: `{'unique_payloads': 187, 'unique_payload_ratio': 0.73046875, 'max_payload_frequency': 15, 'reused_payload_examples': 92, 'reused_payload_accuracy': 0.10869565217391304}`

Pass rule: Pass requires disjoint train/eval IDs, source >= target+0.05, source >= best destructive/shortcut control+0.10, and paired CI95 low versus best control > 0.
