# SVAMP32 C2C Generated-Answer Packet Audit

- date: `2026-05-05`
- status: `generated_answer_packet_is_answer_leak_not_method`
- reference rows: `32`
- output JSON: `results/svamp32_c2c_generated_answer_packet_audit_20260505/generated_answer_packet_audit.json`

## Result

| Condition | Correct | Teacher-only | Clean | Helps | Harms |
|---|---:|---:|---:|---:|---:|
| generated_answer_value_packet | 16/32 | 10 | 10 | 10 | 2 |
| same_byte_visible_answer_text | 16/32 | 10 | 10 | 10 | 2 |
| generated_answer_index_packet | 16/32 | 10 | 10 | 10 | 2 |
| target_only | 8/32 | 0 | 0 | 0 | 0 |
| source_alone | 5/32 | 0 | 0 | 3 | 6 |
| text_to_text | 2/32 | 0 | 0 | 1 | 7 |
| wrong_row_value_packet | 2/32 | 0 | 0 | 2 | 8 |
| index_row_shuffle | 6/32 | 3 | 3 | 3 | 5 |
| same_source_choice_wrong_row | 15/32 | 9 | 9 | 9 | 2 |
| candidate_roll | 2/32 | 0 | 0 | 1 | 7 |
| candidate_derangement | 2/32 | 0 | 0 | 1 | 7 |
| zero_packet | 8/32 | 0 | 0 | 0 | 0 |

## Packet Contract

- kind: `generated_answer_value_or_candidate_index_packet`
- source-private: `False`
- teacher-derived and not deployable: `True`
- average candidate-index bits per row: `1.688`
- average candidate-index bytes per row: `0.211`
- average visible-answer text bytes per row: `1.656`
- cacheline-rounded bytes per row: `64.0`

## Interpretation

Direct generated-answer alignment recovers the dense C2C replay only by sending the answer itself, or an equivalent candidate index. The `same_byte_visible_answer_text` control is identical to the generated answer packet, so this is an upper-bound and leakage audit rather than a source-private communication method.

- answer-label clean IDs: `10`
- publishable source-necessary clean IDs after destructive controls: `0`

## Decision

Do not promote generated-answer value/index packets as LatentWire. The next C2C-distillation gate must use pre-answer state or a source-side packet target that is not equivalent to revealing the generated answer.
