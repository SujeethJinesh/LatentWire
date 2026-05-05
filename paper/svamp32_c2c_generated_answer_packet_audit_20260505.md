# SVAMP32 C2C Generated-Answer Packet Audit

- date: `2026-05-05`
- status: `generated_answer_packet_is_answer_leak_not_method`
- artifact:
  `results/svamp32_c2c_generated_answer_packet_audit_20260505/generated_answer_packet_audit.json`
- manifest:
  `results/svamp32_c2c_generated_answer_packet_audit_20260505/manifest.md`
- script:
  `scripts/analyze_svamp32_c2c_generated_answer_packet_audit.py`

## Question

Can the next C2C-distillation gate use the C2C generated answer directly as the
packet target without falling into answer leakage or source-choice shortcuts?

## Result

No. Direct generated-answer packets recover the repaired C2C replay, but they
are equivalent to same-byte visible answer text and are nearly matched by a
same-source-choice wrong-row index control.

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

Packet contract:

- generated answer value/index packet;
- average candidate-index bits per row: `1.688`;
- average candidate-index bytes per row: `0.211`;
- average visible-answer text bytes per row: `1.656`;
- cacheline-rounded bytes per row: `64.0`;
- source-private: `false`;
- teacher-derived and not deployable: `true`.

## Interpretation

This is a useful upper-bound and leakage audit, not a method. The value packet,
same-byte visible answer text, and generated-answer candidate-index packet are
identical on the row set. The same-source-choice wrong-row control reaches
`15/32` and `9/10` clean rows, showing that much of the candidate-index surface
is just a source-choice artifact.

## Decision

Do not promote generated-answer value/index packets as LatentWire. The
generated-answer-aligned branch must now mean pre-answer state, source-side
teacher distillation, or another target that is not equivalent to revealing the
answer.

## Next Gate

Promote a pre-answer C2C-state or source-side distillation gate:

- no teacher-generated prefix;
- no generated-answer value or index packet;
- target must be a pre-answer hidden/KV/control signal or a source-derived
  predictor of the dense-teacher behavioral delta;
- first local implementation should locate the generated final-answer onset and
  summarize only state/logit evidence before that onset;
- mandatory controls include target-only, source-alone, text-to-text,
  wrong-row, same-source-choice wrong-row, candidate roll/derangement,
  same-length wrong-row, label-shuffle, same-byte visible text, post-answer
  extraction, answer-token masking, and source-index/rank/score analogues when
  meaningful.
