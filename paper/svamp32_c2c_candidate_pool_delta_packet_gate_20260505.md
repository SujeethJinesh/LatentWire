# SVAMP32 C2C Candidate-Pool Delta Packet Gate

- date: `2026-05-05`
- status: `open_loop_c2c_candidate_score_distillation_ruled_out`
- artifact:
  `results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/candidate_pool_delta_packet_gate.json`
- manifest:
  `results/svamp32_c2c_candidate_pool_delta_packet_gate_mps_20260505/manifest.md`
- script:
  `scripts/analyze_svamp32_c2c_candidate_pool_delta_packet_gate.py`

## Question

Can we remove teacher-generated-prefix leakage and distill C2C into a tiny
candidate-pool packet by scoring public numeric answer candidates with the C2C
teacher and the target model?

## Result

No. The open-loop candidate-score surface is not faithful to C2C's generated
teacher behavior on this slice.

| Condition | Correct | Teacher-only | Clean | Helps | Harms |
|---|---:|---:|---:|---:|---:|
| matched | 3/32 | 0 | 0 | 0 | 3 |
| target_only | 6/32 | 1 | 1 | 0 | 0 |
| zero_delta | 6/32 | 1 | 1 | 0 | 0 |
| row_shuffle | 7/32 | 1 | 1 | 1 | 0 |
| same_top_wrong_row | 6/32 | 1 | 1 | 0 | 0 |
| candidate_roll | 8/32 | 2 | 2 | 2 | 0 |
| candidate_derangement | 4/32 | 1 | 1 | 0 | 2 |
| coeff_shuffle | 8/32 | 2 | 2 | 2 | 0 |
| coeff_sign_flip | 10/32 | 3 | 3 | 4 | 0 |
| target_derived_packet | 6/32 | 1 | 1 | 0 | 0 |
| teacher_top_index | 3/32 | 0 | 0 | 0 | 3 |

Packet contract:

- format: open-loop public-candidate C2C-minus-target z-score delta;
- coefficient bits: `4`;
- scale bits per row: `8`;
- average packet bytes per row: `2.875`;
- cacheline-rounded average bytes per row: `64.00`;
- source-state private: `true`, but teacher-derived and not deployable;
- candidate exposure: quantized score deltas over public numeric candidates.

## Interpretation

This gate fixed the previous teacher-prefix leakage, but exposed a different
failure: C2C's short-answer candidate likelihood surface does not match its
generative behavior. The dense teacher's generated answers reach `16/32`, but
the open-loop `teacher_top_index` candidate scorer reaches only `3/32` and
recovers no clean teacher-only rows.

A quick untracked sensitivity check with a `#### {answer}` continuation template
also failed: matched `4/32`, target-only `7/32`, teacher-top `4/32`, and no
clean matched rows. That weakens answer-continuation scoring as the next
C2C-distillation route.

## Decision

Do not train a source predictor for this exact candidate-score packet family.
The next C2C-distillation route should distill the generated-answer behavior
directly, or use hidden/KV state captured before the answer decision, instead
of short numeric continuation likelihoods.

## Next Gate

Promote a generated-answer-aligned C2C distillation gate:

- teacher label: C2C generated numeric answer or C2C-vs-target generated-answer
  delta, not short-answer likelihood;
- receiver input: no teacher prefix text;
- packet: compact candidate label/code or hidden/KV trace before answer
  emission;
- controls: target-only, source-alone, text-to-text, wrong-row,
  same-source-choice wrong-row, candidate roll/derangement, source-index/rank,
  source-score, and same-byte visible text;
- pass bar: recover at least two clean C2C-only rows beyond all controls.

## Lay Explanation

We tried asking the dense C2C model to score possible numeric answers directly.
That did not behave like the C2C model's normal step-by-step generation. So this
packet was distilling the wrong teacher signal.
