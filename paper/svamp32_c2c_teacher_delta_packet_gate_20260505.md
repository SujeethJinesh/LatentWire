# SVAMP32 C2C Teacher-Delta Packet Gate

- date: `2026-05-05`
- status: `teacher_forced_full_vocab_delta_packet_ruled_out`
- artifact:
  `results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/teacher_delta_packet_gate.json`
- manifest:
  `results/svamp32_c2c_teacher_delta_packet_gate_mps_20260505/manifest.md`
- script:
  `scripts/analyze_svamp32_c2c_teacher_delta_packet_gate.py`

## Question

Does a sparse top-k packet of C2C teacher-minus-target token-logit deltas recover
the dense C2C teacher's useful behavior beyond target-only, zero-delta,
wrong-row, atom-shuffle, coefficient-shuffle, and sign-flip controls?

## Result

The packet does not separate from the strongest target-cache controls.

| Condition | Correct | Teacher-only | Clean | Exact replay | Token match |
|---|---:|---:|---:|---:|---:|
| matched | 14/32 | 8 | 8 | 0 | 0.741 |
| target_only | 14/32 | 8 | 8 | 0 | 0.739 |
| zero_delta | 14/32 | 8 | 8 | 0 | 0.739 |
| row_shuffle | 13/32 | 7 | 7 | 0 | 0.737 |
| atom_shuffle | 14/32 | 8 | 8 | 0 | 0.741 |
| coeff_shuffle | 14/32 | 8 | 8 | 0 | 0.741 |
| coeff_sign_flip | 14/32 | 8 | 8 | 0 | 0.738 |

Packet contract:

- format: teacher-forced sparse full-vocab C2C-minus-target logit delta;
- top-k: `4`;
- coefficient bits: `4`;
- average packet bytes per row: `794.22`;
- cacheline-rounded average bytes per row: `832.00`;
- source-private: `false`, because the packet is computed from dense C2C
  teacher logits.

## Interpretation

The apparent matched recovery is explained by conditioning the target on the
teacher-generated prefix. Target-only under that teacher prefix already recovers
the same `14/32` and the same `8` clean teacher-only rows. The matched packet
adds no clean source-necessary row, and atom/coeff destructive controls do not
weaken the outcome.

This rules out training a deployable source predictor for this exact packet
format. It also shows that the next C2C-distillation gate should avoid
teacher-forced prefix leakage and should operate on a compact answer/candidate
decision surface or an open-loop receiver.

## Next Gate

Promote a tighter candidate-pool or open-loop C2C teacher-delta packet gate:

- no teacher-generated prefix as target-only context;
- packet over answer/candidate evidence rather than full vocabulary;
- byte budget at the 1-8 byte scale;
- controls: target-only, zero packet, wrong-row, same-source-choice wrong-row,
  candidate roll/derangement, source-index/rank/score, and same-byte visible
  text;
- pass bar: recover at least two clean C2C-only rows beyond all controls.

## Lay Explanation

We tried sending a tiny-ish list of next-token nudges copied from the full C2C
teacher. It looked like it helped only because the target was already reading
the teacher's previous words. When the nudge was removed, shuffled, or broken,
the result barely changed, so this is not real source communication.
