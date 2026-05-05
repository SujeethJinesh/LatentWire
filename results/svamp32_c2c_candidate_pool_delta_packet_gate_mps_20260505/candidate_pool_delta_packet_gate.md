# SVAMP32 C2C Candidate-Pool Delta Packet Gate

- date: `2026-05-05`
- status: `c2c_candidate_pool_delta_packet_capacity_fails_controls`
- reference rows: `32`
- average candidate count: `3.75`
- average packet bytes per row: `2.88`
- clean source-necessary IDs: `0`

## Summary

| Condition | Correct | Teacher-only | Clean | Helps | Harms |
|---|---:|---:|---:|---:|---:|
| `matched` | 3/32 | 0 | 0 | 0 | 3 |
| `target_only` | 6/32 | 1 | 1 | 0 | 0 |
| `zero_delta` | 6/32 | 1 | 1 | 0 | 0 |
| `row_shuffle` | 7/32 | 1 | 1 | 1 | 0 |
| `same_top_wrong_row` | 6/32 | 1 | 1 | 0 | 0 |
| `candidate_roll` | 8/32 | 2 | 2 | 2 | 0 |
| `candidate_derangement` | 4/32 | 1 | 1 | 0 | 2 |
| `coeff_shuffle` | 8/32 | 2 | 2 | 2 | 0 |
| `coeff_sign_flip` | 10/32 | 3 | 3 | 4 | 0 |
| `target_derived_packet` | 6/32 | 1 | 1 | 0 | 0 |
| `teacher_top_index` | 3/32 | 0 | 0 | 0 | 3 |

## Decision

Do not train a source predictor for this exact candidate-delta packet unless a follow-up removes the winning control.

## Claim Boundary

- This is a dense-C2C-teacher capacity gate, not a deployable source-causal receiver.
- The target is not conditioned on the C2C teacher-generated prefix.
- The packet is derived from C2C teacher candidate scores and can expose answer-candidate preferences.
- A pass would justify training a source-side or representation-side predictor for the same candidate packet.
