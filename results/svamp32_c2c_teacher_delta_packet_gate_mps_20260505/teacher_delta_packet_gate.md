# SVAMP32 C2C Teacher-Delta Packet Gate

- date: `2026-05-05`
- status: `teacher_delta_packet_capacity_fails_controls`
- reference rows: `32`
- packet top-k: `4`
- coeff bits: `4`
- average packet bytes per row: `794.22`
- clean source-necessary IDs: `0`

## Summary

| Condition | Correct | Teacher-only | Clean | Exact teacher replay | Token match |
|---|---:|---:|---:|---:|---:|
| `matched` | 14/32 | 8 | 8 | 0 | 0.741 |
| `target_only` | 14/32 | 8 | 8 | 0 | 0.739 |
| `zero_delta` | 14/32 | 8 | 8 | 0 | 0.739 |
| `row_shuffle` | 13/32 | 7 | 7 | 0 | 0.737 |
| `atom_shuffle` | 14/32 | 8 | 8 | 0 | 0.741 |
| `coeff_shuffle` | 14/32 | 8 | 8 | 0 | 0.741 |
| `coeff_sign_flip` | 14/32 | 8 | 8 | 0 | 0.738 |

## Decision

Do not train a source predictor for this exact packet format; the matched packet does not separate from destructive controls.

## Claim Boundary

- This is a C2C-teacher packet-capacity gate, not a deployable source-causal receiver.
- It transmits sparse token-logit deltas from the dense teacher and therefore cannot by itself prove source-private latent communication.
- Target-only under the teacher-generated prefix is a strong target-cache control; matching that control is not source-causal evidence.
- A pass would only justify training a source-side predictor for the same packet; a fail rules out this packet format.
