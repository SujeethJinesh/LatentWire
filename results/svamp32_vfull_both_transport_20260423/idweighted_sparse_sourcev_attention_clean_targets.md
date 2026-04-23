# SVAMP32 Gate Sweep Clean-Target Readout

- date: `2026-04-23`
- status: `no_matched_gate_candidate_for_controls`
- target: `8/32`
- C2C teacher: `16/32`
- teacher-only IDs: `10`
- clean residual targets: `6`

## Rows

| Method | Status | Correct | Clean residual | Teacher-only | Delta vs self-repair | Target losses | Oracle self+clean bound |
|---|---|---:|---:|---:|---:|---:|---:|
| rotalign_kv_gate_0.15 | `matched_candidate_below_clean_gate` | 10/32 | 1 | 2 | -4 | 1 | 15/32 |
| rotalign_kv_gate_0.17 | `matched_candidate_below_clean_gate` | 10/32 | 1 | 2 | -4 | 2 | 15/32 |
| rotalign_kv_gate_0.12 | `matched_candidate_below_clean_gate` | 8/32 | 0 | 1 | -6 | 2 | 14/32 |

## Clean IDs By Row

- `rotalign_kv_gate_0.15`: `aee922049c757331`
- `rotalign_kv_gate_0.17`: `aee922049c757331`
- `rotalign_kv_gate_0.12`: none
