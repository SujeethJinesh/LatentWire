# SVAMP32 Target-Self-Repair Paper Gate

- date: `2026-04-23`
- status: `no_candidate_passes_target_self_repair_gate`
- target: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- target_self_repair C2C-only recovered: `3/10`

## Gate

- minimum correct: `16/32`
- minimum delta vs target_self_repair: `+1`
- minimum C2C-only recovered: `5/10`
- minimum C2C-only unique vs target_self_repair: `2`
- maximum target losses: `1`
- maximum retained by any source control: `1`

## Candidate Decisions

| Candidate | Status | Correct | Delta vs self-repair | C2C-only | Unique vs self-repair | Max source-control retained | Target losses | Failing criteria |
|---|---|---:|---:|---:|---:|---:|---:|---|
| query_pool_matched | `fails_paper_gate` | 9/32 | -5 | 1 | 1 | 1 | 1 | min_correct, beats_target_self_repair, min_teacher_only, min_unique_vs_target_self_repair |

## Criteria Detail

### query_pool_matched
- `target_self_repair_present`: `pass`
- `source_controls_present`: `pass`
- `min_correct`: `fail`
- `beats_target_self_repair`: `fail`
- `min_teacher_only`: `fail`
- `min_unique_vs_target_self_repair`: `fail`
- `max_losses_vs_target`: `pass`
- `max_source_control_retained`: `pass`
- C2C-only unique vs target_self_repair: `575d7e83d84c1e67`
- C2C-only retained by source controls: `575d7e83d84c1e67`
