# Target-Side Candidate Headroom Audit

- date: `2026-04-27`
- status: `target_side_candidate_headroom_ranked`
- git commit: `76abab281d43ec30e509d58e750c3232f2adcb0e`

| Rank | Surface | Role | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `live_target_self_repair` | `target_self_repair_surface` | 21/70 | 13/70 | 47/70 | 26 | 0/3 | target+t2t+target_self_repair_no_source_pool |
| 2 | `canonical_live` | `canonical` | 21/70 | 13/70 | 33/70 | 12 | 0/6 | target+t2t_only |

## Clean IDs In Target-Side Pool

- `live_target_self_repair`: none
- `canonical_live`: none

## Next Gate

Use `live_target_self_repair` only if the next method can target 0 clean IDs already present in the target-side pool; otherwise switch to a new candidate-surface generator.
