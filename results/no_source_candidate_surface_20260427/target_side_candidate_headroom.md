# Target-Side Candidate Headroom Audit

- date: `2026-04-27`
- status: `target_side_candidate_headroom_ranked`
- git commit: `76abab281d43ec30e509d58e750c3232f2adcb0e`

| Rank | Surface | Role | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `zero_source_candidate_surface` | `no_source_target_pool` | 21/70 | 13/70 | 48/70 | 27 | 0/3 | target+t2t+target_self+process_repair+zero_source_seed_pool |
| 2 | `target_self_surface` | `target_self_only` | 21/70 | 13/70 | 47/70 | 26 | 0/3 | target+t2t+target_self_repair |
| 3 | `canonical_live` | `canonical` | 21/70 | 13/70 | 33/70 | 12 | 0/6 | target+t2t_only |

## Clean IDs In Target-Side Pool

- `zero_source_candidate_surface`: none
- `target_self_surface`: none
- `canonical_live`: none

## Next Gate

Use `zero_source_candidate_surface` only if the next method can target 0 clean IDs already present in the target-side pool; otherwise switch to a new candidate-surface generator.
