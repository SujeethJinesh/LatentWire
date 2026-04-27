# Target-Side Candidate Headroom Audit

- date: `2026-04-27`
- status: `target_side_candidate_headroom_ranked`
- git commit: `5feb0e05568c65cc44ff1aadec4973f52b0ccf82`

| Rank | Surface | Role | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `svamp32_full32_s8` | `target_no_source_full32` | 8/32 | 6/32 | 18/32 | 10 | 0/2 | 8_target_samples_per_id |

## Clean IDs In Target-Side Pool

- `svamp32_full32_s8`: none

## Next Gate

Use `svamp32_full32_s8` only if the next method can target 0 clean IDs already present in the target-side pool; otherwise switch to a new candidate-surface generator.
