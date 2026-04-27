# Target-Side Candidate Headroom Audit

- date: `2026-04-27`
- status: `target_side_candidate_headroom_ranked`
- git commit: `6a09be6d62e3804af408cad178b8274b965b7da6`

| Rank | Surface | Role | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `sampled_clean3` | `sampled_target_pool` | 0/3 | 3/3 | 2/3 | 2 | 2/3 | math7b_svamp70_residual_clean3_target_only_samples16 |

## Clean IDs In Target-Side Pool

- `sampled_clean3`: `14bfbfc94f2c2e7b`, `a07cd6cc8f1c832e`

## Next Gate

Use `sampled_clean3` only if the next method can target 2 clean IDs already present in the target-side pool; otherwise switch to a new candidate-surface generator.
