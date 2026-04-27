# Target-Side Candidate Headroom Audit

- date: `2026-04-27`
- status: `target_side_candidate_headroom_ranked`
- git commit: `afbf022b1e7e1e96c1bd76f72c28e6f538f73abf`

| Rank | Surface | Role | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `sampled_clean3` | `sampled_target_pool` | 0/3 | 3/3 | 1/3 | 1 | 1/3 | zero_source_pool_plus_target_samples_clean3 |

## Clean IDs In Target-Side Pool

- `sampled_clean3`: `14bfbfc94f2c2e7b`

## Next Gate

Use `sampled_clean3` only if the next method can target 1 clean IDs already present in the target-side pool; otherwise switch to a new candidate-surface generator.
