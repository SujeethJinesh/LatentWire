# Target-Side Candidate Headroom Audit

- date: `2026-04-27`
- status: `target_side_candidate_headroom_ranked`
- git commit: `4aaa23d7d2a3dbbabccadfbfbe399e56d66bb0d3`

| Rank | Surface | Role | Target | Source | Target-Side Oracle | Oracle Gain | Clean In Pool | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `svamp70_holdout` | `old_holdout` | 8/70 | 8/70 | 28/70 | 20 | 2/2 | weak_one_id_signal |
| 2 | `svamp70_chal171_240` | `next_candidate` | 22/70 | 8/70 | 39/70 | 17 | 1/1 | adjacent_weak_surface |
| 3 | `svamp70_chal241_310` | `next_candidate` | 10/70 | 5/70 | 23/70 | 13 | 1/4 | adjacent_clean_surface |
| 4 | `svamp70_chal311_380` | `next_candidate` | 21/70 | 8/70 | 34/70 | 13 | 0/2 | adjacent_weak_surface |
| 5 | `svamp70_live` | `old_live` | 21/70 | 13/70 | 33/70 | 12 | 0/6 | canonical_sidecar_saturated |
| 6 | `gsm70_qwen25math` | `next_candidate` | 4/70 | 3/70 | 14/70 | 10 | 0/2 | gsm_surface |

## Clean IDs In Target-Side Pool

- `svamp70_holdout`: `ab1e71e8928661d0`, `daea537474de16ac`
- `svamp70_chal171_240`: `4157958051c69d70`
- `svamp70_chal241_310`: `561daa750422c0e4`
- `svamp70_chal311_380`: none
- `svamp70_live`: none
- `gsm70_qwen25math`: none

## Next Gate

Use `svamp70_holdout` only if the next method can target 2 clean IDs already present in the target-side pool; otherwise switch to a new candidate-surface generator.
