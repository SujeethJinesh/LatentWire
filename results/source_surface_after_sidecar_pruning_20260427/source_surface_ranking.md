# Durable Source-Surface Ranking

- date: `2026-04-27`
- status: `primary_surface_selected`
- git commit: `4aaa23d7d2a3dbbabccadfbfbe399e56d66bb0d3`
- min clean source-only: `4`
- min numeric coverage: `0`

| Rank | Surface | Role | Decision | Clean | Source-only | Oracle gain | Target | Source | Notes |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `svamp70_live` | `old_live` | `primary_ready` | 6 | 9 | 9 | 21/70 | 13/70 | canonical_sidecar_saturated |
| 2 | `svamp70_chal241_310` | `next_candidate` | `primary_ready` | 4 | 4 | 4 | 10/70 | 5/70 | adjacent_clean_surface |
| 3 | `svamp70_holdout` | `old_holdout` | `weak_clean_headroom` | 2 | 6 | 6 | 8/70 | 8/70 | weak_one_id_signal |
| 4 | `svamp70_chal311_380` | `next_candidate` | `weak_clean_headroom` | 2 | 3 | 3 | 21/70 | 8/70 | adjacent_weak_surface |
| 5 | `gsm70_qwen25math` | `next_candidate` | `weak_clean_headroom` | 2 | 3 | 3 | 4/70 | 3/70 | gsm_surface |
| 6 | `svamp70_chal171_240` | `next_candidate` | `weak_clean_headroom` | 1 | 2 | 2 | 22/70 | 8/70 | adjacent_weak_surface |

## Top Clean IDs

- `svamp70_live`: `14bfbfc94f2c2e7b`, `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`, `bd9d8da923981d69`, `ce08a3a269bf0151`
- `svamp70_chal241_310`: `0ee313c160b638a9`, `561daa750422c0e4`, `cd5623c80cf95da9`, `e90d2681e386fb04`
- `svamp70_holdout`: `ab1e71e8928661d0`, `daea537474de16ac`

## Next Gate

Run the next method gate on `svamp70_live` first, then replay on the canonical holdout with identical controls.
