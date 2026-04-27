# Durable Source-Surface Ranking

- date: `2026-04-27`
- status: `primary_surface_selected`
- git commit: `f1ab3de32b87f4f50177e5678adee8bc99adf436`
- min clean source-only: `5`
- min numeric coverage: `0`

| Rank | Surface | Role | Decision | Clean | Source-only | Oracle gain | Target | Source | Notes |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `svamp70_live` | `primary_live` | `primary_ready` | 6 | 9 | 9 | 21/70 | 13/70 | canonical_live_surface |
| 2 | `svamp32_qwen25math` | `smoke_debug` | `weak_clean_headroom` | 4 | 5 | 5 | 8/32 | 6/32 | tiny_debug_surface |
| 3 | `svamp70_chal241_310` | `adjacent_falsifier` | `weak_clean_headroom` | 4 | 4 | 4 | 10/70 | 5/70 | best_adjacent_clean_surface |
| 4 | `svamp70_holdout` | `canonical_holdout` | `weak_clean_headroom` | 2 | 6 | 6 | 8/70 | 8/70 | canonical_holdout_surface |
| 5 | `svamp70_chal311_380` | `weak_candidate` | `weak_clean_headroom` | 2 | 3 | 3 | 21/70 | 8/70 | weak_adjacent_surface |
| 6 | `gsm70_qwen25math` | `weak_candidate` | `weak_clean_headroom` | 2 | 3 | 3 | 4/70 | 3/70 | weak_gsm_surface |
| 7 | `svamp32_qwen25math_instruct` | `weak_candidate` | `weak_clean_headroom` | 2 | 2 | 2 | 8/32 | 3/32 | weak_instruct_surface |
| 8 | `svamp70_chal171_240` | `weak_candidate` | `weak_clean_headroom` | 1 | 2 | 2 | 22/70 | 8/70 | weak_adjacent_surface |

## Top Clean IDs

- `svamp70_live`: `14bfbfc94f2c2e7b`, `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`, `bd9d8da923981d69`, `ce08a3a269bf0151`
- `svamp32_qwen25math`: `14bfbfc94f2c2e7b`, `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`
- `svamp70_chal241_310`: `0ee313c160b638a9`, `561daa750422c0e4`, `cd5623c80cf95da9`, `e90d2681e386fb04`

## Next Gate

Run the next method gate on `svamp70_live` first, then replay on the canonical holdout with identical controls.
