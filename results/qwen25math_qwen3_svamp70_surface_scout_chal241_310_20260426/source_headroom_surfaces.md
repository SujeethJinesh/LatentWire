# Source-Headroom Surface Scan

- date: `2026-04-26`
- min source-only threshold: `6`

| Surface | Status | Target | Source | Source-only | Oracle | ID source | Note |
|---|---|---:|---:|---:|---:|---|---|
| svamp70_live | `strong_source_complementary_surface` | 21/70 | 13/70 | 9 | 30/70 | example_id/example_id | original_positive_surface |
| svamp70_holdout | `strong_source_complementary_surface` | 8/70 | 8/70 | 6 | 14/70 | example_id/example_id | guard_failed_holdout |
| svamp70_chal171_240 | `weak_source_complementary_surface` | 22/70 | 8/70 | 2 | 24/70 | example_id/example_id | weak_surface |
| svamp70_chal241_310 | `weak_source_complementary_surface` | 10/70 | 5/70 | 4 | 14/70 | example_id/example_id | clean4_but_source_only4 |

## Ranked Decision

- `svamp70_live`: status=`strong_source_complementary_surface`, source_only=`9`, oracle=`30/70`, strict_ids=`True`
- `svamp70_holdout`: status=`strong_source_complementary_surface`, source_only=`6`, oracle=`14/70`, strict_ids=`True`
- `svamp70_chal241_310`: status=`weak_source_complementary_surface`, source_only=`4`, oracle=`14/70`, strict_ids=`True`
- `svamp70_chal171_240`: status=`weak_source_complementary_surface`, source_only=`2`, oracle=`24/70`, strict_ids=`True`

## Artifact Paths

- `svamp70_live` target `results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl` (source_alone)
- `svamp70_holdout` target `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl` (source_alone)
- `svamp70_chal171_240` target `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl` (source_alone)
- `svamp70_chal241_310` target `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl` (source_alone)
