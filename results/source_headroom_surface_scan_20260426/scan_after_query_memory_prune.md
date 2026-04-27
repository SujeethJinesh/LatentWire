# Source-Headroom Surface Scan

- date: `2026-04-26`
- min source-only threshold: `6`

| Surface | Status | Target | Source | Source-only | Oracle | ID source | Note |
|---|---|---:|---:|---:|---:|---|---|
| svamp70_live_source | `strong_source_complementary_surface` | 21/70 | 13/70 | 9 | 30/70 | example_id/example_id | qwen25math_qwen3_svamp70_live_source |
| svamp70_holdout_source | `strong_source_complementary_surface` | 8/70 | 8/70 | 6 | 14/70 | example_id/example_id | qwen25math_qwen3_svamp70_holdout_source |
| svamp70_chal171_source | `weak_source_complementary_surface` | 22/70 | 8/70 | 2 | 24/70 | example_id/example_id | qwen25math_qwen3_svamp70_chal171_240_source |
| svamp70_chal241_source | `weak_source_complementary_surface` | 10/70 | 5/70 | 4 | 14/70 | example_id/example_id | qwen25math_qwen3_svamp70_chal241_310_source |
| svamp70_chal311_source | `weak_source_complementary_surface` | 21/70 | 8/70 | 3 | 24/70 | example_id/example_id | qwen25math_qwen3_svamp70_chal311_380_source |
| gsm70_math_source | `weak_source_complementary_surface` | 4/70 | 3/70 | 3 | 7/70 | example_id/example_id | qwen25math_qwen3_gsm70_source |
| svamp32_math_chat_source | `weak_source_complementary_surface` | 8/32 | 6/32 | 5 | 13/32 | example_id/example_id | qwen25math_qwen3_svamp32_chat_source |

## Ranked Decision

- `svamp70_live_source`: status=`strong_source_complementary_surface`, source_only=`9`, oracle=`30/70`, strict_ids=`True`
- `svamp70_holdout_source`: status=`strong_source_complementary_surface`, source_only=`6`, oracle=`14/70`, strict_ids=`True`
- `svamp32_math_chat_source`: status=`weak_source_complementary_surface`, source_only=`5`, oracle=`13/32`, strict_ids=`True`
- `svamp70_chal241_source`: status=`weak_source_complementary_surface`, source_only=`4`, oracle=`14/70`, strict_ids=`True`
- `gsm70_math_source`: status=`weak_source_complementary_surface`, source_only=`3`, oracle=`7/70`, strict_ids=`True`

## Artifact Paths

- `svamp70_live_source` target `results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl` (source_alone)
- `svamp70_holdout_source` target `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl` (source_alone)
- `svamp70_chal171_source` target `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl` (source_alone)
- `svamp70_chal241_source` target `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl` (source_alone)
- `svamp70_chal311_source` target `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl` (source_alone)
- `gsm70_math_source` target `results/qwen25math_qwen3_gsm70_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_alone.jsonl` (source_alone)
- `svamp32_math_chat_source` target `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl` (target_alone); source `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl` (source_alone)
