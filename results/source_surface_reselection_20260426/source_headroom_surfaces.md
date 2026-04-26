# Source-Headroom Surface Scan

- date: `2026-04-26`
- min source-only threshold: `6`

| Surface | Status | Target | Source | Source-only | Oracle | ID source | Note |
|---|---|---:|---:|---:|---:|---|---|
| svamp70_live | `strong_source_complementary_surface` | 21/70 | 13/70 | 9 | 30/70 | example_id/example_id | strong_live_surface |
| svamp70_holdout | `strong_source_complementary_surface` | 8/70 | 8/70 | 6 | 14/70 | example_id/example_id | strong_holdout_surface |
| svamp70_chal171_240 | `weak_source_complementary_surface` | 22/70 | 8/70 | 2 | 24/70 | example_id/example_id | weak_oracle_surface |
| svamp70_chal241_310 | `weak_source_complementary_surface` | 10/70 | 5/70 | 4 | 14/70 | example_id/example_id | weak_clean_source_surface |
| svamp32_qwen25math | `weak_source_complementary_surface` | 8/32 | 6/32 | 5 | 13/32 | example_id/example_id | failed_prefix_connectors |
| svamp32_qwen25math_instruct | `weak_source_complementary_surface` | 8/32 | 3/32 | 2 | 10/32 | example_id/example_id | weak_source_surface |
| svamp32_deepseek | `weak_source_complementary_surface` | 8/32 | 5/32 | 1 | 9/32 | example_id/example_id | weak_source_surface |
| gsm70_qwen25math | `weak_source_complementary_surface` | 4/70 | 3/70 | 3 | 7/70 | example_id/example_id | weak_gsm_surface |

## Ranked Decision

- `svamp70_live`: status=`strong_source_complementary_surface`, source_only=`9`, oracle=`30/70`, strict_ids=`True`
- `svamp70_holdout`: status=`strong_source_complementary_surface`, source_only=`6`, oracle=`14/70`, strict_ids=`True`
- `svamp32_qwen25math`: status=`weak_source_complementary_surface`, source_only=`5`, oracle=`13/32`, strict_ids=`True`
- `svamp70_chal241_310`: status=`weak_source_complementary_surface`, source_only=`4`, oracle=`14/70`, strict_ids=`True`
- `gsm70_qwen25math`: status=`weak_source_complementary_surface`, source_only=`3`, oracle=`7/70`, strict_ids=`True`

## Artifact Paths

- `svamp70_live` target `results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl` (source_alone)
- `svamp70_holdout` target `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl` (source_alone)
- `svamp70_chal171_240` target `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl` (source_alone)
- `svamp70_chal241_310` target `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl` (source_alone)
- `svamp32_qwen25math` target `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl` (target_alone); source `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl` (source_alone)
- `svamp32_qwen25math_instruct` target `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/target_alone.jsonl` (target_alone); source `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_alone.jsonl` (source_alone)
- `svamp32_deepseek` target `results/surface_scout_deepseek_qwen_svamp32_20260426/target_alone.jsonl` (target_alone); source `results/surface_scout_deepseek_qwen_svamp32_20260426/source_alone.jsonl` (source_alone)
- `gsm70_qwen25math` target `results/qwen25math_qwen3_gsm70_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_alone.jsonl` (source_alone)
