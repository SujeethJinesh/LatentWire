# Source-Headroom Surface Scan

- date: `2026-04-26`
- min source-only threshold: `5`

| Surface | Status | Target | Source | Source-only | Oracle | ID source | Note |
|---|---|---:|---:|---:|---:|---|---|
| qwen25math_qwen3_svamp32_chat | `strong_source_complementary_surface` | 8/32 | 6/32 | 5 | 13/32 | example_id/example_id | best_current_c2c_surface |
| qwen25math_instruct_qwen3_svamp32 | `weak_source_complementary_surface` | 8/32 | 3/32 | 2 | 10/32 | example_id/example_id | instruct_source_variant |
| qwen25math_qwen3_gsm70 | `weak_source_complementary_surface` | 4/70 | 3/70 | 3 | 7/70 | example_id/example_id | gsm70_source_surface |
| qwen25math_qwen3_svamp70 | `strong_source_complementary_surface` | 21/70 | 13/70 | 9 | 30/70 | example_id/example_id | svamp70_source_sidecar_surface |
| qwen25math_qwen3_svamp70_holdout | `strong_source_complementary_surface` | 8/70 | 8/70 | 6 | 14/70 | example_id/example_id | svamp70_holdout_surface |
| deepseek_qwen_svamp32 | `weak_source_complementary_surface` | 8/32 | 5/32 | 1 | 9/32 | example_id/example_id | deepseek_source_variant |

## Ranked Decision

- `qwen25math_qwen3_svamp70`: status=`strong_source_complementary_surface`, source_only=`9`, oracle=`30/70`, strict_ids=`True`
- `qwen25math_qwen3_svamp70_holdout`: status=`strong_source_complementary_surface`, source_only=`6`, oracle=`14/70`, strict_ids=`True`
- `qwen25math_qwen3_svamp32_chat`: status=`strong_source_complementary_surface`, source_only=`5`, oracle=`13/32`, strict_ids=`True`
- `qwen25math_qwen3_gsm70`: status=`weak_source_complementary_surface`, source_only=`3`, oracle=`7/70`, strict_ids=`True`
- `qwen25math_instruct_qwen3_svamp32`: status=`weak_source_complementary_surface`, source_only=`2`, oracle=`10/32`, strict_ids=`True`

## Artifact Paths

- `qwen25math_qwen3_svamp32_chat` target `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl` (target_alone); source `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl` (source_alone)
- `qwen25math_instruct_qwen3_svamp32` target `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/target_alone.jsonl` (target_alone); source `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_alone.jsonl` (source_alone)
- `qwen25math_qwen3_gsm70` target `results/qwen25math_qwen3_gsm70_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_alone.jsonl` (source_alone)
- `qwen25math_qwen3_svamp70` target `results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl` (source_alone)
- `qwen25math_qwen3_svamp70_holdout` target `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl` (target_alone); source `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl` (source_alone)
- `deepseek_qwen_svamp32` target `results/surface_scout_deepseek_qwen_svamp32_20260426/target_alone.jsonl` (target_alone); source `results/surface_scout_deepseek_qwen_svamp32_20260426/source_alone.jsonl` (source_alone)
