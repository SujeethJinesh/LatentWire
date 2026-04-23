# Source-Headroom Surface Scan

- date: `2026-04-23`
- min source-only threshold: `5`

| Surface | Status | Target | Source | Source-only | Oracle | ID source | Note |
|---|---|---:|---:|---:|---:|---|---|
| fresh_svamp32_source | `weak_source_complementary_surface` | 8/32 | 5/32 | 3 | 11/32 | example_id/example_id | fresh_svamp32_source |
| fresh_svamp32_t2t | `weak_source_complementary_surface` | 8/32 | 2/32 | 1 | 9/32 | example_id/example_id | fresh_svamp32_text |
| fresh_svamp32_c2c | `strong_source_complementary_surface` | 8/32 | 16/32 | 10 | 18/32 | example_id/example_id | fresh_svamp32_c2c |

## Ranked Decision

- `fresh_svamp32_c2c`: status=`strong_source_complementary_surface`, source_only=`10`, oracle=`18/32`, strict_ids=`True`
- `fresh_svamp32_source`: status=`weak_source_complementary_surface`, source_only=`3`, oracle=`11/32`, strict_ids=`True`
- `fresh_svamp32_t2t`: status=`weak_source_complementary_surface`, source_only=`1`, oracle=`9/32`, strict_ids=`True`

## Artifact Paths

- `fresh_svamp32_source` target `results/svamp_exactid_baselines32_20260423/target_alone.jsonl` (target_alone); source `results/svamp_exactid_baselines32_20260423/source_alone.jsonl` (source_alone)
- `fresh_svamp32_t2t` target `results/svamp_exactid_baselines32_20260423/target_alone.jsonl` (target_alone); source `results/svamp_exactid_baselines32_20260423/text_to_text.jsonl` (text_to_text)
- `fresh_svamp32_c2c` target `results/svamp_exactid_baselines32_20260423/target_alone.jsonl` (target_alone); source `results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl` (c2c_generate)
