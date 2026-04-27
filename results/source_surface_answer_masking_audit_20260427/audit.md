# Source-Surface Answer-Masking Audit

- date: `2026-04-27`
- git commit: `afbf022b1e7e1e96c1bd76f72c28e6f538f73abf`
- candidate paths: `15`
- loaded surfaces with clean IDs: `12`
- skipped: `3`

| Rank | Surface | N | Clean Field | Clean | Clean In Pool | Answer-Unexplained Clean In Pool |
|---:|---|---:|---|---:|---:|---:|
| 1 | `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json` | 70 | `clean_source_only` | 2 | 2 | 0 |
| 2 | `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json` | 70 | `clean_source_only` | 1 | 1 | 0 |
| 3 | `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json` | 70 | `clean_source_only` | 4 | 1 | 0 |
| 4 | `results/source_contrastive_target_sets_20260426/svamp70_c2c_vs_process_repair_target_set.json` | 70 | `clean_source_only` | 10 | 1 | 0 |
| 5 | `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json` | 3 | `clean_source_only` | 3 | 1 | 0 |
| 6 | `results/no_source_candidate_surface_20260427/source_contrastive_target_set.json` | 70 | `clean_source_only` | 3 | 0 | 0 |
| 7 | `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json` | 70 | `clean_source_only` | 2 | 0 | 0 |
| 8 | `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json` | 70 | `clean_source_only` | 6 | 0 | 0 |
| 9 | `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json` | 70 | `clean_source_only` | 2 | 0 | 0 |
| 10 | `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json` | 32 | `clean_source_only` | 4 | 0 | 0 |
| 11 | `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_contrastive_target_set.json` | 32 | `clean_source_only` | 2 | 0 | 0 |
| 12 | `results/target_self_repair_candidate_surface_20260427/live_target_self_repair_target_set.json` | 70 | `clean_source_only` | 3 | 0 | 0 |

## Top Surface Details

### `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json`

- clean in pool IDs: `ab1e71e8928661d0`, `daea537474de16ac`
- answer-unexplained clean in pool IDs: none

### `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json`

- clean in pool IDs: `4157958051c69d70`
- answer-unexplained clean in pool IDs: none

### `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json`

- clean in pool IDs: `561daa750422c0e4`
- answer-unexplained clean in pool IDs: none

### `results/source_contrastive_target_sets_20260426/svamp70_c2c_vs_process_repair_target_set.json`

- clean in pool IDs: `1d50b408c8f5cd2c`
- answer-unexplained clean in pool IDs: none

### `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json`

- clean in pool IDs: `14bfbfc94f2c2e7b`
- answer-unexplained clean in pool IDs: none

### `results/no_source_candidate_surface_20260427/source_contrastive_target_set.json`

- clean in pool IDs: none
- answer-unexplained clean in pool IDs: none

### `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json`

- clean in pool IDs: none
- answer-unexplained clean in pool IDs: none

### `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json`

- clean in pool IDs: none
- answer-unexplained clean in pool IDs: none

### `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json`

- clean in pool IDs: none
- answer-unexplained clean in pool IDs: none

### `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json`

- clean in pool IDs: none
- answer-unexplained clean in pool IDs: none

## Decision Rule

A surface is immediately useful for the next source-sidecar gate only if `answer_unexplained_clean_in_pool` is nonzero. If all reachable clean IDs are explained by source final or verified numeric answers, the surface is a candidate-headroom diagnostic rather than positive-method evidence.

## Command

```bash
scripts/audit_source_surface_answer_masking.py --results-root results --date 2026-04-27 --output-json results/source_surface_answer_masking_audit_20260427/audit.json --output-md results/source_surface_answer_masking_audit_20260427/audit.md
```
