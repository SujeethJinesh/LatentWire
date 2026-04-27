# Source-Surface Answer-Masking Audit

- date: `2026-04-27`
- git commit: `d5299139948b7c5ef594f7846d7f7ae75f9f0139`
- candidate paths: `1`
- loaded surfaces with clean IDs: `1`
- skipped: `0`

| Rank | Surface | N | Clean Field | Clean | Clean In Pool | Answer-Unexplained Clean In Pool |
|---:|---|---:|---|---:|---:|---:|
| 1 | `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json` | 70 | `clean_source_only` | 7 | 3 | 0 |

## Top Surface Details

### `results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json`

- clean in pool IDs: `4c84ebf42812703b`, `d64f6e35083ffe8c`, `de1bf4d142544e5b`
- answer-unexplained clean in pool IDs: none

## Decision Rule

A surface is immediately useful for the next source-sidecar gate only if `answer_unexplained_clean_in_pool` is nonzero. If all reachable clean IDs are explained by source final or verified numeric answers, the surface is a candidate-headroom diagnostic rather than positive-method evidence.

## Command

```bash
scripts/audit_source_surface_answer_masking.py --results-root results/qwen25_7b_qwen3_svamp70_surface_scout_20260427 --date 2026-04-27 --output-json results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/answer_masking_audit.json --output-md results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/answer_masking_audit.md
```
