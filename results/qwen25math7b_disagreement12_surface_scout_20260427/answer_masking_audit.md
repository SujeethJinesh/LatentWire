# Source-Surface Answer-Masking Audit

- date: `2026-04-27`
- git commit: `68aae9003216437ec128bdfe36a672631f2bf420`
- candidate paths: `1`
- loaded surfaces with clean IDs: `1`
- skipped: `0`

| Rank | Surface | N | Clean Field | Clean | Clean In Pool | Answer-Unexplained Clean In Pool |
|---:|---|---:|---|---:|---:|---:|
| 1 | `results/qwen25math7b_disagreement12_surface_scout_20260427/source_contrastive_target_set.json` | 12 | `clean_source_only` | 5 | 3 | 0 |

## Top Surface Details

### `results/qwen25math7b_disagreement12_surface_scout_20260427/source_contrastive_target_set.json`

- clean in pool IDs: `561daa750422c0e4`, `ab1e71e8928661d0`, `daea537474de16ac`
- answer-unexplained clean in pool IDs: none

## Decision Rule

A surface is immediately useful for the next source-sidecar gate only if `answer_unexplained_clean_in_pool` is nonzero. If all reachable clean IDs are explained by source final or verified numeric answers, the surface is a candidate-headroom diagnostic rather than positive-method evidence.

## Command

```bash
scripts/audit_source_surface_answer_masking.py --results-root results/qwen25math7b_disagreement12_surface_scout_20260427 --date 2026-04-27 --output-json results/qwen25math7b_disagreement12_surface_scout_20260427/answer_masking_audit.json --output-md results/qwen25math7b_disagreement12_surface_scout_20260427/answer_masking_audit.md
```
