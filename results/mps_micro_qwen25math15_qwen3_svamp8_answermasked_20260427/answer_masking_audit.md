# Source-Surface Answer-Masking Audit

- date: `2026-04-27`
- git commit: `1e63d3b8a3cc914cceed712d9bc3100f26e172a0`
- candidate paths: `1`
- loaded surfaces with clean IDs: `1`
- skipped: `0`

| Rank | Surface | N | Clean Field | Clean | Clean In Pool | Answer-Unexplained Clean In Pool |
|---:|---|---:|---|---:|---:|---:|
| 1 | `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/source_contrastive_target_set.json` | 8 | `source_only` | 1 | 1 | 0 |

## Top Surface Details

### `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/source_contrastive_target_set.json`

- clean in pool IDs: `aee922049c757331`
- answer-unexplained clean in pool IDs: none

## Decision Rule

A surface is immediately useful for the next source-sidecar gate only if `answer_unexplained_clean_in_pool` is nonzero. If all reachable clean IDs are explained by source final or verified numeric answers, the surface is a candidate-headroom diagnostic rather than positive-method evidence.

## Command

```bash
scripts/audit_source_surface_answer_masking.py --results-root results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427 --date 2026-04-27 --output-json results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/answer_masking_audit.json --output-md results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/answer_masking_audit.md
```
