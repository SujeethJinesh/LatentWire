# Source-Surface Answer-Masking Audit

- date: `2026-04-27`
- git commit: `3b562ce79d5415c1b8cf8394089a37e55617bd5e`
- candidate paths: `1`
- loaded surfaces with clean IDs: `0`
- skipped: `0`

| Rank | Surface | N | Clean Field | Clean | Clean In Pool | Answer-Unexplained Clean In Pool |
|---:|---|---:|---|---:|---:|---:|

## Top Surface Details

## Decision Rule

A surface is immediately useful for the next source-sidecar gate only if `answer_unexplained_clean_in_pool` is nonzero. If all reachable clean IDs are explained by source final or verified numeric answers, the surface is a candidate-headroom diagnostic rather than positive-method evidence.

## Command

```bash
scripts/audit_source_surface_answer_masking.py --results-root results/fresh_cpu_svamp8b_answernull_20260427 --date 2026-04-27 --output-json results/fresh_cpu_svamp8b_answernull_20260427/answer_masking_audit.json --output-md results/fresh_cpu_svamp8b_answernull_20260427/answer_masking_audit.md
```
