# SVAMP32 Query Pool Transport Manifest

- date: `2026-04-23`
- gate: `query_pool_transport transfer on frozen SVAMP32 C2C-teacher surface`
- checkpoint: `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`
- materialized eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- repo fix: `runtime/source attention-derived position scores are resized to translated KV length before selection`
- focused tests: `./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py -k 'query_pool_transport or resize_position_scores or shuffle_examples_uses_mismatched_source_prompt' -q`
- test result: `5 passed`
- result: `candidate_teacher_recovery_explained_by_controls`

## Gate Sweep

| Gate | Correct | Note |
|---:|---:|---|
| `0.10` | `9/32` | only live row |
| `0.15` | `7/32` | below target |
| `0.25` | `6/32` | below target |

## Gate `0.10` Row Summary

- matched: `9/32`, wins `3e8a5691f5443495`, `575d7e83d84c1e67`, loss `c042f0a2949ff8e6`
- zero-source: `8/32`, wins `3e8a5691f5443495`, `575d7e83d84c1e67`, losses `de2a795ab37694af`, `c042f0a2949ff8e6`
- shuffled-source: `9/32`, wins `3e8a5691f5443495`, `575d7e83d84c1e67`, loss `c042f0a2949ff8e6`
- only teacher-only recovered ID: `575d7e83d84c1e67`
- that teacher-only ID is retained by zero-source, shuffled-source, and source-alone
- exact ordered ID parity: target / matched / zero-source / shuffled-source all `true`

## Artifact Hashes

- `live_gate_sweep.jsonl`
  - sha256: `b17bca44480e188620ab89e8da9c1888173df87f35ce8e1d0c7f9195c1154f79`
- `live_gate_sweep.jsonl.meta.json`
  - sha256: `20e611551a0aced69c1dbb3876b19acd18847322d8d9d29e0c5680318efb8b2b`
- `query_pool_transport_gate010_matched.jsonl`
  - sha256: `db0fe1bd0ff37a89f39cd0e5d2e25940b2efaadb4dcbddb0523a3df145f55626`
- `query_pool_transport_gate010_matched.jsonl.meta.json`
  - sha256: `9485ba5c5cd975211e5845b1610ade75267bf8986ba239c1946c6f5484987a34`
- `query_pool_transport_gate010_zero_source.jsonl`
  - sha256: `7017bc7e915f81eb0592000d15ac1d594845093b2bcc8e4ab1660aac6e2fa9a2`
- `query_pool_transport_gate010_zero_source.jsonl.meta.json`
  - sha256: `3b3397074814a7ce5cd5a563dd887d65249bc34974cf0924043ed5026a850305`
- `query_pool_transport_gate010_shuffled_source_salt1.jsonl`
  - sha256: `0a04c4f19a6746778c867549a7b8ffcbd8bb396338101164c0c05fef5ebbc03b`
- `query_pool_transport_gate010_shuffled_source_salt1.jsonl.meta.json`
  - sha256: `30ba85a0dcd8084a0488c44757475ce7abcb4eddadf1c530d14074e84ce69153`
- `c2c_teacher_probe_gate010.json`
  - sha256: `05d5ffd9f23cd7765e42895735b423a30d2aab1daa901080ae19ad57d5778723`
- `c2c_teacher_probe_gate010.md`
  - sha256: `dfc6b9f4fea867b9a8a5b407c85ffe3c22084dbad5a1774a3762258761eaa0d9`
