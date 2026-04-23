# SVAMP32 Query Pool Transport Manifest

- date: `2026-04-23`
- gate: `query_pool_transport transfer on frozen SVAMP32 C2C-teacher surface`
- checkpoint: `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`
- materialized eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- repo fix: `runtime/source attention-derived position scores are resized to translated KV length before selection`
- focused tests: `./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py -k 'query_pool_transport or resize_position_scores or shuffle_examples_uses_mismatched_source_prompt' -q`
- test result: `5 passed`
- result: `candidate_teacher_recovery_explained_by_controls`
- target-self-repair paper gate: `no_candidate_passes_target_self_repair_gate`
- innovation target-set status: `residual_headroom_available`
- clean-target paper gate: `no_candidate_passes_target_self_repair_gate`

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

## Target-Self-Repair Gate

- target_self_repair: `14/32`, `3/10` C2C-only recoveries, `0` target losses
- query_pool_matched: `9/32`, `1/10` C2C-only recoveries, `1` target loss
- delta versus target_self_repair: `-5`
- failing criteria: `min_correct`, `beats_target_self_repair`, `min_teacher_only`, `min_unique_vs_target_self_repair`
- retained by source controls: `575d7e83d84c1e67`

## Innovation Target Set

- clean residual C2C-only targets: `6`
- clean target IDs: `13cb77b698eeadb5`, `1d50b408c8f5cd2c`, `2de1549556000830`, `6e9745b37ab6fc45`, `aee922049c757331`, `e3ab8666238a289e`
- target_self_repair already recovers: `4c84ebf42812703b`, `4d780f825bb8541c`, `de1bf4d142544e5b`
- source/source-control explained: `575d7e83d84c1e67`
- target_self_repair plus C2C teacher oracle: `21/32`
- required clean residual wins if preserving target_self_repair: `2`

## Clean-Target Paper Gate

- clean residual target set present: `true`
- minimum clean residual C2C-only recovered: `2`
- minimum clean source-necessary recovered: `2`
- query_pool_matched clean residual recovered: `0/6`
- query_pool_matched clean source-necessary recovered: `0/6`
- additional failing criteria: `min_clean_residual_recovered`,
  `min_clean_source_necessary`

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
- `c2c_teacher_probe_gate010_with_target_repair.json`
  - sha256: `1967901c0062696eac924c9c30ce0316f89999e5b099c9f06f7ee059eb9b8dbc`
- `c2c_teacher_probe_gate010_with_target_repair.md`
  - sha256: `b0ae99127d5427a6c83ae1b01121e04c84b99ce57002ec216bc884acba47c2b7`
- `paper_gate_gate010_with_target_repair.json`
  - sha256: `3795cf64ec4eb96cda2cad3d41afa45021727662424a89cd27993cfb78fbb603`
- `paper_gate_gate010_with_target_repair.md`
  - sha256: `0f6fe50d7ce8f693eea068c3d85ea353969b6cad6f6675f434e0f09f65b2e8c6`
- `svamp32_innovation_target_set_20260423.json`
  - sha256: `9f9e3faef2a9d7632be65e3ef99e8af8ec2cb0576fa0d4fa926b1f1772daf3f0`
- `svamp32_innovation_target_set_20260423.md`
  - sha256: `c82b8bdf9b0b8a5265fe39ecc2b6fd3e66982beda9e6f9d1cd9279c226706f74`
- `paper_gate_gate010_with_clean_targets.json`
  - sha256: `b8777c582675c9281ebd749a63e63bd79e11e3777e21ae457b62ef21265880eb`
- `paper_gate_gate010_with_clean_targets.md`
  - sha256: `62cebb57541db1df8157fe39069ca454c177c766828deb9f9d6cd260546d71a7`
