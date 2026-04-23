# SVAMP32 Query Innovation Resampler Gate 0.15 Manifest

- date: `2026-04-23`
- gate: `query-innovation-resampler transfer on frozen SVAMP32 C2C-teacher surface`
- checkpoint: `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`
- materialized eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- result: `candidate_teacher_recovery_explained_by_controls`

## Gate Sweep

| Gate | Correct | Note |
|---:|---:|---|
| `0.10` | `7/32` | below target |
| `0.15` | `9/32` | only live gate |
| `0.25` | `8/32` | target parity |

## Gate `0.15` Row Summary

- matched: `9/32`, wins `3e8a5691f5443495`, `575d7e83d84c1e67`, loss `c042f0a2949ff8e6`
- zero-source: `8/32`, win `575d7e83d84c1e67`, loss `c042f0a2949ff8e6`
- shuffled-source: `9/32`, wins `3e8a5691f5443495`, `575d7e83d84c1e67`, loss `c042f0a2949ff8e6`
- only teacher-only recovered ID: `575d7e83d84c1e67`
- that teacher-only ID is retained by both zero-source and shuffled-source controls

## Artifact Hashes

- `live_gate_sweep.jsonl`
  - sha256: `68101b3cbb5c4af4783e84a62383630ad39edbe91fff0c43aee7780b4d5fc723`
- `live_gate_sweep.jsonl.meta.json`
  - sha256: `a901e5a81316a35b811e470451092041df6eacff41fb0adb6a04a3db04dc59d9`
- `query_innovation_gate015_matched.jsonl`
  - sha256: `e12144343219188de5d0d676ba3ce46cce4156e770ef5e3ae00eba7943351435`
- `query_innovation_gate015_zero_source.jsonl`
  - sha256: `a855a41036e28839514a1f699ff9159ce81bb77de8509573b141fc7f8d5f8ad2`
- `query_innovation_gate015_zero_source.jsonl.meta.json`
  - sha256: `74a80f1ca61576fe1ac8e22f7434a8896e314f330126382307775abe1d47e997`
- `query_innovation_gate015_shuffled_source_salt1.jsonl`
  - sha256: `afb9a5958e788e891adf556bd5cb92467f95f58769d205b1ce5487f685547c51`
- `query_innovation_gate015_shuffled_source_salt1.jsonl.meta.json`
  - sha256: `ad1939d34999936ac5a1a23f87538b2eed09d228a0a9ecca536cb6bec1fbd618`
- `c2c_teacher_probe_gate015.json`
  - sha256: `f555e71b40048e1c31a206bb7a5e32cc02334b5e5fa642aea726f954e7b92423`
- `c2c_teacher_probe_gate015.md`
  - sha256: `6455f2b908dec6bcb381ef3443142b72f7e55ef249c791b98b522ffa35ea7d81`
