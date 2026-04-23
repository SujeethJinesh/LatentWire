# SVAMP32 Dynalign Source Controls Manifest

- date: `2026-04-23`
- gate: `legacy dynalign exact-ID source controls on SVAMP32 C2C-teacher surface`
- materialized eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- translator: `checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt`

## Salt 1

- matched candidate: `results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt1_telemetry.jsonl`
- zero-source control: `results/svamp32_dynalign_source_controls_20260423/dynalign_salt1_zero_source.jsonl`
  - sha256: `f7f9fc0c6bb6ebcaf6830b0d85f51bd497e4e2cec633328c840379349aca6cf9`
- shuffled-source control: `results/svamp32_dynalign_source_controls_20260423/dynalign_salt1_shuffled_source_salt1.jsonl`
  - sha256: `67706bcc57c58023d568e1dd8372c1dd4e67de27266d1ebece2ad7d41abfb94d`
- probe json: `results/svamp32_dynalign_source_controls_20260423/c2c_teacher_probe_salt1_controls.json`
  - sha256: `5167ec6f19b3611b6af0f9b260a25e96f1249c413c2b7a68a21776b8bd85a86b`
- probe markdown: `results/svamp32_dynalign_source_controls_20260423/c2c_teacher_probe_salt1_controls.md`
  - sha256: `dfc668c718c3364f3bebf3fcf9b6af8bd5b057175c51865c2223504279709588`
- outcome: `candidate_teacher_recovery_explained_by_controls`

## Salt 2

- matched candidate: `results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt2_telemetry.jsonl`
- zero-source control: `results/svamp32_dynalign_source_controls_20260423/dynalign_salt2_zero_source.jsonl`
  - sha256: `0fed2589e15778957ab2be1629e0a15092007f4aecd814a3229873362edb827f`
- shuffled-source control: `results/svamp32_dynalign_source_controls_20260423/dynalign_salt2_shuffled_source_salt2.jsonl`
  - sha256: `70401ff103b409fbb905a0ec4a58371b8a16fcda5b3d26d0d29b7d43cf27f223`
- probe json: `results/svamp32_dynalign_source_controls_20260423/c2c_teacher_probe_salt2_controls.json`
  - sha256: `72b4d7a6c968d735ad519496b2411a6f4ee9674141c9acaec6fa65b2ea894792`
- probe markdown: `results/svamp32_dynalign_source_controls_20260423/c2c_teacher_probe_salt2_controls.md`
  - sha256: `23a0451ae0eb6afac7055f3f54de7588d0f24ee52cbc5058e46fda3a43a6335c`
- outcome: `candidate_teacher_recovery_partially_control_explained`

## Result

- salt 1 is killed as a source-specific signal
- salt 2 keeps one matched-only teacher-only ID (`e3ab8666238a289e`), but remains
  a weak lower-bound comparator rather than a promotable method
