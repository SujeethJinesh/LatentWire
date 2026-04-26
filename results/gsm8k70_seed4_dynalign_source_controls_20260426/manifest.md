# GSM8K70 Seed4 Dynalign Source-Control Manifest

- date: `2026-04-26`
- scale-up rung: strict small same-family seed/source-control gate
- status: `fails_live_gate`
- code commit before run: `eb9a5108b990c130511c2464dbdd2a46e168fcc8`
- branch tested: `dynalign_module_replace_residrank16`
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- dataset: `data/gsm8k_eval_70.jsonl`
- exact materialized IDs:
  `results/gsm8k70_seed_repeat_full_20260422/_artifacts/gsm8k_eval_70.jsonl`
- seed: `4`
- device: `mps`
- dtype: `float32`

## Gate

Run a fresh finite seed for the strongest older GSM70 dynalign row. Promotion
required exact ordered ID parity, `70/70` numeric coverage, no empty
predictions, and beating the target row before source controls would be
interpreted.

## Evidence

| Metric | Value |
|---|---:|
| accuracy | 0.0571 |
| correct | 4/70 |
| paired wins vs target | 3 |
| paired losses vs target | 3 |
| paired ties vs target | 64 |
| numeric coverage | 70/70 |
| empty predictions | 0 |
| checkpoint nonfinite numel | 0 |
| source-control status | not_run_live_gate_failed |

## Artifacts

- summary JSON:
  `results/gsm8k70_seed4_dynalign_source_controls_20260426/seed4_residual_sweep.json`
  - sha256:
    `324d2b84ff5f47c920e6352534adb183526b7aecd070c4f4d6394e4f743ffcbc`
- summary markdown:
  `results/gsm8k70_seed4_dynalign_source_controls_20260426/seed4_residual_sweep.md`
  - sha256:
    `da5c9b5389bf5ec976254ede1f556edaa47e83da977c0c3a55cb4d18421a7e28`
- prediction JSONL:
  `results/gsm8k70_seed4_dynalign_source_controls_20260426/dynalign_module_replace_residrank16_seed4.jsonl`
  - sha256:
    `0a442c7aa43708e2aa8301a6ceb8e986f53d3523edca881ce2904b858c58589c`
- prediction metadata:
  `results/gsm8k70_seed4_dynalign_source_controls_20260426/dynalign_module_replace_residrank16_seed4.jsonl.meta.json`
  - sha256:
    `19ee22759cf87c91dc87e67b9dbf8135a432bd350f0a39b2ad877cba7e9aee7e`
- checkpoint health JSON:
  `results/gsm8k70_seed4_dynalign_source_controls_20260426/seed4_checkpoint_health.json`
  - sha256:
    `7d1bff6d269127773fb1e1797a47405130be82ee07ea30d009e953e84e4169fb`
- run log:
  `.debug/gsm8k70_integrated_source_controls_20260426/seed4/logs/run_seed4_source_controls.log`
  - sha256:
    `324d2b84ff5f47c920e6352534adb183526b7aecd070c4f4d6394e4f743ffcbc`
- checkpoint tensor, not tracked:
  `checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_seed4.pt`
  - sha256:
    `1d9e667fe90a7fbe4b06d982796d09f58398f18144780bb20f4950e774a0d26e`

## Decision

Fail the gate and kill raw GSM70 dynalign scale-up as the current live method
branch. The next branch is a source-derived latent/token sidecar or
token/layer-level C2C-residual distillation gate on SVAMP32.
