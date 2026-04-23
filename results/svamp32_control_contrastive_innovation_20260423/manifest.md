# SVAMP32 Source-Control Contrastive Innovation Manifest

Date: 2026-04-23

Purpose: matched-only SVAMP32 screen for a default-off source-control objective
under `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`.

Tracked readouts:

- `control_zero_shuffle_w010_m001_attention_clean_targets.json`
- `control_zero_shuffle_w010_m001_attention_clean_targets.md`

Scratch artifacts:

- checkpoint: `.debug/svamp32_control_contrastive_innovation_20260423/checkpoints/qwen25_to_qwen3_svamp32_control_zero_shuffle_w010_m001_r16_bank16_seed1.pt`
- matched sweep predictions: `.debug/svamp32_control_contrastive_innovation_20260423/preds/control_zero_shuffle_w010_m001_attention_gate_sweep.jsonl`
- logs: `.debug/svamp32_control_contrastive_innovation_20260423/logs/`

Summary:

- status: `no_matched_gate_candidate_for_controls`
- best matched row: `rotalign_kv_gate_0.12`, `9/32`
- clean residual recovered: `0/6`
- target_self_repair reference: `14/32`
- decision: do not run source-destroying controls for this checkpoint
