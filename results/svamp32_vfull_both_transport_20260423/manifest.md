# SVAMP32 Value Transport Screen Manifest

Date: 2026-04-23

Purpose: matched-only runtime K/V selection screen on the prior ID-weighted
query-innovation checkpoint.

Checkpoint:

- `.debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt`
- sha256: `4e67fdf2b6ea2c962036aad080ec3fe6c64a4083c627a282b925bdf546b90831`

Tracked readouts:

- `idweighted_both_vfull_attention_clean_targets.json`
- `idweighted_both_vfull_attention_clean_targets.md`
- `idweighted_sparse_sourcev_attention_clean_targets.json`
- `idweighted_sparse_sourcev_attention_clean_targets.md`

Scratch artifacts:

- `.debug/svamp32_vfull_both_transport_20260423/preds/idweighted_both_vfull_attention_gate_sweep.jsonl`
- `.debug/svamp32_vfull_both_transport_20260423/preds/idweighted_both_vfull_attention_gate_sweep.jsonl.meta.json`
- `.debug/svamp32_vfull_both_transport_20260423/preds/idweighted_sparse_sourcev_attention_gate_sweep.jsonl`
- `.debug/svamp32_vfull_both_transport_20260423/preds/idweighted_sparse_sourcev_attention_gate_sweep.jsonl.meta.json`
- `.debug/svamp32_vfull_both_transport_20260423/logs/`

Summary:

- V-full both transport: best `9/32`, clean residual `0/6`, target losses `2`
- sparse source-attention V: best `10/32`, clean residual `1/6`, target losses `1`
- decision: no source-destroying controls; runtime value-side selection on this checkpoint is saturated
