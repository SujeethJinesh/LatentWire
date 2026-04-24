# SVAMP32 Perceiver-Query Connector Results - 2026-04-24

## Summary

- status: `no_matched_gate_candidate_for_controls`
- best matched row: `rotalign_kv_gate_0.15`
- best matched accuracy: `10/32`
- clean residual recovered: `0/6`
- target-self-repair delta: `-4`
- decision: no source-control follow-up for this checkpoint

## Tracked Artifacts

- `perceiver_queries_w030_m010_attention_clean_targets.json`
- sha256: `fbccf197d063dfd133f584a0397322f2e35f6e6de710b8ec92cf5dc594335e3c`
- `perceiver_queries_w030_m010_attention_clean_targets.md`
- sha256: `fd80e0f402bfa2e49166262d29b1950b3d2abb550bf28fc2c7b63d23e8b062e9`

## Scratch Artifacts

- `.debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt`
- sha256: `ad64ffd29b5e31f029e9a4d14d75ed6bcb64906d44dc7746532f1606146712f0`
- `.debug/svamp32_perceiver_query_connector_20260424/preds/perceiver_queries_combined_attention_gate_sweep.jsonl`
- sha256: `a45d014b712f1e315210335a899cd12f18ada8d24e11c40addd53609350927e0`
- `.debug/svamp32_perceiver_query_connector_20260424/logs/calibrate_perceiver_queries_w030_m010_r16_q8_seed1.log`
- `.debug/svamp32_perceiver_query_connector_20260424/logs/evaluate_perceiver_queries_combined_attention_gate_sweep.log`

## Reproduction Notes

See `paper/svamp32_perceiver_query_connector_20260424.md` for full calibration,
evaluation, and analyzer commands. The run used exact-ID SVAMP32, Qwen2.5-0.5B
Instruct as source, Qwen3-0.6B as target, K-only transport, attention-selected
positions, `8` learned connector queries, and zero/shuffle source-control
training.
