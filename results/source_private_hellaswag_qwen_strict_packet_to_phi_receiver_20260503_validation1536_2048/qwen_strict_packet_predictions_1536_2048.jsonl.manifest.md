# JSONL Range Materialization

- date: `2026-05-03`
- source: `results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260503_rank_score_channel_qwen05_train512_validation1024_2048/bagged_gate/predictions.jsonl`
- output: `results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/qwen_strict_packet_predictions_1536_2048.jsonl`
- start index: `513`
- count: `512`
- source sha256: `f568bfade523aca74b560c813f74aa22f907b85e2b84c79caba5ac4b71d219e3`
- output sha256: `780629fbf2b4d2134af252caa8829c5f2bfd3795bf560d5dd42057126e5ce5aa`

## IDs

- first metadata id: `None`
- last metadata id: `None`

## Command

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py --source results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260503_rank_score_channel_qwen05_train512_validation1024_2048/bagged_gate/predictions.jsonl --output results/source_private_hellaswag_qwen_strict_packet_to_phi_receiver_20260503_validation1536_2048/qwen_strict_packet_predictions_1536_2048.jsonl --start-index 513 --count 512 --run-date 2026-05-03
```
