# GSM8K 32-Example Smoke Contract

- date: `2026-04-21`
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- slice: `32` examples from `data/gsm8k_eval_70.jsonl`
- checkpoint: `checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt`

| Row | Accuracy | N | Win vs target | Loss vs target | Tie vs target | Tokens avg | Latency avg | Ex/s | Numeric coverage | Empty preds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target_alone | 0.0625 | 32 | 0 | 0 | 32 | 63.44 | 0.0000 | 3200000000.0000 | 32 | 0 |
| text_to_text | 0.0312 | 32 | 1 | 2 | 29 | 64.00 | 0.0000 | 3200000000.0000 | 32 | 0 |
| rotalign_kv | 0.0625 | 32 | 1 | 1 | 30 | 57.03 | 0.0000 | 3200000000.0000 | 28 | 0 |
| c2c_generate | 0.1250 | 32 | 3 | 1 | 28 | 62.31 | 2.5502 | 0.3921 | 32 | 0 |

## Checks

- PASS: `row_counts_match_slice` — counts=[32, 32, 32, 32] expected=32
- PASS: `example_ids_identical` — all four rows share the same ordered example IDs
- PASS: `greedy_config` — max_new_tokens=64; evaluate and c2c both run greedy generation
- PASS: `no_empty_predictions` — empty_predictions=[0, 0, 0, 0]
- FAIL: `numeric_extraction_coverage` — coverage=[32, 32, 28, 32] threshold=31
- PASS: `target_rerun_byte_identical` — rerun target predictions exactly match the main target row
- PASS: `target_offline_rescore_matches` — offline=0.0625 sidecar=0.0625
- PASS: `c2c_beats_target_by_two` — c2c=0.1250 target=0.0625
- PASS: `rotalign_ties_or_beats_target` — rotalign=0.0625 target=0.0625
