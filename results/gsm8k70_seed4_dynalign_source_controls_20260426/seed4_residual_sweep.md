# GSM8K32 Residual Rank Sweep

- date: `2026-04-25`
- baseline contract: `/Users/sujeethjinesh/Desktop/LatentWire/results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k_smoke_contract_20260421.md`
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- calibration file: `.debug/calibration_64.txt`
- seed: `4`
- bridge bank size: `4`
- source controls: `enabled`
- accept/fallback: `disabled`
- slice: `70` examples from `data/gsm8k_eval_70.jsonl`

| Base | Residual rank | Bridge bank | Accuracy | Win vs target | Loss vs target | Tie vs target | Numeric coverage | Empty preds | Status | Nonfinite | First bad key | Reused ckpt | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| dynalign_module_replace | 16 | 4 | 0.0571 | 3 | 3 | 64 | 70 | 0 | ok | 0 | - | no | no |

## Checks

- `dynalign_module_replace_residrank16` — row_count_matches_slice=PASS, example_ids_match_target=PASS, no_empty_predictions=PASS, numeric_extraction_coverage=PASS, beats_target=FAIL

## Source Controls

| Row | Status | Passed | Readout | Controls |
|---|---|---:|---|---|
| `dynalign_module_replace_residrank16` | not_run_live_gate_failed | no | `-` | - |

## Checkpoint Health

- `dynalign_module_replace_residrank16` — status=ok, nonfinite_numel=0, first_bad_key=-, top_tensor=quant_aux_proj_V.12 (max_abs=16326.4326, nonfinite=0)
