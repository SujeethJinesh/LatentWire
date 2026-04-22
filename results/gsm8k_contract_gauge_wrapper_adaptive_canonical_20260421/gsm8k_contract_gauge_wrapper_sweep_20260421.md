# GSM8K32 Gauge Wrapper Sweep

- date: `2026-04-21`
- baseline contract: `/Users/sujeethjinesh/Desktop/LatentWire/results/gsm8k_smoke_contract_20260421/gsm8k_smoke_contract_20260421.md`
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- calibration file: `.debug/calibration_64.txt`
- slice: `32` examples from `data/gsm8k_eval_70.jsonl`

| Candidate | Alignment | Residual rank | Accuracy | Win vs target | Loss vs target | Tie vs target | Numeric coverage | Empty preds | Promote? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dynalign_resid16_adaptive | grouped_adaptive_canonical_transport | 16 | 0.1250 | 2 | 0 | 30 | 32 | 0 | yes |

## Checks

- `dynalign_resid16_adaptive` — row_count_matches_slice=PASS, example_ids_match_target=PASS, no_empty_predictions=PASS, numeric_extraction_coverage=PASS, beats_target=PASS
