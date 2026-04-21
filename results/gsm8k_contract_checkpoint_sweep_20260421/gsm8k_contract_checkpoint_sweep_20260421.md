# GSM8K32 Contract Checkpoint Sweep

- date: `2026-04-21`
- baseline contract: `/Users/sujeethjinesh/Desktop/LatentWire/results/gsm8k_smoke_contract_20260421/gsm8k_smoke_contract_20260421.md`
- source -> target: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`
- slice: `32` examples from `data/gsm8k_eval_70.jsonl`

| Candidate | Accuracy | Win vs target | Loss vs target | Tie vs target | Numeric coverage | Empty preds | Promote? |
|---|---:|---:|---:|---:|---:|---:|---:|
| dynalign_module_replace | 0.0938 | 1 | 0 | 31 | 32 | 0 | yes |
| spanalign_module_replace | 0.0938 | 1 | 0 | 31 | 32 | 0 | yes |
| bytespan_module_replace | 0.0312 | 0 | 1 | 31 | 32 | 0 | no |
| sae_adapter | 0.0000 | 0 | 2 | 30 | 30 | 0 | no |

## Checks

- `dynalign_module_replace` — row_count_matches_slice=PASS, example_ids_match_target=PASS, no_empty_predictions=PASS, numeric_extraction_coverage=PASS, beats_target=PASS
- `spanalign_module_replace` — row_count_matches_slice=PASS, example_ids_match_target=PASS, no_empty_predictions=PASS, numeric_extraction_coverage=PASS, beats_target=PASS
- `bytespan_module_replace` — row_count_matches_slice=PASS, example_ids_match_target=PASS, no_empty_predictions=PASS, numeric_extraction_coverage=PASS, beats_target=FAIL
- `sae_adapter` — row_count_matches_slice=PASS, example_ids_match_target=PASS, no_empty_predictions=PASS, numeric_extraction_coverage=FAIL, beats_target=FAIL
