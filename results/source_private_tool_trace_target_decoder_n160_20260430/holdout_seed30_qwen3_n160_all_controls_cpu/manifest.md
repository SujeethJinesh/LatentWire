# Source-Private Tool-Trace Target-Decoder Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/benchmark.jsonl --output-dir results/source_private_tool_trace_target_decoder_n160_20260430/holdout_seed30_qwen3_n160_all_controls_cpu --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 160 --seed 30 --max-new-tokens 24 --no-enable-thinking --conditions target_only matched_packet shuffled_packet random_same_byte structured_json_2byte structured_free_text_2byte --progress-jsonl .debug/source_private_target_decoder_n160_20260430/holdout_seed30_qwen3_n160_all_controls_cpu_progress.jsonl --partial-predictions-jsonl results/source_private_tool_trace_target_decoder_n160_20260430/holdout_seed30_qwen3_n160_all_controls_cpu/target_predictions.partial.jsonl --progress-every 8
```

## Outcome

- pass gate: `True`
- examples: `160`
- matched accuracy: `0.719`
- target-only accuracy: `0.250`
- best control accuracy: `0.263`

## Artifacts

- `target_predictions.jsonl`
- `target_predictions.partial.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
