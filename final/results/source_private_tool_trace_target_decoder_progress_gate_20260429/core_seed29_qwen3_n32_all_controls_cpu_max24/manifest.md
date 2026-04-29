# Source-Private Tool-Trace Target-Decoder Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl --output-dir results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n32_all_controls_cpu_max24 --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 32 --seed 29 --max-new-tokens 24 --no-enable-thinking --conditions target_only matched_packet shuffled_packet random_same_byte structured_json_2byte structured_free_text_2byte --progress-jsonl .debug/source_private_target_decoder_progress_20260429/core_seed29_qwen3_n32_all_controls_cpu_max24_progress.jsonl --progress-every 1
```

## Outcome

- pass gate: `True`
- examples: `32`
- matched accuracy: `0.688`
- target-only accuracy: `0.250`
- best control accuracy: `0.250`

## Artifacts

- `target_predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
