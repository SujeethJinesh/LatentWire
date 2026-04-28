# Source-Private Tool-Trace Target-Decoder Smoke Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py --benchmark-jsonl .debug/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_benchmark/benchmark.jsonl --output-dir results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n16 --model Qwen/Qwen3-0.6B --device mps --dtype float32 --limit 16 --seed 29 --max-new-tokens 24 --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `16`
- matched accuracy: `0.688`
- target-only accuracy: `0.250`
- best control accuracy: `0.250`

## Artifacts

- `target_predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
