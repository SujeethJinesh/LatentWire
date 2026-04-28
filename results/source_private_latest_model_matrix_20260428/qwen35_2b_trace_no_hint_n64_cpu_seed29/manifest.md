# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n64_cpu_seed29 --model Qwen/Qwen3.5-2B --device cpu --dtype float32 --limit 64 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `64`
- packet valid rate: `1.000`
- matched model packet accuracy: `1.000`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
