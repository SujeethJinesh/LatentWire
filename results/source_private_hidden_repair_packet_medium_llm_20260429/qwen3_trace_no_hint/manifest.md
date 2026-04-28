# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_hidden_repair_packet_medium_llm_20260429/qwen3_trace_no_hint --model Qwen/Qwen3-0.6B --device mps --dtype float32 --limit 500 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `500`
- packet valid rate: `0.776`
- matched model packet accuracy: `0.808`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
