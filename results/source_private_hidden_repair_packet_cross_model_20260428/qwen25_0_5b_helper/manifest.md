# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_smoke_20260428/benchmark.jsonl --output-dir results/source_private_hidden_repair_packet_cross_model_20260428/qwen25_0_5b_helper --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 64 --seed 28 --max-new-tokens 8 --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `64`
- packet valid rate: `0.984`
- matched model packet accuracy: `0.984`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
