# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_smoke_20260428/benchmark.jsonl --output-dir results/source_private_hidden_repair_packet_weakened_helper_20260428/phi3_trace_no_hint --model microsoft/Phi-3-mini-4k-instruct --device mps --dtype float32 --limit 64 --seed 28 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
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
