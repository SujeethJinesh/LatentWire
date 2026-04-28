# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_seed_repeat_20260429/core_seed31/benchmark.jsonl --output-dir results/source_private_hidden_repair_packet_seed_repeat_20260429/core_seed31_llm/qwen3_raw_log_no_trace --model Qwen/Qwen3-0.6B --device mps --dtype float32 --limit 500 --seed 31 --max-new-tokens 8 --prompt-mode raw_log_no_trace --no-enable-thinking
```

## Outcome

- pass gate: `False`
- examples: `500`
- packet valid rate: `0.000`
- matched model packet accuracy: `0.250`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
