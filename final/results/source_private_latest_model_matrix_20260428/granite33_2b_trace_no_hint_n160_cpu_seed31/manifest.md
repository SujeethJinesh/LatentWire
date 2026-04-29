# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/granite33_2b_trace_no_hint_n160_cpu_seed31 --model ibm-granite/granite-3.3-2b-instruct --device cpu --dtype float32 --limit 160 --seed 31 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `160`
- packet valid rate: `0.537`
- matched model packet accuracy: `0.631`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
