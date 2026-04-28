# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_non_qwen_packet_20260428/ibm_granite__granite_3_3_2b_instruct_n160_seed29_copied_helper_cpu --model ibm-granite/granite-3.3-2b-instruct --device cpu --dtype float32 --limit 160 --seed 29 --max-new-tokens 8 --prompt-mode copied_helper --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `160`
- packet valid rate: `0.738`
- matched model packet accuracy: `0.800`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
