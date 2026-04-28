# Source-Private Hidden-Repair Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_non_qwen_packet_20260428/allenai__olmo_2_0425_1b_instruct_n16_seed29_copied_helper_mps --model allenai/OLMo-2-0425-1B-Instruct --device mps --dtype float32 --limit 16 --seed 29 --max-new-tokens 16 --prompt-mode copied_helper --no-enable-thinking
```

## Outcome

- pass gate: `False`
- examples: `16`
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
