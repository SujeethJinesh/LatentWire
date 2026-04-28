# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_llm_packet_20260428 --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 160 --seed 28 --max-new-tokens 8
```

## Outcome

- pass gate: `True`
- examples: `160`
- packet valid rate: `0.919`
- matched model packet accuracy: `0.938`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`

## Artifact Hashes

- `model_packets.jsonl`: `367c12f9ce1abdb9f0a12a0bca4303e0edd08db38576a56bb85cb86114ff30b4`
- `predictions.jsonl`: `6d2f213b7d8182ef6689bee144e111f87dd231832f2a4aadcdb639469cd3cf2c`
- `summary.json`: `d97dd1538ba571eaab3b4c900337c53092232e9b0c89b1c625b41e0d2628fee7`
- `summary.md`: `08a5e9dce88df23cc8259e96c1eb29f59d439285f570385a5707deffe502f4f9`