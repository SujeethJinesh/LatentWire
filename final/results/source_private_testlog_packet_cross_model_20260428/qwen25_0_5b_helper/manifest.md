# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_cross_model_20260428/qwen25_0_5b_helper --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 160 --seed 28 --max-new-tokens 8 --prompt-mode helper_line --no-enable-thinking
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

- `model_packets.jsonl`: `c053691b8d496e9ce7a83b6ab16ffc911acf9d39306f4e7395e53dcbe31a08a8`
- `predictions.jsonl`: `a0f44149bf844b33edd83c51662b0029be8c568ae3fec21d895dec0c77e2865d`
- `summary.json`: `fbff407416d99fdddbe0ae1e774e5fbf0b232840937c7766215302e83a006d0a`
- `summary.md`: `a4d22dfb22938323d9d243e4ef3bef516df9c3d7f1e21d7886b320e68a66006e`