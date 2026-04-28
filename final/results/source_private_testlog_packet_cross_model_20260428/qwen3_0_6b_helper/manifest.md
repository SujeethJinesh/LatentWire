# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_cross_model_20260428/qwen3_0_6b_helper --model Qwen/Qwen3-0.6B --device mps --dtype float32 --limit 160 --seed 28 --max-new-tokens 8 --prompt-mode helper_line --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `160`
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

## Artifact Hashes

- `model_packets.jsonl`: `c738be79a9dcf9068e48cc5a8a7cb598a5a55ace1e39a276505299ae6263d5ba`
- `predictions.jsonl`: `9f3d7c40aff4b86c223fb9edfa64dc5da2ac6532f77e0075b8433ce96aa3d00f`
- `summary.json`: `357caec776a8ddbb24d6202fca31828d3f0f11bcc10af9837fc2e28c2d99249a`
- `summary.md`: `2e3ba2141437cb4efe42ea30f244073c16d604b4186cef50651d870a35eca760`