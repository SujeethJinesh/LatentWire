# Source-Private Evidence Packet Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_evidence_packet_llm_packet.py --benchmark-jsonl results/source_private_evidence_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_evidence_packet_llm_packet_20260428 --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 16 --budget-bytes 2 --seed 28 --max-new-tokens 24
```

## Outcome

- pass gate: `False`
- examples: `16`
- budget bytes: `2`
- matched model packet accuracy: `0.250`
- target-only accuracy: `0.250`
- source-final-only accuracy: `1.000`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`

## Artifact Hashes

- `model_packets.jsonl`: `83fb343599f2e34f67bf24c1754478d50e493b021472074c52fe3acb20f980c6`
- `predictions.jsonl`: `7b9fbdbb6d1a44aba68a77ded0ab858626fc1fc1fcfddfa4c8984341e5073348`
- `summary.json`: `e3466073fc4dd1d05da29d642ce4611cdee6d961ffeda56f5a480eb1011b3269`
- `summary.md`: `9e8f0f41ec5d3deeb14a1d1d73da467f4c1c027366044647f02d90752da972dd`