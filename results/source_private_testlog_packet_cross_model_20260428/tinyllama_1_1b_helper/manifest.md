# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_cross_model_20260428/tinyllama_1_1b_helper --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device mps --dtype float32 --limit 160 --seed 28 --max-new-tokens 8 --prompt-mode helper_line --no-enable-thinking
```

## Outcome

- pass gate: `False`
- examples: `160`
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

## Artifact Hashes

- `model_packets.jsonl`: `f873e8e06a872ea2fad4ec92f743ac0a0e37c23d6317c9e6cc3bcac926e44a8e`
- `predictions.jsonl`: `280cf7cdef29dbd872e9ffe731de42da290f4c0a138d5d79db5f867d859eda56`
- `summary.json`: `d9a505a077b5e78ec3edb5fca55e40bf644a3c4393bd48011655250703975c9d`
- `summary.md`: `c5d473967aa1ae59650d6a22bea62019c0042bbcd0b0616a8e3de53e7be79a80`