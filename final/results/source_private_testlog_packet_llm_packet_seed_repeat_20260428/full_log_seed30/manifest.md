# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/full_log_seed30 --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 160 --seed 30 --max-new-tokens 8 --prompt-mode full_log --no-enable-thinking
```

## Outcome

- pass gate: `False`
- examples: `160`
- packet valid rate: `0.163`
- matched model packet accuracy: `0.344`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`

## Artifact Hashes

- `model_packets.jsonl`: `21a5decf2f96885d0fc195856e3a8f64c1a5b1871f9333ac214b28a94eba10df`
- `predictions.jsonl`: `8bac90f38d4c46aa682da2c8373227fcba115bb6825b0be16c39a1cbdbf48934`
- `summary.json`: `a83d7d20fa6b8a02ab16491114defdb6743537705f2f3bc43088de4772d6f508`
- `summary.md`: `f9eff1ee8bc84d0371a361bfff44b0c97ea0ec994dcb43508b05382756d7dcf2`