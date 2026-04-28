# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/helper_seed29 --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 160 --seed 29 --max-new-tokens 8 --prompt-mode helper_line --no-enable-thinking
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

- `model_packets.jsonl`: `a57d9d08fce95060596e53acd36312eee550e1d2255eee93f68ec3edcd0c5c4f`
- `predictions.jsonl`: `fac16f414e3ab4463a630dc1f2c6397b6710c8de158e3932dfde4f672105e3a2`
- `summary.json`: `e74fcf74ea214f379e54a472fdde20fa00e61100f649d88ac3f824e701e40761`
- `summary.md`: `e45ac2d4cf104b694f1139a205d57cc8006828ee2f9a3c5b10bed739ede809a0`