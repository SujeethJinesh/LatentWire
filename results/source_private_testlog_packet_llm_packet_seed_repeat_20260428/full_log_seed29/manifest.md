# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/full_log_seed29 --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 160 --seed 29 --max-new-tokens 8 --prompt-mode full_log --no-enable-thinking
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

- `model_packets.jsonl`: `76aecb9fe385df6ba18a60a862ae84c73f813fefa1e56867c20fe531f909824f`
- `predictions.jsonl`: `b22b65bf3dbe101adb4c27082905c20d9971fbc0e5a57effb571066b006466fe`
- `summary.json`: `5f585b7153724d7f523c35fff4fe0f321d267455f39dca132aef53e0651eb010`
- `summary.md`: `ea00e2cab38e2525745e279952d09f3f3690f6e9c9dcc4f3aa4f01751baef21b`