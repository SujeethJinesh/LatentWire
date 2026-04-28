# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/helper_seed30 --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32 --limit 160 --seed 30 --max-new-tokens 8 --prompt-mode helper_line --no-enable-thinking
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

- `model_packets.jsonl`: `56b223a1ea4bc2cb7db7cedff45ed18dc8b9214cd663239e1813fd086d766158`
- `predictions.jsonl`: `00402d7a9ae1ac191966cae95e1b539c9c9f3c5836c0f50029d3a4bb6174a97f`
- `summary.json`: `a27eaa4fe5dafbd3da0b9c6bedbc3245fcd0f44e9816d91d3ac6f05fd5e77112`
- `summary.md`: `314d0fcd5492cc86ad7444e6ed9a467a5f693f410ab8a80b842bf097f70279f7`