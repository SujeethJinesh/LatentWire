# Source-Private Test-Log Model-Packet Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl --output-dir results/source_private_testlog_packet_cross_model_20260428/phi3_mini_helper --model microsoft/Phi-3-mini-4k-instruct --device mps --dtype float32 --limit 160 --seed 28 --max-new-tokens 8 --prompt-mode helper_line --no-enable-thinking
```

## Outcome

- pass gate: `True`
- examples: `160`
- packet valid rate: `0.950`
- matched model packet accuracy: `0.912`
- target-only accuracy: `0.250`

## Artifacts

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`

## Artifact Hashes

- `model_packets.jsonl`: `74767e7bc1100cda1299c5012e631cce3f44a0847774c6ecfde8f03f34e50450`
- `predictions.jsonl`: `b15c673695e30f7a259a885c15ca08a9085fdafcd623426099c28c141c5370ad`
- `summary.json`: `99e58e7c086c4be13c9b04555c6ae39517acc2e4022463703081c2003793bbc9`
- `summary.md`: `e9b19e87394f6e4ba9b875fb2852593b4d93683d12143b13c8e5fb03adb19cae`