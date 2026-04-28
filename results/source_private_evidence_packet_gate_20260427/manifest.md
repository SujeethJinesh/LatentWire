# Source-Private Evidence Packet Gate Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_evidence_packet_gate.py --examples 128 --candidates 4 --seed 17 --syndrome-bytes 2 --output-dir results/source_private_evidence_packet_gate_20260427
```

## Artifacts

- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`

## Outcome

- examples: `128`
- syndrome bytes/example: `2`
- matched syndrome accuracy: `1.000`
- best no-source accuracy: `0.250`
- best source-destroying control accuracy: `0.250`
- matched minus best no-source: `0.750`
- pass gate: `True`
