# Source-Private Evidence Packet Strict-Small Manifest

## Command

```bash
./venv_arm64/bin/python scripts/run_source_private_evidence_packet_strict_small.py --examples 160 --candidates 4 --seed 28 --budgets 2,4,8,16,32 --output-dir results/source_private_evidence_packet_strict_small_20260428
```

## Outcome

- strict-small pass: `True`
- passing budgets: `[2, 4, 8, 16, 32]`
- best budget bytes: `2`

## Artifacts

- `benchmark.jsonl`
- `sweep_summary.json`
- `sweep_summary.md`
- `manifest.json`
- `manifest.md`
- `predictions_budget2.jsonl`
- `summary_budget2.json`
- `predictions_budget4.jsonl`
- `summary_budget4.json`
- `predictions_budget8.jsonl`
- `summary_budget8.json`
- `predictions_budget16.jsonl`
- `summary_budget16.json`
- `predictions_budget32.jsonl`
- `summary_budget32.json`

## Artifact Hashes

- `benchmark.jsonl`: `20ebf626cfaa8b9584729fe76237fed022e18d795d200c432cd5d89d5b762caf`
- `predictions_budget16.jsonl`: `b3d4a73d217a39a7f1c3eac14217bb95b5f268bbe1703911d9659e34c301f2a9`
- `predictions_budget2.jsonl`: `c0e87767538960f3dec5270bdafe783c8cfcde98e505eb2a11bbf2a97a007cdc`
- `predictions_budget32.jsonl`: `c82de649c51d0a63a24b2e18c843174c85a142e41cc50b6b28aa556d0d9e60f1`
- `predictions_budget4.jsonl`: `eab7df9f503eecddfdce5afe1ad5fc6e16353dc33898c6f31dbba7943242b141`
- `predictions_budget8.jsonl`: `13fe897e554ef055fa0fb486c337540b67704cba2c791f57294af0faf461ce11`
- `summary_budget16.json`: `37e478dc6c5a32810ae04c2009308aad75d0ec12617755257e1373ff8edfa62f`
- `summary_budget2.json`: `cffdb3ac86fbab8b07d421eae16705c471c80f73b67ffdf5beb438e6201546e2`
- `summary_budget32.json`: `f52e6a7a9290ffe68b7eb1642f9dfb6d64d64c019da3e06e3176fb25cacdce11`
- `summary_budget4.json`: `00710541d48b43ff8d79f82e6a1050f24eb06dbc25af794a6f980f3410db06a8`
- `summary_budget8.json`: `f7a3729822bda6b6c3830fb819e14540f3715a7d420d7b6a9018d5c9fd6d3fbd`
- `sweep_summary.json`: `7517fa30b095608d78987afb9b03e76190697452e60d4aede23741d4cc76dff8`
- `sweep_summary.md`: `1b89ee2acbbdd6250077d288c8640b16f5ec9adba35c0e15ba40ae5d4d8ad3d9`