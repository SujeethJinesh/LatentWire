# Experimental Shared Utilities

Shared Mac-local utilities for the relevant hybrid-quantization branches:
SSQ-LR, HORN, and HBSM.

These helpers are intentionally small and deterministic. They are not GPU
kernels and they do not support throughput, latency, HBM, or energy claims.
Use them for preregistered Mac gates only.

## Utilities

- `fp4_simulator.py`: deterministic INT/FP-style cast-and-cast-back
  quantization simulators and quality-gap recovery helpers.
- `activation_dumper.py`: lightweight tensor packet save/load helpers for
  cached traces.
- `boundary_inspector.py`: layer-kind and attention/SSM boundary helpers.
- `hybrid_architecture_maps.py`: explicit config-derived boundary maps used to
  validate real trace packet provenance.
- `sensitivity_metrics.py`: quality, drift, and rank-correlation metrics.
- `check_gate_packet.py`: packet validator for synthetic and real Mac-local
  gate results, with stricter `--mode real --project ...` contracts.
- `hybrid_trace_packet_runbook.md`: required real-packet schema for SSQ-LR,
  HORN, and HBSM.

## Local Test

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

## Architecture Map Artifact

The current config-only map is:

```text
experimental/shared/results/hybrid_architecture_maps_20260506/
```

Regenerate it with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_architecture_maps
```

## Claim Boundary

Passing these tests only means the local utilities are deterministic and
internally consistent. Any promoted paper claim still requires the relevant
project gate to pass.
