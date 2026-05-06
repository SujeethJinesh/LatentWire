# Experimental Shared Utilities

Shared Mac-local utilities for the new hybrid-quantization branches:
SinkKV, SSQ-LR, HORN, and HBSM.

These helpers are intentionally small and deterministic. They are not GPU
kernels and they do not support throughput, latency, HBM, or energy claims.
Use them for preregistered Mac gates only.

## Utilities

- `fp4_simulator.py`: deterministic INT/FP-style cast-and-cast-back
  quantization simulators and quality-gap recovery helpers.
- `activation_dumper.py`: lightweight tensor packet save/load helpers for
  cached traces.
- `boundary_inspector.py`: layer-kind and attention/SSM boundary helpers.
- `sensitivity_metrics.py`: quality, drift, and rank-correlation metrics.

## Local Test

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

## Claim Boundary

Passing these tests only means the local utilities are deterministic and
internally consistent. Any promoted paper claim still requires the relevant
project gate to pass.
