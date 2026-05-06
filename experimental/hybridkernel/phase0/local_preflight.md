# HybridKernel Local Preflight

- created UTC: `2026-05-06T04:11:46.450851+00:00`
- status: `BLOCKED_TRITON_UNAVAILABLE`
- python: `3.11.6`
- executable: `/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python`
- platform: `macOS-26.4.1-arm64-arm-64bit`

## Runtime

- torch importable: `True`
- torch version: `2.6.0`
- cuda available: `False`
- cuda version: `None`
- cuda device count: `0`
- mps built: `True`
- mps available: `True`

## Triton

- importable: `False`
- version: `None`
- pip index checked: `True`
- install possible from current index: `False`
- blocker: `Triton is not importable and pip index found no matching package among triton, triton-cpu, and triton-nightly for this environment.`

## Pip Index Results

| package | returncode | available | timed out |
|---|---:|---:|---:|
| `triton` | `1` | `False` | `False` |
| `triton-cpu` | `1` | `False` | `False` |
| `triton-nightly` | `1` | `False` | `False` |

## Interpretation

Mac-local Phase 4 Triton interpreter tests should skip; do not add more kernels or claim performance until a native NVIDIA handoff environment is available. Triton is not importable and pip index found no matching package among triton, triton-cpu, and triton-nightly for this environment.
