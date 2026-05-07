# SinkKV Progress

## 2026-05-06

Status: **NEW / deterministic synthetic probe passed; real Mac gate pending**.

Created the project as the positive-method successor to SinkAware systems
framing. The first gate is simulated low-precision KV quality recovery from
sink protection. No GPU work is allowed before the Mac gate passes.

Deterministic no-download probe:

- packet: `phase2/results/sinkkv_deterministic_probe/`
- decision: `SYNTHETIC_PASS_REAL_DUMPS_NEXT`
- uniform MXFP4 output rel-L2: `0.097249`
- sink-protected matched-budget output rel-L2: `0.068215`
- recent-protected matched-budget output rel-L2: `0.112774`

Interpretation: this only validates policy mechanics and byte accounting on a
synthetic sink-heavy tensor. It authorizes the first real cached Q/K/V dump, not
GPU work or paper claims.

Next exact gate: run the first real cached-K/V simulation using the frozen
policy in `phase2/reference/sinkkv_policy.py`.
