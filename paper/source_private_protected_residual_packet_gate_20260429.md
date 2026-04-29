# Source-Private Protected Residual Packet Gate

- date: `2026-04-29`
- artifact: `results/source_private_protected_residual_packet_gate_20260429/`
- script: `scripts/build_source_private_protected_residual_packet_gate.py`
- test: `tests/test_build_source_private_protected_residual_packet_gate.py`
- scale rung: medium confirmation / codec ablation

## Purpose

This gate tests whether a TurboQuant/QJL-inspired packet codec can become a
distinct technical contribution beyond the scalar Wyner-Ziv packet. The source
encoder ranks random-rotated scalar coordinates by calibration separation
between the source prediction and the correct target candidate, sends a
protected scalar head, and uses remaining bytes for a sign-sketch residual.

## Result

The strict promotion gate fails: `0/9` rows pass the full protected-codec rule.
The failure is not because source controls break. Every row beats target-only
and keeps protected source-destroying controls clean. The failure is that the
implementation misses the predeclared systems/accuracy bar: p50 decode latency
is `3.56-7.33 ms` rather than `<2 ms`, and the 6-byte rows trail scalar WZ by
`0.029-0.039` on two remaps.

## Evidence

| Remap | Budget | Protected | Scalar WZ | QJL | Canonical RASP | Target | Best protected control | Protected-control | Protected-scalar | p50 ms | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 101 | 2 | 0.430 | 0.418 | 0.396 | 0.350 | 0.250 | 0.279 | 0.150 | +0.012 | 3.562 | near-miss |
| 101 | 4 | 0.447 | 0.432 | 0.461 | 0.494 | 0.250 | 0.264 | 0.184 | +0.016 | 3.824 | near-miss |
| 101 | 6 | 0.447 | 0.463 | 0.447 | 0.494 | 0.250 | 0.264 | 0.184 | -0.016 | 6.682 | near-miss |
| 103 | 2 | 0.438 | 0.436 | 0.439 | 0.363 | 0.250 | 0.266 | 0.172 | +0.002 | 3.905 | near-miss |
| 103 | 4 | 0.465 | 0.475 | 0.461 | 0.520 | 0.250 | 0.250 | 0.215 | -0.010 | 3.973 | near-miss |
| 103 | 6 | 0.479 | 0.508 | 0.484 | 0.520 | 0.250 | 0.244 | 0.234 | -0.029 | 7.326 | fail |
| 107 | 2 | 0.432 | 0.418 | 0.393 | 0.350 | 0.250 | 0.256 | 0.176 | +0.014 | 4.315 | near-miss |
| 107 | 4 | 0.436 | 0.445 | 0.453 | 0.506 | 0.250 | 0.258 | 0.178 | -0.010 | 4.046 | near-miss |
| 107 | 6 | 0.453 | 0.492 | 0.457 | 0.506 | 0.250 | 0.250 | 0.203 | -0.039 | 6.609 | fail |

## Interpretation

Protected residual packets are useful but not promotable as the headline
method. They improve the 2-byte frontier on remaps `101` and `107`, remain
within `0.02` of scalar WZ on seven of nine rows, and beat the best protected
source-destroying control by `0.150-0.234`. However, canonical RASP is stronger
at 4-6 bytes and scalar WZ remains the better learned packet at 6 bytes.

The paper should treat this as a principled compression/quantization ablation:
it answers the reviewer question "did you compare to a TurboQuant/QJL-style
codec?" but should not be claimed as a new winning method without an optimized
vectorized decode path and a stricter rerun.

## Next Gate

Do not tune this exact codec before solving higher-priority reviewer issues.
The next stronger ICLR gate remains either:

1. `n=256` target-model decoder replication on core and holdout, or
2. an anchor-relative sparse/dictionary packet smoke aimed at the cross-family
   failure mode.
