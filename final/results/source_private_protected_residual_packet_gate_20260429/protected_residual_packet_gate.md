# Source-Private Protected Residual Packet Gate

- pass gate: `False`
- rows: `9`
- protected pass rows: `0`
- remaps with protected pass: `[]`
- remap seeds: `[101, 103, 107]`
- budgets: `[2, 4, 6]`
- max protected accuracy: `0.479`
- min passing protected-control margin: `-`
- min passing protected-scalar margin: `-`
- max passing p50 decode latency ms: `-`

## Rows

| Remap | Budget | N | Protected | Scalar WZ | QJL | Canonical RASP | Target | Best protected control | Protected-control | Protected-scalar | p50 ms | Protected pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 512 | 0.430 | 0.418 | 0.396 | 0.350 | 0.250 | 0.279 | 0.150 | 0.012 | 3.562 | `True` |
| 101 | 4 | 512 | 0.447 | 0.432 | 0.461 | 0.494 | 0.250 | 0.264 | 0.184 | 0.016 | 3.824 | `True` |
| 101 | 6 | 512 | 0.447 | 0.463 | 0.447 | 0.494 | 0.250 | 0.264 | 0.184 | -0.016 | 6.682 | `True` |
| 103 | 2 | 512 | 0.438 | 0.436 | 0.439 | 0.363 | 0.250 | 0.266 | 0.172 | 0.002 | 3.905 | `True` |
| 103 | 4 | 512 | 0.465 | 0.475 | 0.461 | 0.520 | 0.250 | 0.250 | 0.215 | -0.010 | 3.973 | `True` |
| 103 | 6 | 512 | 0.479 | 0.508 | 0.484 | 0.520 | 0.250 | 0.244 | 0.234 | -0.029 | 7.326 | `True` |
| 107 | 2 | 512 | 0.432 | 0.418 | 0.393 | 0.350 | 0.250 | 0.256 | 0.176 | 0.014 | 4.315 | `True` |
| 107 | 4 | 512 | 0.436 | 0.445 | 0.453 | 0.506 | 0.250 | 0.258 | 0.178 | -0.010 | 4.046 | `True` |
| 107 | 6 | 512 | 0.453 | 0.492 | 0.457 | 0.506 | 0.250 | 0.250 | 0.203 | -0.039 | 6.609 | `True` |

## Interpretation

This gate tests whether a TurboQuant/QJL-inspired packet codec can make the compact source-private method more principled: protected scalar coordinates are selected by calibration separation, while remaining bytes carry a sign-sketch residual. It is a codec contribution only if it preserves scalar WZ accuracy at comparable bytes with clean controls and low CPU decode overhead.
