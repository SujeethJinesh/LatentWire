# C3 Rotation Config Grid

Status: `READY_FOR_GATED_GPU_SMOKE_TEMPLATE`.

This worker built a config grid, not results. The grid includes the exact
current ParoQuant baseline:

- group size 128
- 8 rotations
- clip `[0.25, 4.0]`

The full grid is:

- group size: 64, 128, 256
- rotations: 4, 8, 12, 16
- clip: `[0.125, 8.0]`, `[0.25, 4.0]`, `[0.5, 2.0]`

Total configs: 36, including the baseline.

## Gate

Do not run the full grid immediately. Recommended order:

1. Falcon and DeepSeek ParoQuant smoke, because those decide whether rotation
   is a four-model remedy.
2. Granite clip-only smoke from C2, because it targets the known ParoQuant
   tail.
3. If the clip-only smoke shows held-out improvement, expand to group size and
   rotation count.

## Overhead Notes

This implementation is an algorithmic reproduction of ParoQuant. Runtime kernel
overhead is not measured here. Expected offline calibration/search cost grows
roughly with `num_rotations` and with the number of model layers; smaller
groups increase group count and metadata overhead.

## Pass / Fail

A config can be promoted only if it beats the exact baseline on held-out
confirmation traces by one of:

- median recovery improves by at least 0.05;
- CI lower improves by at least 0.10;
- worst-trace or 25% CVaR improves with median loss no worse than 0.05.

Calibration-only gains are not sufficient.

