# 2026-05-27T21:15Z M-PRED Granite scope reduction

## Finding

The first Granite M-PRED attempt began with `mpred_top5_alpha_0_5`. One prompt
completed after roughly 22 minutes of endpoint scoring. At that throughput, the
full seven-arm M-PRED Granite sweep would exceed the 12 GPU-hour experiment cap.

## Decision

Stop the low-priority alpha=0.5 run and restart with a bounded high-information
subset:

- `mpred_top10_alpha_0_95`
- `mpred_top5_alpha_0_95`
- `mpred_random_alpha_top5`
- `random_walk_top5`

The restart uses `--batch-size 2`; the scorer has adaptive CUDA-OOM fallback to
split batches if needed.

## Rationale

The alpha=0.95 arms are the theory-matched arms from the FFT autocorrelation
finding and are more informative than spending the cap on alpha=0.5. Top-10
tests the positive-method branch against M11b's strongest budget; top-5 tests
whether Granite CI width tightens; the two controls preserve the basic M-PRED
vs random comparison.

## Status of stopped run

The stopped run is an intentional scope-reduction artifact, not a scientific
result. It produced no M-PRED score cache and is not claim-bearing.
