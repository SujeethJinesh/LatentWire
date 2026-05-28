# 2026-05-28T02:15Z GPU cap raised to 500 and M-FISH next gate

## Update

The planning GPU cap is raised from 360 to 500 cumulative hours.

## Consequence

Budget is no longer the blocker for the positive-method queue. The active
blocker is M-FISH infrastructure: faithful Fisher scoring requires gradients at
the long-decode activation surfaces, not cached magnitudes.

## Next gate

Inspect and prototype the smallest faithful gradient-capture path for M-FISH.
Do not run a full M-FISH experiment until the gradient surface is verified and
a preregistered runner/checker path is in place.
