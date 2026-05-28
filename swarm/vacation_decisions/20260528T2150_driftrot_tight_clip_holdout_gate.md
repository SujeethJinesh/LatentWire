# DriftRot Tight-Clip Held-Out Gate

Date: 2026-05-28T21:50Z

## Decision

Run a small held-out Granite subset for tight ParoQuant `[0.5, 2.0]` before
making any DriftRot clip/CVaR claim.

## Rationale

The tight-clip candidate looked strong on inspected traces `[4, 7, 9, 10]`,
including the severe ParoQuant tail at prompt 4. Because those traces are now
selection data, they cannot support a final method claim by themselves.

## Held-Out Traces

Use positive-gap traces not in the inspected subset:

- prompt 1: strong original ParoQuant recovery, checks no regression;
- prompt 2: weak original ParoQuant recovery, checks improvement opportunity;
- prompt 5: high positive gap and near-1 original recovery, checks stability.

Gate:

- promote only if tight clip improves CI/tail behavior without losing more than
  0.05 median recovery versus original ParoQuant on these held-out traces;
- demote clip/CVaR retuning to appendix if the held-out subset regresses.

This remains a Tier-2 DriftRot candidate, not the headline method unless a
calibration/confirmation protocol is later frozen and repeated.
