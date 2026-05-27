# M-PRED Scoop Check

Timestamp: 2026-05-27T20:00Z

## Decision

No direct scoop found for the M-PRED mechanism. Proceed to preregistration.

## Adjacent Work

- DecDEC dynamically identifies salient channels at each decoding step but is reactive and residual-fetch based, not one-step-ahead Kalman/AR protection.
- AttentionPredictor predicts next-token attention scores for KV-cache compression, not activation-channel sets.
- Rotated Runtime Smooth and ASER smooth activations/outliers but do not predict future protected sets.
- GuidedQuant is closer to Fisher/loss-guided protection than to M-PRED.

## Next Step

Author the M-PRED preregistration and implement the bounded runner/checker before any GPU run.
