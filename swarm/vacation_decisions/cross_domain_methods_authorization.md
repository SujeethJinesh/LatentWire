# Vacation Decision: Cross-Domain Methods Authorization

Date: 2026-05-18 UTC

## Situation

The human authorized three cross-domain candidate methods while the Phase 9
queue was already active:

- M30 spaced rehearsal protection
- M31 pulsed full-precision protection
- M33 stable core plus periodic recalibration

The authorization explicitly warned that the synthetic simulations behind
these ideas were mechanism-coherence checks, not empirical validation.

## Candidate Methods

M30 protects channels by recency/freshness, motivated by spaced rehearsal and
forgetting-curve analogies. It is deprioritized because the synthetic check
suggested it may protect recently-departed channels while leaving current
outliers unprotected. It should run only if M11b, M26, M27, M31, and M33 all
fail and there is still a strong reason to test the mechanism.

M31 performs periodic BF16 pulse passes to reset accumulated error. It is
conditional on the KL accumulation experiment. If the fitted KL trajectory has
decay `>= 0.95`, M31 becomes authorized. If decay is `<= 0.80`, M31 should be
documented as deferred. If decay is near threshold, roughly `0.88-0.97`, ask
the human rather than applying the number mechanically.

M33 combines stable-core protection with periodic recalibration. It is
conditional on M26. If M26 passes cleanly, M33 is redundant. If M26 kills
cleanly, M33 is unlikely to help. If M26 is ambiguous or shows partial signal
in the `0.05-0.25` recovery range, M33 becomes authorized.

## Synthetic-Validation Caveats

The synthetic checks assumed independent channels, linear error aggregation,
Ornstein-Uhlenbeck-like trajectories, and easier spectra than the empirical
Granite packet. Real activation dynamics are correlated, heavy-tailed,
task-dependent, and propagate through nonlinear attention and KV state. Treat
synthetic results as evidence about internal coherence only, not as pass/fail
probabilities.

## Decision

No new method from this set is unconditionally authorized. Continue the current
queue first:

1. M26 stable-core size check and, if suitable, M26.
2. M27 layer-stratified protection.
3. ParoQuant baseline.
4. KL accumulation.

Then use M26 and KL results to decide whether M31 or M33 is authorized.

## What Would Invalidate This Decision

The human may decide that the mechanical M11b PASS should move immediately to
replication and skip all cross-domain alternatives. That would override this
document. Conversely, if KL accumulation shows clear long-horizon compounding
or M26 shows a partial stable-core signal, the relevant conditional method
should be promoted according to the criteria above.
