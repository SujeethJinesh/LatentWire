# Source-Private Candidate-Conditioned Residual Code Smoke, 2026-05-01

## Status

- paper readiness: useful negative evidence for ICLR, but not a promoted
  technical contribution.
- artifact:
  `results/source_private_candidate_conditioned_residual_code_smoke_20260501/`
- code:
  `scripts/run_source_private_candidate_conditioned_residual_code_smoke.py`
- test:
  `tests/test_run_source_private_candidate_conditioned_residual_code_smoke.py`
- references:
  `references/556_candidate_conditioned_residual_code_smoke_refs_20260501.md`

## Question

Can a learned receiver-side residual-code rule improve the live candidate-local
packet while training against strict destructive controls?

The smoke regenerates train/eval benchmark rows locally. It uses the existing
source-private learned-synonym packet and candidate-local residual score
surface, then trains a small linear receiver over candidate-local features. The
training target is answer selection for matched packets and prior fallback for
strict destructive controls. Evaluation uses disjoint regenerated rows for
core-to-holdout, holdout-to-core, and same-family-all directions.

## Result

The gate fails.

| Direction | Matched | Base matched | Target | Best control | Controls clean? |
|---|---:|---:|---:|---:|---|
| core-to-holdout | `0.625` | `0.875` | `0.250` | `0.258` | yes |
| holdout-to-core | `0.250` | `0.750` | `0.250` | `0.250` | yes |
| same-family-all | `0.500` | `0.812` | `0.250` | `0.266` | yes |

A small control-weight sweep was also run in `.debug/`. The best low-control
setting recovered more matched signal (`0.750/0.625/0.812`) but leaked the
same-family control above the allowed target band. Higher control weights kept
controls clean but made the receiver fall back to the prior too often.

## Interpretation

Pruned:

- learned receiver-side calibration over the existing candidate-local score
  surface. It is too conservative when trained against strict controls and does
  not beat the base residual receiver.

Promoted:

- the live hand-built candidate-local residual packet remains the strongest
  current method.
- the margin/threshold atlas remains the right diagnostic surface for future
  methods.

Still alive:

- a true candidate-conditioned residual code or syndrome, where the packet
  itself is learned/selected against the receiver candidate basis. The failed
  branch only learned a receiver-side rule over an already fixed packet.

## Lay Explanation

This tried to teach the receiver when to trust the tiny private hint and when
to ignore it. The receiver learned to ignore fake hints, but it also ignored too
many real hints, especially in holdout-to-core. That means this is not the
method improvement we need.

## Next Gate

Do not integrate this learned calibration layer into the live runner. The next
method gate should change the packet construction itself: a learned
candidate-conditioned syndrome/codebook packet trained to increase matched
margins while keeping strict controls near the target prior.
