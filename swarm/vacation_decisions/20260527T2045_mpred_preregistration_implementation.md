# 2026-05-27T20:45Z M-PRED preregistration and implementation

## Decision

Proceed from M-PRED scoop check to a bounded M-PRED experiment.

## Rationale

The scoop check found no direct prior art for one-step predictive protected-channel
selection in W4A16 long-decode reasoning. V2 showed M11b top-10 does not cleanly
generalize to DeepSeek or Falcon-H1, so the next positive-method gate should test
whether predictive tracking improves over lagging EMA on the strongest available
decision surface.

## Implementation scope

- Added `preregister_om_phase9_mpred.md`.
- Added a checker for M-PRED packets with preregistered PASS / AMBIGUOUS / KILL
  criteria.
- Added a runner that reuses validated M11b baseline score caches and cached
  activations, builds M-PRED endpoint protected sets, and spends GPU only on new
  M-PRED/control regimes.
- Refactored activation processing to stream summaries with NumPy accumulators
  rather than materializing all prompt-position-channel vectors.

## Important scope note

The current W4A16 scoring path applies protected sets as static endpoint tensor
exclusions. This experiment evaluates M-PRED's predicted final protected set at
position 10,000 for the 512-token endpoint scoring window. It does not claim a
fully online per-token production kernel.

## Next gate

Run M-PRED Granite first under the 12 GPU-hour experiment cap, then run Nemotron
only if the cumulative budget remains safely below the 340-hour stop threshold.
