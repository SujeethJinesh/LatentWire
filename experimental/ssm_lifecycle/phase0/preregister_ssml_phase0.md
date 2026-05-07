# SSM-State Lifecycle Compression — Phase 0 Preregistration

**Frozen on**: 2026-05-07
**Frozen by**: human author (before any data observed)

## Hypothesis

In hybrid Mamba-Transformer models, the recurrent SSM state at the end of
a long reasoning trace is statistically distinguishable from state at the
beginning. Older state can be compressed more aggressively than fresh
state.

## Relationship to killed SSQ-LR branch

This branch is conceptually adjacent to the killed SSQ-LR branch but
materially different. SSQ-LR tested cross-model transfer of a static
mixed-precision recipe per layer. SSM-State Lifecycle tests age-based
distributional drift of state within a single trace. If Phase 0 fails
similarly to SSQ-LR's S3 cross-model gate (effect too small or
non-existent), this branch is killed cleanly with a fresh KILL manifest;
do not reopen as SSQ-LR.

## Falsifiable predictions

- **PASS (state ages)**: KS-test p<0.01 on at least 50% of layers
  comparing state distributions at decode positions 100 vs 10000, AND
  magnitude drift ratio ≥2× on at least 50% of layers.
  Decision string: PASS_SSML_PHASE0_STATE_AGES
- **KILL (state stable)**: thresholds not met.
  Decision string: KILL_SSML_PHASE0_STATE_STABLE

## Model

- ibm-granite/granite-4.0-h-tiny

## Promptset

- Source: AIME-2025
- Count: 12 traces (shared dump)
- SHA-256: <COMMITTED TO RESULT PACKET>

## Procedure

1. Shared dump pass captures SSM state tensors at decode positions
   {100, 500, 1000, 5000, 10000} for every Mamba layer in
   Granite-4.0-H-Tiny, for all 12 traces.
2. For each Mamba layer, run two-sample Kolmogorov-Smirnov test
   comparing flattened state distributions at position 100 vs position
   10000. Compute p-value.
3. For each Mamba layer, compute magnitude drift ratio:
   mean(|state_10000|) / mean(|state_100|).
4. Apply decision rule above.

## Statistical readout

- KS test per layer per trace, then aggregate across traces (Fisher's
  combined p-value or per-layer Bonferroni-corrected).
- Drift ratio per layer per trace, then median across traces.
- Pass requires both criteria satisfied on ≥50% of Mamba layers.

## Forbidden inputs

- Must not change the {100, 500, 1000, 5000, 10000} position grid post-hoc.
- Must not change the ≥50% layer threshold post-hoc.
- Must not change the 2× drift ratio threshold post-hoc.
- Must not exclude layers post-hoc based on observed effect.

## On pass

Append ssm_lifecycle_phase1 to queue.yml. Phase 1 extends to
Granite-4.0-H-Small with longer traces (50K positions) and adds
age-conditioned compression recipe.

## On kill

Write experimental/KILLED_ssm_lifecycle_state_stable/README.md.
Explicitly note in the kill manifest that this is NOT a re-kill of
SSQ-LR; the failure mode is different (stable distributions vs failed
cross-model transfer).
