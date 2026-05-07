# Cross-Layer Quantization Error Compounding — Theoretical Preregistration

**Frozen on**: 2026-05-07
**Frozen by**: human author (before any data observed)

## Hypothesis

A closed-form upper bound F(N, σ_block, σ_outlier, depth_pattern) on
accumulated FP4 quantization error across N layers in hybrid models
exists and is tight to within 2× of empirically measured BF16-vs-FP4
output drift.

## Falsifiable predictions

- **PASS (bound is tight)**: derived bound F is within 2× of measured
  drift at all four depths {1, 5, 10, 15}.
  Decision string: PASS_CLE_BOUND_TIGHT
- **KILL (bound is loose or wrong)**: at any depth, the bound is >5×
  measured drift, OR the bound predicts a different functional form
  (e.g., bound is constant when measured drift grows linearly).
  Decision string: KILL_CLE_BOUND_LOOSE

## Model

- ibm-granite/granite-4.0-h-tiny

## Promptset

- Source: AIME-2025
- Count: 12 traces
- SHA-256: <COMMITTED TO RESULT PACKET>

## Procedure

### Theoretical derivation (preregistered, before measurement)

Derive an upper bound F on the L2 norm of the difference between BF16
and FP4 outputs after N layers, as a function of:
- N: number of quantized layers
- σ_block: per-block quantization variance (NVFP4 / MXFP4)
- σ_outlier: outlier-channel variance contribution
- depth_pattern: which layers are quantized vs left BF16

The bound must be derivable from first principles before any measurement.
Submit the derivation to the result packet at:
  experimental/cross_layer_error/results/<run_id>/derivation.md

### Empirical validation

1. Run Granite-4.0-H-Tiny in BF16 and record output logits at decode
   position 1000 for all 12 traces.
2. For each depth N in {1, 5, 10, 15}, quantize the first N layers to
   FP4 (NVFP4) and re-run inference. Record output logits.
3. Compute measured drift = mean L2 distance between BF16 and FP4 logits
   at depth N.
4. Compute predicted drift from the bound F(N, ...).
5. Apply decision rule above.

## Statistical readout

- Per-prompt drift, then mean across 12 prompts.
- Bootstrap (n=1000) for 95% CI.
- Bound is tight if predicted/measured ∈ [1.0, 2.0] at every depth.
- Bound is loose if predicted/measured > 5.0 at any depth.

## Forbidden inputs

- Must not adjust the bound after observing measurements (no fitting to
  data).
- Must not change the depth grid {1, 5, 10, 15} post-hoc.
- Must not select a subset of layers for quantization based on observed
  drift.

## On pass

Theoretical paper. Optional MLSys 2027 follow-up implementing
layer-wise precision allocation guided by the bound.

## On kill

Write experimental/KILLED_cross_layer_error_bound_loose/README.md.
The derivation document is preserved as a record of the attempted
approach and reasons it failed.
