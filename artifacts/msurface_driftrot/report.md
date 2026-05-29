# M-SURFACE DriftRot Diagnostic

Created: `2026-05-28T15:37:22Z`

## Status

- Paper readiness: not ICLR-ready.
- Current story: rotation is positive across the current model set; M-SURFACE
  asks whether a cleaner internal surface explains or improves drift handling.
- Blocking gap: no internal surface has yet shown lower drift than the
  post-block surface where channel-set protection was originally measured.

A Granite two-trace hook sanity run has now completed:

`experimental/outlier_migrate/phase9/results/om_phase9_msurface_granite_sanity_20260529T0032Z`

## Decision

**Decision: `KILL_OR_DEFER_SURFACE_NO_LOWER_DRIFT`.**

The same-run diagnostic found that the cheap internal module-hook surfaces
drifted more than the post-block control:

| Surface | Mean strict set-leaving | Median strict set-leaving | Mean delta vs same-layer post-block |
|---|---:|---:|---:|
| post-block residual output | 0.578 | 0.585 | 0.000 |
| Mamba `out_proj` input | 0.887 | 0.902 | +0.311 |
| attention `o_proj` input | 0.780 | 0.817 | +0.183 |

No internal surface had strict leaving `<0.30-0.40` or at least `0.15`
absolute lower than post-block. The result argues against promoting this
module-hook M-SURFACE branch to endpoint scoring.

## Hypothesis

Block-output drift may overstate the drift seen at the actual quantized GEMM input surfaces. If an internal surface is more stable, DriftRot or SurfaceProtect should target that surface rather than the post-block residual.

## Promotion Rule

Promote M-SURFACE/SurfaceRot only if the diagnostic shows:

- internal strict set-leaving `<0.30-0.40`, or
- internal strict set-leaving at least `0.15` absolute below the same-run post-block control.

If Granite does not satisfy this gate, skip Falcon M-SURFACE unless a later Falcon result specifically needs placement evidence.

## Next Gate

Do not run Falcon M-SURFACE from this surface family. Reopen surface placement
only with a new local tensor target, such as SSM B/C internals, or with a
KLLOOK-backed hypothesis that the cheap hook surfaces are the wrong candidate
pool.
