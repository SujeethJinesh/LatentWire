# M-SURFACE DriftRot Diagnostic

Created: `2026-05-28T15:37:22Z`

## Status

- Paper readiness: not ICLR-ready.
- Current story: rotation is positive on Granite and Nemotron; M-SURFACE asks whether a cleaner internal surface explains or improves drift handling.
- Blocking gap: no cached internal-surface activations exist for Granite or Falcon.

No GPU jobs were run. This packet prepares a guarded two-trace diagnostic only.

## Decision

**Decision: `GUARDED_DIAGNOSTIC_ONLY`.**

Cached block-output drift is high:

- Granite block-output strict set-leaving: `0.566`.
- Falcon block-output strict set-leaving: `0.674`.

No cached evidence shows any internal surface with strict leaving `<0.30-0.40` or at least `0.15` below block output. Granite has cheap module hooks for Mamba `out_proj` input and attention `o_proj` input, so Granite remains the recommended first diagnostic. Falcon should stay deferred unless Granite identifies a lower-drift surface or ParoQuant Falcon is weak.

## Hypothesis

Block-output drift may overstate the drift seen at the actual quantized GEMM input surfaces. If an internal surface is more stable, DriftRot or SurfaceProtect should target that surface rather than the post-block residual.

## Promotion Rule

Promote M-SURFACE/SurfaceRot only if the diagnostic shows:

- internal strict set-leaving `<0.30-0.40`, or
- internal strict set-leaving at least `0.15` absolute below the same-run post-block control.

If Granite does not satisfy this gate, skip Falcon M-SURFACE unless a later Falcon result specifically needs placement evidence.

