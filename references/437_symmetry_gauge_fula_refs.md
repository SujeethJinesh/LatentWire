# Symmetry Gauge FuLA Refs (2026-04-22)

Purpose: capture the strongest math-side ideas that are still distinct from
the already-saturated fixed-rotation and simple wrapper lane.

## Strongest Sources

1. Complete Gauge Symmetries in Transformer Architectures
   Link: https://openreview.net/forum?id=KrkbYbK0cH
   Why it matters: formalizes the symmetry classes we may need to quotient out
   before transport.

2. Curvature Meets Bispectrum
   Link: https://openreview.net/forum?id=pcqyhDvG0i
   Why it matters: suggests invariant diagnostics beyond plain Procrustes
   residuals.

3. ATLAS symmetry discovery
   Link: https://openreview.net/forum?id=VXKt1lwysO
   Why it matters: principled symmetry discovery rather than assuming one fixed
   gauge family.

4. FuLA
   Link: https://arxiv.org/abs/2505.20142
   Why it matters: multi-layer functional latent alignment instead of a single
   stitched interface.

5. I-FuLA
   Link: https://openreview.net/forum?id=hJvcbkf2nO
   Why it matters: further evidence that multi-layer functional alignment is a
   distinct regime from one-layer stitching.

6. Procrustes Bounds and Applications
   Link: https://openreview.net/forum?id=DLEzSo1DIk
   Why it matters: constrained orthogonal alignment with interpretable failure
   modes.

## Exact Next Math-Side Ablations

1. quotient/gauge-invariant canonicalization plus invariant matching on the
   frozen same-pair contract
2. FuLA-style multi-layer functional alignment as a real alternative to the
   single-interface dynalign residual lane
3. whitened orthogonal Procrustes initializer versus affine or OT-style
   initializers at matched repair capacity
4. shortcut controls so improved stitch quality is not confused with true
   reasoning transfer

## Minimal Telemetry

- gauge-fixed residual norm
- singular spectrum / conditioning
- head-match entropy
- layerwise stitch residual
- cue/shortcut leakage controls
- help/harm on the same exact contract IDs

## Current Read

- The strongest underexplored math-side pivots are quotient/gauge-aware
  canonicalization and multi-layer functional alignment, not another static
  rotation or wrapper.
