# SSM Shape-Conditioned Codec Diagnostic

**Date**: 2026-05-08
**Source packet**: `experimental/ssm_shape_codec/phase0/results/ssc_phase0_20260508T173705Z`
**Decision**: `KILL_SSC_PHASE0_NO_CODEC_GAIN`

## Paper Readiness

This branch is not ICLR/COLM-ready and should not receive a paper draft. The
positive-method story remains OutlierMigrate plus any future branches that clear
their preregistered gates.

## Proximate Failure

The failure is **hypothesis too weak at the preregistered effect size**. The
shape-conditioned codec improved offline SSM-state reconstruction, but the mean
relative NMSE reduction was `0.07133119147037506`, below the preregistered
`0.10` pass threshold. The bootstrap 95% CI was
`[0.06882070525106689, 0.073901076081768]`, so the miss is not a noisy boundary
case.

The packet is artifact-complete and the checker applied the frozen decision rule
mechanically.

## Fairness Check

- Held-out AIME indices: `12-23`
- Calibration indices: `12-17`
- Test indices: `18-23`
- Mamba layers: `36`
- Positions: `{100, 10000}`
- Metric rows: `432`
- Artifact complete: true
- CI lower bound `>0.05`: true
- Method non-worse at both positions: true
- Mean reduction `>=0.10`: false

The gate was fair. Rerunning seeds, relaxing the `0.10` threshold, or
cherry-picking layers and positions would be p-hacking.

## Unexpected Pattern

There is a real but insufficient offline reconstruction effect. Position `100`
nearly reached the shelf with relative NMSE reduction
`0.09574283944620796`, while position `10000` reached only
`0.04691954349454216`. Prompt-level means were stable, and a few layers had
large reductions, including layer `39` at `0.25735352382423055` and layer `18`
at `0.19733187866818575`.

Those pockets are diagnostic only. They cannot be used to rescue this branch.

## Pivot Assessment

The following fresh positive-method hypotheses are possible in principle:

1. Calibration-gated layer-position codec: choose shape-conditioned versus
   pooled codebooks using calibration-only criteria, then test on fresh prompts
   or a fresh model.
2. Position-continuous quantile transport instead of two independent codebooks.
3. Metadata-efficient sparse codec that applies shape codebooks only where a
   calibration split predicts a large gain, with a subsequent decode-quality
   gate.

None is strong enough to justify immediate continuation in this swarm cycle.
The correct action is to stop pivoting this branch now and return to the
existing priority queue. A future depth-2 pivot would require a fresh
preregistration on a genuinely new calibration-defined surface before any new
held-out data are inspected.
