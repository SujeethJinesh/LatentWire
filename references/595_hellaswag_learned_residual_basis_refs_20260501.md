# References: HellaSwag Learned Residual Basis Scout

## Purpose

This memo records the prior-work boundary for the train-only learned residual
basis scout. The branch passes one strong slice but fails all-slice promotion,
so it should be framed as a diagnostic that motivates richer sparse/crosscoder
dictionaries.

## Primary Sources

- Sparse Crosscoders learn shared feature dictionaries across layers/models and
  separate shared from model-specific features. This is the closest inspiration
  for the next branch, but PCA residual bases are much weaker than a trained
  sparse crosscoder.
  Source: https://transformer-circuits.pub/2024/crosscoders/index.html

- Sparse autoencoders revealing universal feature spaces motivate the common
  dictionary direction. The current PCA scout is not an SAE and should not be
  described as one.
  Source: https://arxiv.org/abs/2410.06981

- Universal Sparse Autoencoders motivate cross-model dictionary learning, but
  LatentWire still needs a task-gated source-private packet result before making
  an ICLR common-basis claim.
  Source: https://arxiv.org/abs/2502.03714

- Relative Representations motivate anchor-relative coordinates. The negative
  anchor results explain why learned dictionaries are now higher priority than
  hand-designed anchor charts.
  Source: https://arxiv.org/abs/2209.15430

- C2C and KVComm remain the core non-text communication comparators because they
  transfer or fuse KV/cache state. The learned residual basis here does not
  transmit raw hidden coefficients or KV state.
  Sources: https://arxiv.org/abs/2510.03215 and https://arxiv.org/abs/2510.03346

## Non-Claim Boundary

Do not claim that PCA, dictionary learning, SAEs, or crosscoders are new. Do not
claim cross-model generality yet. The safe statement is:

> A train-only learned residual basis recovers more source-private hidden signal
> than anchors or random sign sketches on one HellaSwag slice, but it does not
> survive the five-slice promotion gate.

## Experimental Outcome

Single-slice scout:

- best variant: `pca256_top2_gold`
- accuracy: `0.528320`
- best label-copy: `0.500000`
- delta vs label-copy: `+0.028320`
- CI95 low: `+0.010742`
- score-only: `0.497070`
- scout pass: true

Five-slice promotion:

- pass gate: false
- strict pass slices: `2/5`
- weighted selected accuracy: `0.476172`
- weighted best label-copy: `0.461523`
- weighted delta vs label-copy: `+0.014648`
- min CI95 low vs label-copy: `-0.017578`

Decision: keep learned-basis as a live diagnostic, but require a richer
sparse/crosscoder objective before making a positive common-basis claim.
