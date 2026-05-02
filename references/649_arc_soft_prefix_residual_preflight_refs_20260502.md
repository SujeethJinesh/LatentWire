# ARC Soft-Prefix Residual Preflight References

Date: 2026-05-02

## Purpose

This memo supports the row-centered residual soft-prefix follow-up. The
experiment asks whether choice-relative source evidence is a better input to a
target-loss connector than absolute selected-choice hidden features.

## Primary Sources Checked

- Prefix-Tuning is the frozen-LM continuous-prefix baseline. LatentWire must
  differ by using per-example source-conditioned prefixes, not task-level
  learned prefixes. Source: https://arxiv.org/abs/2101.00190
- Perceiver IO motivates flexible learned query bottlenecks over structured
  inputs and outputs. Source: https://arxiv.org/abs/2107.14795
- Flamingo motivates resampling external features into a frozen/partly frozen
  language-model interface. Source: https://arxiv.org/abs/2204.14198
- BLIP-2 / Q-Former is the closest query-connector precedent for bridging
  frozen encoders and frozen LMs. Source: https://arxiv.org/abs/2301.12597
- LLM Augmented LLMs / CALM composes models through learned cross-attention and
  is a direct LLM-to-LLM learned-composition boundary. Source:
  https://arxiv.org/abs/2401.02412
- Relative Representations motivates anchor/common-basis comparison under
  representation invariances; LatentWire should cite it as a baseline and
  motivation, not as a solved mechanism. Source:
  https://openreview.net/forum?id=SrC-nwieGJ
- Discovering Latent Knowledge / CCS motivates contrastive hidden directions,
  but choice-centered residuals are communication features, not truth probes.
  Source: https://arxiv.org/abs/2212.03827
- Cache-to-Cache and KVComm remain required cache-transfer competitors because
  they communicate model state directly. Sources:
  https://arxiv.org/abs/2510.03215 and https://arxiv.org/abs/2510.03346

## Novelty Boundary

The residual soft-prefix result is still not a positive method. The defensible
future claim would be:

> A source-conditioned, per-example, rate-capped soft-prefix/query connector
> trained on target loss, whose input is row-centered choice evidence and whose
> gain disappears under source-destroying controls while surviving static
> prefix, same-byte text, score-only, same-family, and cross-family baselines.

## Reviewer-Critical Ablations

- Source necessity: zero source, shuffled source, wrong-row source, same-norm
  noise, train-mean source, and label-shuffled training.
- Feature factorization: absolute selected hidden, row-centered hidden
  residual, score-only, score plus residual, and tokenwise query pooling.
- Connector identity: static prefix, source-conditioned soft prefix, Q-Former
  style query connector, C2C/KVComm cache-transfer competitor, and same-byte
  visible text.
