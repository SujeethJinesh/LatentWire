# TokAlign: Efficient Vocabulary Adaptation via Token Alignment

- Date: 2025-06-04
- Link: https://arxiv.org/abs/2506.03523

## Why it matters here

- It treats vocabulary mismatch as an **alignment problem** rather than a
  downstream adapter-capacity problem.
- The useful transplant is the idea that token alignment can be built first,
  then reused as a stable interface for later distillation or adaptation.

## Potential use in LatentWire

- Build a stronger **token/span remapping stage** before the local bridge:
  - start from raw prompt spans,
  - refine them into a contextual alignment map,
  - then fit the bridge on top of those aligned token groups.
- This supports moving from hard same-position pairing to a learned or
  calibrated **token remapping interface**.

## Current read

- Most relevant as support for the next live lane:
  **contextual token/span alignment before the bridge**, not another local
  bridge-capacity tweak.
