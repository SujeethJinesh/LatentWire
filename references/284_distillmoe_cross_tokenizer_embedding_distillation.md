# DistillMoE: Multi-Faceted Knowledge Distillation for Cross-Tokenizer Embedding Models

- Date: 2025-09-18
- Link: https://openreview.net/forum?id=VIYNWGb3TL

## Why it matters here

- It treats cross-tokenizer transfer as a **mixture of distillation views**
  rather than a single token-level loss.
- The useful transplant is not the exact embedding-model setting, but the idea
  that token alignment, hidden-state agreement, and output-space supervision
  can be mixed rather than collapsed into one fixed teacher.

## Potential use in LatentWire

- Upstream **token/span remapping** could choose between:
  - span-overlap alignment,
  - contextual hidden-state alignment,
  - output-likelihood alignment.
- The bridge fit could then weight those views per prompt or per layer instead
  of assuming one globally correct token pairing.

## Current read

- This is most relevant as support for a **contextual remapping teacher**
  rather than another local bridge-capacity tweak.
