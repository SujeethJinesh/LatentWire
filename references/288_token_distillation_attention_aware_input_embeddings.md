# Token Distillation: Attention-Aware Input Embeddings for New Tokens

- Date: 2026-01-26
- Link: https://openreview.net/forum?id=n20ml5nGEo

## Why it matters here

- It treats token adaptation as an **attention-aware distillation** problem,
  not just a vocabulary bookkeeping problem.
- The useful transplant is the idea that new or remapped tokens should be
  supervised by how they participate in attention, not only by output logits.

## Potential use in LatentWire

- A stronger upstream remapping teacher could combine:
  - token/span overlap,
  - next-token output overlap,
  - and attention-side interaction structure.
- That is a natural follow-up to the current dynamic-alignment branch, which
  already moved beyond raw span overlap into output-aware scoring.

## Current read

- Most relevant as support for a **multi-view token remapping teacher**
  instead of another local bridge-capacity tweak.
