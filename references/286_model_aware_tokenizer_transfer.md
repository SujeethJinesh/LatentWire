# Model-Aware Tokenizer Transfer

- Date: 2025-09-20
- Link: https://openreview.net/forum?id=IyV1QEc95F

## Why it matters here

- It argues that tokenizer transfer should preserve **inter-token
  communication patterns**, not just output probabilities.
- The useful transplant is that token remapping should respect how the model
  uses token neighborhoods, attention structure, and local communication.

## Potential use in LatentWire

- Extend the upstream remapping teacher so it scores candidate alignments by:
  - local span overlap,
  - contextual neighborhood similarity,
  - and token-interaction structure.
- That is exactly the direction suggested by the new contextual-remapping
  failure: raw span overlap helps offline fit, but is still too weak as a
  held-out teacher.

## Current read

- Most relevant as support for a **richer contextual remapping teacher**
  before the bridge, especially one that is aware of token interactions.
