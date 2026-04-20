# AlignSAE

- Title: `AlignSAE: Concept-Aligned Sparse Autoencoders`
- Date: 2025-12-01
- Link: https://arxiv.org/abs/2512.02004
- Why it matters here:
  - another strong sparse-basis reference for turning cross-model alignment into a concept-slot matching problem instead of dense coordinate matching
  - supports the claim that the remaining math-based pivot is a shared sparse feature interface, not another small rotation tweak

Most transplantable mechanism:
- bind model activations into aligned sparse concept slots before transport or prediction supervision

Immediate use in our setting:
- use it as supporting evidence if we pivot from dense grouped transport to an explicit shared sparse dictionary bridge
