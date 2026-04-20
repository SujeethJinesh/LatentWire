# Contextual Dynamic Mapping

- Title: `Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping`
- Date: 2025-02-16
- Link: https://arxiv.org/abs/2502.11104
- Why it matters here:
  - strongest direct reference for dynamic output-side alignment when token spaces do not line up cleanly
  - useful if the next bridge should be supervised by a dynamic output mapping rather than static next-token KL

Most transplantable mechanism:
- align teacher and student outputs with a context-dependent mapping before applying distillation

Immediate use in our setting:
- use it as the main anchor if the next output-side teacher branch adds dynamic token/output alignment on top of the bridge
