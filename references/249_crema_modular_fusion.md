## CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion

- Title: `CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion`
- Link: https://openreview.net/forum?id=3UaOlzDEt2
- Why it matters here:
  - useful modular-fusion reference when one static bridge is too weak and we need a small set of specialized modules instead of one projector
  - supports a per-layer modular fusion block rather than a single global bridge

Most transplantable mechanism:
- a small modular fusion family where different experts handle different interaction regimes

Immediate use in our setting:
- try a per-layer modular bridge with a few experts instead of one residual adapter family
