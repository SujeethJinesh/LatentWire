# AME

- Title: `AME: Aligned Manifold Entropy for Robust Vision-Language Distillation`
- Date: 2025-08-12
- Link: https://arxiv.org/abs/2508.08644
- Why it matters here:
  - recent evidence that a shared manifold plus projection functions can stabilize teacher/student transfer better than plain tokenwise output matching
  - relevant if the next LatentWire branch needs a richer output-space teacher than static or lightly reweighted top-k KL

Most transplantable mechanism:
- learn a shared aligned manifold for teacher and student features, then apply entropy-regularized distillation in that manifold rather than only in raw output space

Immediate use in our setting:
- supports a future dynamic-remapping teacher that supervises a bridge/module in a shared latent output space instead of relying only on positionwise next-token logits
