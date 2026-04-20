# Attention Editing

- Title: `Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion`
- Date: 2026-04-07
- Link: https://arxiv.org/abs/2604.05688
- Why it matters here:
  - strongest recent template for replacing a weak residual bridge with a materially different attention-side module
  - useful if the next live branch should stop looking like a tiny decoder correction and start looking like a true interface conversion layer

Most transplantable mechanism:
- progressively distill a replacement attention module against the teacher architecture rather than only fitting a small residual in latent space

Immediate use in our setting:
- use it as the main literature anchor if the next branch becomes an attention-side module replacement or a more explicit interface layer on top of grouped transport
