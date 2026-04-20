# LLM Modules

- Title: `LLM Modules: Knowledge Transfer from a Large to a Small Model using Enhanced Cross-Attention`
- Date: 2025-02-12
- Link: https://arxiv.org/abs/2502.08213
- Why it matters here:
  - direct recent precedent for replacing plain latent transfer with a small cross-attention interface between frozen models
  - useful if the next LatentWire branch should stop looking like a residual bridge and start looking like an explicit attention-side module

Most transplantable mechanism:
- insert a compact cross-attention transfer module between frozen source-side states and the smaller target model, then distill through that module rather than only through pointwise corrections

Immediate use in our setting:
- anchor the next “materially different interface” branch if we move from projector-style bridges to a tiny attention-side transfer module
