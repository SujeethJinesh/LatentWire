# Text-to-LoRA

- Title: `Text-to-LoRA: Instant Transformer Adaption`
- Date: 2025-06-06
- Link: https://arxiv.org/abs/2506.06105
- Why it matters here:
  - cleanest recent example of producing a low-rank adapter directly from text/context
  - useful if the bridge should be generated from prompt-local information rather than learned as one global module

Most transplantable mechanism:
- use a lightweight generator to emit adapter parameters conditioned on context instead of learning one fixed adapter

Immediate use in our setting:
- use it as a lightweight literature anchor for prompt-conditioned generated bridge weights
