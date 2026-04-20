# X-Fusion

- Title: `X-Fusion: Introducing New Modality to Frozen Large Language Models`
- Date: 2025-04-29
- Link: https://arxiv.org/abs/2504.20996
- Why it matters here:
  - recent precedent for inserting a small modality/interface module between frozen towers rather than trying to adapt the whole backbone
  - useful if the next LatentWire branch should look like a compact transfer interface instead of another residual bridge

Most transplantable mechanism:
- use a lightweight dual-tower interface module that aligns frozen upstream features to the target model through a dedicated learned adapter rather than through direct pointwise correction

Immediate use in our setting:
- good reference if the next branch becomes a small attention-side transfer interface between transported source-side KV signals and the frozen target model
