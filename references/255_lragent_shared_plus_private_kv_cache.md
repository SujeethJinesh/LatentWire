# LRAgent

- Title: `LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents`
- Date: 2026-02-01
- Link: https://arxiv.org/abs/2602.01053
- Why it matters here:
  - best recent reference for decomposing runtime state into a shared base plus adapter-specific residual
  - directly supports the current shared-plus-private bridge direction instead of a single monolithic translator

Most transplantable mechanism:
- represent the cache or interface as reusable shared structure plus a low-rank private delta

Immediate use in our setting:
- cite it as the closest runtime systems precedent for an asymmetric shared-plus-private KV bridge
