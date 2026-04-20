## ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding

- Date: 2025-09-17
- Link: https://arxiv.org/abs/2509.15235

Why it matters here:

- useful multimodal bridge reference where a lightweight adaptor injects a
  global conditioning signal into a stronger decoder path
- suggests a transferable pattern for cross-model KV communication: frozen
  transport plus a small query-conditioned bridge instead of transport-only
- relevant to the current adapter lane because it gives a concrete lightweight
  architectural precedent without requiring a full retraining pivot
