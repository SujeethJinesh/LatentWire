# UniFusion

- Title: `UniFusion: Vision-Language Model as Unified Encoder in Image Generation`
- Date: 2025-10-14
- Link: https://arxiv.org/abs/2510.12789
- Why it matters here:
  - recent example of pooling and reusing multi-layer frozen representations through a compact learned interface
  - relevant if LatentWire needs a richer interface over multiple transported signals instead of a single shallow residual correction

Most transplantable mechanism:
- use compact layerwise pooling over frozen upstream representations before a small downstream interface module consumes them

Immediate use in our setting:
- motivates a later bridge that pools multiple transport-side signals into a compact attention/module replacement interface instead of using only one local residual path
