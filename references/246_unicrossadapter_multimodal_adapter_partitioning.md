## UniCrossAdapter: Multimodal Adaptation of CLIP for Radiology Report Generation

- Title: `UniCrossAdapter: Multimodal Adaptation of CLIP for Radiology Report Generation`
- Link: https://arxiv.org/abs/2503.15940
- Why it matters here:
  - useful bridge-design reference when one monolithic projector looks too weak and we need per-role or per-layer adapters on top of a frozen backbone
  - supports splitting the tiny bridge into smaller target-facing correction modules instead of assuming one residual map should fix every layer the same way

Most transplantable mechanism:
- distribute lightweight adapters across different interaction roles rather than fitting one global residual bridge

Immediate use in our setting:
- keep the current grouped transport frozen
- if prediction-level bridge supervision still stalls, try per-layer or per-role bridge heads instead of a single shared bridge family
