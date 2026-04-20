# SHINE

- Title: `SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass`
- Date: 2026-02-06
- Link: https://arxiv.org/abs/2602.06358
- Why it matters here:
  - strongest recent direct precedent for generating low-rank adapters from context rather than fitting one fixed adapter
  - useful if the next live LatentWire bridge should become a true hypernetwork-style generated interface instead of another static correction

Most transplantable mechanism:
- map a compact context/query summary into per-example low-rank adapter coefficients in one pass

Immediate use in our setting:
- use it as the main anchor if the next branch becomes a query-conditioned generated bridge on top of frozen grouped transport
