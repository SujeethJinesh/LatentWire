# DeltaKV

- Title: `DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity`
- Date: 2026-02-08
- Link: https://arxiv.org/abs/2602.08005
- Why it matters here:
  - strongest recent public-code residual-style compression control adjacent to our transport-plus-correction story
  - useful as the next comparator day after `C2C` if we want to separate heterogeneous communication from modern residual cache compression

Most transplantable mechanism:
- compress around a residual/update view of the cache rather than treating every retained element as an independent static selection decision

Immediate use in our setting:
- use as the next external control harness after KVPress because it is closer to our “frozen base plus correction” family than older pruning-only controls
