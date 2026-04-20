# KVzip

- Title: `KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction`
- Date: 2025-05-29
- Link: https://arxiv.org/abs/2505.23416
- Why it matters here:
  - strongest current external compression control to separate cross-model communication from generic cache compression
  - now the highest-priority next comparator day after exact KVPress

Most transplantable mechanism:
- compress the KV cache query-agnostically while reconstructing context well enough to preserve downstream behavior

Immediate use in our setting:
- use it as the next honest control baseline if we spend another external-comparator day
