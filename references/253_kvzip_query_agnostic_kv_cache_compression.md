## KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

- Title: `KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction`
- Date: 2025-05-29
- Link: https://arxiv.org/abs/2505.23416
- Why it matters here:
  - strongest query-agnostic KV compression comparator after KVPress / Expected Attention and a useful adjacent control against our query-conditioned communication story
  - especially relevant for the paper’s external-comparator ladder because it emphasizes reusable compressed cache state across diverse future queries

Most transplantable mechanism:
- rank KV entries by how much they matter for reconstructing the original context, then use that as a reusable compression/control baseline independent of the future query

Immediate use in our setting:
- keep it as the next clean external fallback comparator if we want a query-agnostic control after KVPress and before spending more on new bridge variants
