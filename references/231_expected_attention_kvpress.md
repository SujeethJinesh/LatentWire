# Expected Attention / KVPress

- Title: `Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution`
- Date: 2025-10-01
- Link: https://arxiv.org/abs/2510.00636
- Code: https://github.com/NVIDIA/kvpress
- Why it matters here:
  - strongest fast external comparator lane for query-aware KV compression
  - useful as a fair negative-boundary comparator even when our in-repo Expected Attention-style approximation does not separate from its shuffled null
  - should be labeled as exact parity only if we actually run the external KVPress implementation
