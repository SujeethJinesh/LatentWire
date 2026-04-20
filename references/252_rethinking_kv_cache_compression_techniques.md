## Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

- Title: `Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving`
- Date: 2025-03-31
- Link: https://arxiv.org/abs/2503.24000
- Why it matters here:
  - useful paper-side reference for the bytes-vs-accuracy and KV lifecycle framing, especially the warning that memory savings alone do not guarantee throughput or end-to-end latency wins
  - helps justify reporting paired flips, bytes, and latency together instead of only top-line accuracy

Most transplantable mechanism:
- benchmark KV methods with throughput and latency-aware artifacts, not just compression ratio or average accuracy

Immediate use in our setting:
- cite it in the reviewer-facing frontier and KV-lifecycle discussion so the paper reads as serving-aware rather than only accuracy-aware
