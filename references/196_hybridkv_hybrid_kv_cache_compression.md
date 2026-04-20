# HybridKV

- Title: `HybridKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference`
- Date: 2026-04-07
- Link: https://arxiv.org/abs/2604.05887

Why it matters here:

- It explicitly separates static and dynamic heads instead of treating all KV
  heads as one homogeneous budget pool.
- That is directly relevant to the current blocker that query-conditioned head
  behavior keeps differing from the best static transport map.
- It is a good reference for paper framing around head heterogeneity and
  dynamic allocation.
