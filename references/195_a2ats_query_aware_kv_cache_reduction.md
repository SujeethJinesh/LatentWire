# A2ATS

- Title: `A²ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary Position Embedding and Query-Aware Vector Quantization`
- Date: 2025-02-18
- Link: https://arxiv.org/abs/2502.12665

Why it matters here:

- It is a direct query-aware KV reference rather than another static cache score.
- The paper is useful for motivating transport costs that target attention-score
  preservation under the live query, not just offline key geometry.
- It is one of the closest recent references for moving from static
  calibration-time descriptors toward genuinely query-conditioned transport.
