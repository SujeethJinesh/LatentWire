# Quest

- Title: `Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference`
- Date: 2024-06-16
- Link: https://arxiv.org/abs/2406.10774
- Why it matters here:
  - useful honest control for query-aware KV selection when the paper needs another external comparator day beyond KVPress
  - makes it easier to separate “query-aware pruning helps” from “cross-model communication helps”

Most transplantable mechanism:
- rank KV entries by query-aware importance and prune aggressively under a fixed memory budget

Immediate use in our setting:
- keep it behind KVzip in comparator priority, but use it as the next fallback external control if we spend more than one additional comparator day
