# Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking

- Link: https://arxiv.org/abs/2506.09944
- Why it matters here: argues that retrieval behavior is query-conditioned and concentrated in specific heads, which supports moving our transport cost into a retrieval-head-aware representation rather than raw head IDs or mean attention templates.
- Most useful takeaway for `latent_bridge`: build transport descriptors from query-conditioned retrieval behavior, not just calibration-time average attention mass.

