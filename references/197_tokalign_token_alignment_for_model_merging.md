# TokAlign

- Title: `TokAlign`
- Date: 2025-06-04
- Link: https://arxiv.org/abs/2506.03523

Why it matters here:

- It is a clean reference for tokenizer / vocabulary alignment when cross-model
  transfer is blocked upstream of the KV path.
- It is not the leading blocker on the current Qwen pair, but it is relevant
  if the paper broadens to more heterogeneous tokenizers or if we later stack a
  vocabulary-alignment fix on top of transport.
- It belongs in the blocker backlog because the user explicitly called out
  vocabulary size and tokenizer mismatch as possible stacked issues.
