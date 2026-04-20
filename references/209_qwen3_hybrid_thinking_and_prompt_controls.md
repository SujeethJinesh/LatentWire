## Qwen3 Hybrid Thinking And Prompt Controls

- Dates:
  - Qwen3 blog: 2025-04-29
  - Qwen3 technical report: 2025-05-14
- Links:
  - https://qwenlm.github.io/blog/qwen3/
  - https://arxiv.org/abs/2505.09388
  - https://qwen.readthedocs.io/en/v3.0/getting_started/quickstart.html
  - https://huggingface.co/Qwen/Qwen3-0.6B

Why it matters here:

- official Qwen3 sources say the model defaults to hybrid thinking mode and can
  be switched into non-thinking behavior with `enable_thinking=False`
- that is the cheapest fairness control for the current Qwen2.5 ->
  Qwen3 cross-model communication evaluations
- this is likely a more meaningful prompt/serialization control than tokenizer
  mismatch for the exact pair, because both official configs report the same
  `vocab_size = 151936`
