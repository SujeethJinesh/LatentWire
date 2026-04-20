## AdapterTune: Zero-Initialized Low-Rank Adapters for Frozen Vision Transformers

- Date: 2026-03-16
- Link: https://arxiv.org/abs/2603.14706

Why it matters here:

- clean recent reference for a tiny zero-initialized low-rank adapter on top of
  a frozen backbone
- useful precedent for a bridge/projector that starts as a no-op and only
  learns the residual mismatch left by transport
- supports the current hypothesis that the next positive-method shot is more
  likely to be a small learned bridge than another selector heuristic
