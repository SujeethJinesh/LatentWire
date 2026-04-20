# Task-KV: Instruction-Task-Specific KV Cache Reuse

- Date: 2025-01-25
- Link: https://arxiv.org/abs/2501.15113
- Why it matters here: motivates query- or task-conditioned KV reuse rather than one static reuse rule, which matches the current bridge/transport blocker.
- Most useful takeaway: retrieval or reuse policies should depend on the active task/query state, not just an averaged calibration descriptor.
