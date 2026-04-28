# Source-Private Test-Log Packet Cross-Model Summary

- pass gate: `True`
- passing models: `3/4`
- non-Qwen passing models: `['phi3_mini_helper']`
- interpretation: Helper-line protocol generalizes to Qwen3 and Phi-3, but fails on TinyLlama. This supports protocol-assisted private packet handoff across capable instruction-tuned source models, not universal cross-family extraction.

| Run | Model | Family | Role | Pass | Matched | Target-only | Best control | Valid packets | p50 latency ms |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen25_0_5b_helper | Qwen/Qwen2.5-0.5B-Instruct | qwen2.5 | reference | `True` | 0.938 | 0.250 | 0.250 | 0.919 | 164.86 |
| qwen3_0_6b_helper | Qwen/Qwen3-0.6B | qwen3 | same-vendor/generation | `True` | 1.000 | 0.250 | 0.250 | 1.000 | 334.17 |
| phi3_mini_helper | microsoft/Phi-3-mini-4k-instruct | phi3 | cross-family | `True` | 0.912 | 0.250 | 0.250 | 0.950 | 595.25 |
| tinyllama_1_1b_helper | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama | cross-family-negative | `False` | 0.250 | 0.250 | 0.250 | 0.000 | 496.49 |

## Decision

Promote helper-line test-log packet handoff as cross-model on capable instruction-tuned source models. Keep TinyLlama as a negative capability/control row. Do not claim universal model-agnostic extraction.

Next gate: hidden-test/code-repair variant with the helper-line protocol and the same source-destroying controls.
