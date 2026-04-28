# Latest Cross-Family Model Scout References

- date: `2026-04-28`
- status: primary-source model-card scout
- blocker: latest/MoE generalization cannot be claimed from Qwen rows alone; we need cheap non-Qwen falsification rows and off-machine architecture-diversity rows.

## Candidate Rows

### `allenai/OLMo-2-0425-1B-Instruct`

- source: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
- blocker helped: tests whether the source-packet emitter is tied to Qwen/Phi chat-template behavior.
- mechanism/design idea: add an open-science 1B instruct row before expensive MoE rows.
- next experiment change: add `OLMo-2-0425-1B-Instruct` as the first cross-family local `n=16` smoke after Qwen3.5 n64.
- role: ablation / external-validity row.

### `google/gemma-3-1b-it`

- source: https://huggingface.co/google/gemma-3-1b-it
- blocker helped: tests a strong non-Qwen small family with different model-card/license/access constraints.
- mechanism/design idea: use a small Gemma row as a cross-family protocol-compliance falsification test.
- next experiment change: queue only after access is accepted and the model is cached locally.
- role: ablation / external-validity row.

### `ibm-granite/granite-3.3-2b-instruct`

- source: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct
- blocker helped: adds an Apache-licensed enterprise/open small instruct row with explicit reasoning tags.
- mechanism/design idea: test whether structured reasoning tag conventions interfere with exact diagnostic-code packet emission.
- next experiment change: add as a local `n=16` row after OLMo/Gemma.
- role: ablation / external-validity row.

### `HuggingFaceTB/SmolLM3-3B`

- source: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- blocker helped: tests a 3B small-model research family with dual-mode reasoning and long context.
- mechanism/design idea: use as a higher-memory local row if 1B/2B non-Qwen rows pass.
- next experiment change: not first; queue after the cheaper rows.
- role: ablation / external-validity row.

### `microsoft/Phi-4-mini-instruct`

- source: https://huggingface.co/microsoft/Phi-4-mini-instruct
- blocker helped: checks whether the existing Phi-3 positive row extends to a successor Phi family.
- mechanism/design idea: run with `float16` if local memory permits, because the row is 3.8B and may require custom-code support.
- next experiment change: queue after Qwen3.5 and one non-Qwen 1B/2B row.
- role: successor-family ablation.

### `mistralai/Ministral-3-3B-Instruct-2512-BF16`

- source: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16
- blocker helped: adds a recent Mistral-family edge/local model with multimodal/text architecture.
- mechanism/design idea: test packet emission under a model marketed for edge deployment and JSON/function-calling behavior.
- next experiment change: off-machine or local only after lighter rows because BF16/MPS support may be fragile.
- role: external-validity row.

### `nvidia/NVIDIA-Nemotron-Nano-9B-v2`

- source: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
- blocker helped: tests whether a hybrid Mamba-Transformer reasoning architecture can emit the same source-private packet.
- mechanism/design idea: architecture-diversity row, not a local Mac row.
- next experiment change: off-machine `n=16` only after cheap rows pass.
- role: architecture-diversity ablation.

### `moonshotai/Kimi-K2-Thinking`

- source: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- blocker helped: non-Qwen sparse-MoE stress test.
- mechanism/design idea: use the same packet benchmark through a remote serving wrapper after Qwen3.6 MoE passes.
- next experiment change: do not run locally; reserve as expensive off-machine stress row.
- role: MoE external-validity / stress row.

## Gate Order

1. Finish `Qwen/Qwen3.5-0.8B` CPU n64.
2. Run `allenai/OLMo-2-0425-1B-Instruct` local n16.
3. Run one of `google/gemma-3-1b-it` or `ibm-granite/granite-3.3-2b-instruct` local n16, depending on cache/access.
4. Only then spend remote cycles on `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.6-35B-A3B-FP8`, `nvidia/NVIDIA-Nemotron-Nano-9B-v2`, or `moonshotai/Kimi-K2-Thinking`.

## Claim Boundary

These rows are a falsification portfolio, not evidence until run. They should
be described as planned external-validity gates unless their result artifacts
exist with the standard source-destroying controls.
