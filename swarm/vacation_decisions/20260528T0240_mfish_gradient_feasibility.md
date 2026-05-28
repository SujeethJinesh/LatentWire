# M-FISH Gradient Feasibility Decision

Date: 2026-05-28T02:40Z

## Decision

Proceed with a dense-Transformer M-FISH implementation first, using DeepSeek-R1-Distill-Qwen-1.5B as the gate model.

## Evidence

- DeepSeek detached-cache probe produced nonzero gradients for all 28 layer outputs with about 3.58 GB allocated memory and 0.061 s one-step backward time after prefix cache construction.
- Granite and Falcon probes failed during backward with autograd in-place mutation errors from hybrid cache state tensors.
- Disabling Granite fast paths before model load did not remove the cache mutation failure.

## Implication

The earlier M-FISH deferral was too broad. M-FISH is not ready for the hybrid Mamba/MoE models, but it is viable for dense Transformer models. The next useful gate is whether Fisher-weighted selection improves DeepSeek, where M11b remained ambiguous and M-PRED killed.

## Constraints

- Do not claim M-FISH works on Granite, Nemotron, or Falcon until the hybrid cache mutation problem is solved.
- Keep hybrid cache workaround work secondary unless DeepSeek M-FISH passes and needs cross-architecture extension.
- Treat a DeepSeek M-FISH PASS as an architecture-fill result, not a universal positive method.
