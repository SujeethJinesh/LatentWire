# M-FISH Gradient Feasibility Probe

Created: 2026-05-28T02:40Z

## Question

Can M-FISH compute per-channel Fisher weights at the decode-time activation
surface without retaining a full 10K-token autograd graph?

## Probe Design

The scratch probe in `.debug/probe_mfish_gradient.py` used:

- frozen model weights,
- a detached prefix cache built under `torch.no_grad()`,
- one gradient-enabled decode step,
- an embedding hook that reintroduces gradient only for the current token,
- forward hooks on transformer layer outputs with retained gradients.

This estimates diagonal Fisher weights for the current-token layer-output
surface while keeping the historical cache detached.

## Results

| Model | Result | Memory | Notes |
|---|---:|---:|---|
| DeepSeek-R1-Distill-Qwen-1.5B | PASS | 3.58 GB allocated | All 28 layer outputs produced nonzero gradients. One-step backward took 0.061 s after prefix cache construction. |
| Granite-4.0-H-Small | FAIL_HYBRID_CACHE_AUTOGRAD | 50.8 GB allocated before failure | Backward fails because hybrid cache tensors are mutated in-place during the gradient-enabled step. Disabling fast paths did not remove the mutation. |
| Falcon-H1-0.5B-Instruct | FAIL_HYBRID_CACHE_AUTOGRAD | not retained | Same in-place cache mutation failure class as Granite. |

Representative DeepSeek all-layer probe:

```json
{
  "captured_layers": 28,
  "loss": 0.06316333264112473,
  "elapsed_seconds": 0.06050372123718262,
  "cuda_memory": {
    "allocated_gb": 3.576095232,
    "max_allocated_gb": 3.579332608,
    "reserved_gb": 3.797942272
  }
}
```

## Decision

M-FISH is feasible for dense Transformer models with detached-cache one-step
Fisher estimation. It is not yet feasible for the hybrid Mamba/MoE cache path
without a more invasive cache clone/no-mutation workaround.

## Next Gate

Implement and run a DeepSeek-focused M-FISH gate first. DeepSeek is the most
informative dense model because M11b top-10 was positive but ambiguous there
while M-PRED killed. A DeepSeek M-FISH PASS would establish a method for an
architecture where budget-tuned EMA did not clearly generalize.
