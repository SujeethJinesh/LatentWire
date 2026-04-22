# Multimodal Verifier Sidecar Refs (2026-04-22)

Purpose: keep multimodal inspiration compact and additive by focusing on
 verifier-gated sidecars, selective preservation, and cheap refinement rather
 than a full architecture pivot.

## Strongest Sources

1. FLASH
   Link: https://arxiv.org/abs/2505.12728
   Why it matters: strongest output-aware verifier/refinement template for a
   compact sidecar on top of an existing bridge.

2. FlowMM
   Link: https://openreview.net/forum?id=TWUpicQMeS
   Why it matters: best precedent for preserving only high-flow bridge states
   instead of repairing everything uniformly.

3. TableMoE
   Link: https://arxiv.org/abs/2506.21393
   Why it matters: useful small-graft reference for confidence-aware routing
   with interpretable bottlenecks.

4. LaViT
   Link: https://arxiv.org/abs/2601.10129
   Why it matters: motivates refining latent trajectories instead of only the
   final output token stream.

5. I2MoE
   Link: https://openreview.net/forum?id=EuJaF5QsMP
   Why it matters: strongest interpretability anchor for interaction-aware
   routing and selection telemetry.

## Exact Next Ablations

1. verifier-gated residual repair on top of `dynalign_module_replace_residrank16`
2. top-k bridge preservation before repair
3. two-pass disagreement-triggered refinement

## Interpretable Telemetry

- gate decision and confidence
- accept/reject/refine rate
- which tokens, heads, or channels were preserved exactly
- residual norm before and after repair
- numeric extraction coverage and exact match
- selection overlap across ablations

## Current Read

- This is not the main same-pair claim yet.
- It is the best compact lateral branch if the routed/value-bank family keeps
  tying rather than improving the frozen contract.
