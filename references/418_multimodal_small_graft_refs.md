# Multimodal Small-Graft Refs (2026-04-21)

Purpose: keep a lateral connector queue alive without turning the same-pair
text lane into a new architecture project.

## Strongest Sources

1. FLASH
   Link: https://arxiv.org/abs/2505.12728
   Why it matters: draft/verify sidecars are the cleanest small-graft template
   for accept / reject / refine bridge logic.

2. FlowMM
   Link: https://openreview.net/forum?id=TWUpicQMeS
   Why it matters: information-flow-guided merging is a good precedent for
   selective preserve-plus-tail bridge routing.

3. AlignVLM
   Link: https://openreview.net/forum?id=hnQeoY6NRU
   Why it matters: minimal latent-space bridges can add alignment capacity
   without changing the backbone.

4. M3-JEPA
   Link: https://openreview.net/forum?id=tYwKQMMjJA
   Why it matters: multi-gate separation of shared vs specific information is a
   useful template for interpretable bridge routing.

5. I2MoE
   Link: https://openreview.net/forum?id=EuJaF5QsMP
   Why it matters: interpretable routing matters if later ablations need to
   show which bridge paths were actually used.

6. Cambrian-1
   Link: https://arxiv.org/abs/2406.16860
   Why it matters: dynamic top-k preservation is a strong small-ablation
   template for bridge token or head selection.

7. OpenUni
   Link: https://arxiv.org/abs/2505.23661
   Why it matters: lightweight connector plus query refinement is a good
   fallback if routed repair fails but a small sidecar still looks plausible.

## Exact Next Ablations

1. verifier-gated residual sidecar on top of `dynalign_module_replace_residrank16`
2. top-k preserve plus residual-tail repair with explicit kept-token telemetry

## Current Read

- If routed residual repair fails, the next lateral text-safe branch should be
  a small sidecar or selective-preservation connector, not a full multimodal
  rearchitecture.
