# Multimodal / Output-Aware Connector Refs (2026-04-21)

Purpose: bank the strongest heterogeneous-connector and output-aware repair
ideas that could plug into the current live lane without collapsing the paper
story into a generic multimodal digression.

## Strongest Sources

1. The Vision Wormhole
   Link: https://arxiv.org/abs/2602.15382
   Why it matters: strongest shared-latent wormhole reference for heterogeneous
   communication through a common connector space.

2. Multimodal Latent Reasoning via Hierarchical Visual Cues Injection
   Link: https://arxiv.org/abs/2602.05359
   Why it matters: iterative latent refinement is a better repair template than
   the current one-shot gate family.

3. Reasoning in the Dark
   Link: https://arxiv.org/abs/2510.12603
   Why it matters: strongest template for interleaving latent and textual
   reasoning over multiple passes.

4. Bridging Modalities via Progressive Re-alignment
   Link: https://arxiv.org/abs/2511.22862
   Why it matters: staged realignment supports a two-pass bridge where the
   first pass stabilizes the connector and the second repairs the answer.

5. Ola
   Link: https://arxiv.org/abs/2502.04328
   Why it matters: progressive modality alignment is a useful curriculum
   template if we later widen beyond same-pair Qwen.

6. AlignVLM
   Link: https://arxiv.org/abs/2502.01341
   Why it matters: latent anchor construction suggests a simple output-aware
   anchor instead of an unconstrained repair head.

7. Alt-MoE
   Link: https://arxiv.org/abs/2409.05929
   Why it matters: route-aware expert conditioning is a better connector
   analogy than generic confidence-only routing.

8. Dense Connector for MLLMs
   Link: https://arxiv.org/abs/2405.13800
   Why it matters: strongest sequence-sidecar reference for keeping richer
   intermediate signals without fully widening the bridge.

## Exact Next Ablations

1. `dynalign_module_replace + byte/sequence sidecar + single residual repair`
   Why now: best direct translation of the wormhole / dense-connector ideas
   into the current live same-pair branch.

2. Two-pass output-aware repair
   Why now: strongest test of whether a second pass fixes semantics or only
   answer formatting on GSM8K32.

3. Output-anchor loss on the repair head
   Why now: strongest lightweight way to make the correction branch care about
   downstream answer structure, not only latent reconstruction.

## Interpretable Telemetry

- whether sidecar activation was confidence-gated
- number of repair passes
- residual norm per pass
- semantic gain vs format-only gain
- extraction coverage before and after repair

## Current Read

- Multimodal connector work supports latent-side repair and sidecars, not a
  wholesale pivot away from the current real lane.
- The key transferable lesson is staged or output-aware repair, not simply
  adding more auxiliary tokens.
