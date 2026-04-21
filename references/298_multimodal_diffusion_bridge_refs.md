# Multimodal / Diffusion Bridge References

Goal: collect recent primary sources on projector interfaces, latent bottlenecks, attention distillation, and cross-modal alignment that can inspire the next LatentWire bridge.

## Most transferable idea

The most actionable pattern is a **small projector or latent bottleneck supervised by target-side refinement signals**. In practice, that means keeping the current transport path frozen and adding a tiny learned interface that is trained with attention, representation, or likelihood distillation rather than raw output KL alone.

## References to add next

1. **X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation** — 2025-03-08
   https://arxiv.org/abs/2503.06134
   Mechanism: a lightweight bridge trained with **attention distillation** into a diffusion-transformer backbone.
   LatentWire ablation: `bridge only` vs `bridge + attention distillation`.
   Cost class: **cheap/local**.

2. **Cross-Tokenizer Distillation via Approximate Likelihood Matching** — 2025-03-25
   https://arxiv.org/abs/2503.20083
   Mechanism: **pure likelihood matching** across different tokenizers without requiring a shared next-token objective.
   LatentWire ablation: `static KL` vs `approximate likelihood matching`.
   Cost class: **cheap/local**.

3. **Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers** — 2025-06-09
   https://arxiv.org/abs/2506.07986
   Mechanism: balances cross-modal attention with token-imbalance-aware weighting and timestep-aware interaction control.
   LatentWire ablation: `uniform bridge strength` vs `query/timestep-weighted bridge`.
   Cost class: **cheap/local**.

4. **Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping** — 2025-02-16
   https://arxiv.org/abs/2502.11104
   Mechanism: contextual remapping fixes sequence misalignment and vocabulary mismatch before distillation.
   LatentWire ablation: `positionwise teacher` vs `contextual remapping teacher`.
   Cost class: **cheap/local**.

5. **AttAnchor: Guiding Cross-Modal Token Alignment in VLMs with Attention Anchors** — 2025-09-27
   https://arxiv.org/abs/2509.23109
   Mechanism: attention anchors act as semantic signposts that route cross-modal alignment.
   LatentWire ablation: `single bridge` vs `anchor-routed bridge`.
   Cost class: **cheap/local**.

6. **PAV-DiT: A Cross-modal Alignment Projected Latent Diffusion Transformer for Synchronized Audio-Video Generation** — 2025-09-15
   https://openreview.net/forum?id=RFrc0g7pSu
   Mechanism: projected latent diffusion with cross-modal alignment and multi-scale attention fusion.
   LatentWire ablation: `static interface` vs `projected latent alignment + multi-scale fusion`.
   Cost class: **future-heavy**.

7. **HiFi-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation** — 2025-09-19
   https://openreview.net/forum?id=72xWFIzG15
   Mechanism: representation alignment guides latent diffusion with dual-stream fusion and balanced text injection.
   LatentWire ablation: `transport only` vs `transport + representation-alignment bridge`.
   Cost class: **future-heavy**.

8. **CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning** — 2025-09-18
   https://openreview.net/forum?id=feuFyonHks
   Mechanism: cyclic information-bottleneck training builds an informative shared latent space across modalities.
   LatentWire ablation: `direct adapter` vs `cyclic shared-latent bridge`.
   Cost class: **future-heavy**.

## Minimal bridge recipe to try next

1. Keep grouped transport frozen.
2. Add a tiny projector after transport.
3. Supervise the projector with one of:
   - attention distillation,
   - target shallow hidden states,
   - or likelihood matching.
4. If routing is needed, use a light anchor/router on top of the projector rather than a full expert bank.

## First ablation ladder

1. transport only
2. transport + tiny projector
3. transport + projector + attention distillation
4. transport + projector + attention distillation + anchor routing

## Excluded as adjacent but less direct

- **TokAlign** and **Model-Aware Tokenizer Transfer**: useful for tokenizer-side alignment, but this file is focused on bridge/projector and diffusion-style interface design.
- **BASIC** and **MASSV**: very useful, but they fit better in the output-aware teacher memo than in the diffusion/projector bucket.
