# Product-Codebook Model-Receiver References

Date: 2026-04-30

Purpose: frame the product-codebook model-mediated receiver and masked-PQ
consistency receiver gates. These sources support the conclusion that PQ and
distance-table lookup are established primitives; LatentWire's novelty is the
source-private, rate-capped task-evidence protocol under destructive controls.

## Sources And Experiment Implications

1. Jegou, Douze, and Schmid, "Product Quantization for Nearest Neighbor
   Search," IEEE TPAMI 2011. https://doi.org/10.1109/TPAMI.2010.57
   - blocker helped: prevents overclaiming PQ itself as novel.
   - mechanism/design idea: subspace code indices plus asymmetric distance
     decoding.
   - next-experiment impact: keep deterministic PQ L2 as the canonical
     receiver baseline for product-codebook packets.
   - role: baseline and theory support.

2. Ge et al., "Optimized Product Quantization," CVPR 2013.
   https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
   - blocker helped: situates codebook/rotation variants as established
     compression design space.
   - mechanism/design idea: optimize the transform/codebook before PQ.
   - next-experiment impact: any future PQ improvement should compare against
     rotation/OPQ-style transforms, not claim raw PQ novelty.
   - role: baseline and ablation.

3. van den Oord et al., "Neural Discrete Representation Learning," NeurIPS
   2017. https://papers.neurips.cc/paper/7210-neural-discrete-representation-learning
   - blocker helped: frames discrete codebooks as neural communication
     interfaces.
   - mechanism/design idea: learned discrete latent indices as compact
     intermediate states.
   - next-experiment impact: supports product-codebook packets as a discrete
     latent sidecar, but not as a frozen-model target-decoder claim.
   - role: framing and inspiration.

4. Austin et al., "Structured Denoising Diffusion Models in Discrete State
   Spaces," OpenReview 2021. https://openreview.net/forum?id=h7-XixPCAL
   - blocker helped: motivates corrupt-packet training for receiver
     self-preservation.
   - mechanism/design idea: train over masked/random/permuted discrete
     corruptions.
   - next-experiment impact: directly motivated the masked-PQ consistency
     receiver diagnostic.
   - role: inspiration and ablation.

5. Chang et al., "MaskGIT: Masked Generative Image Transformer," CVPR 2022.
   https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html
   - blocker helped: gives a practical discrete-token denoising analogy.
   - mechanism/design idea: iterative masked-code refinement.
   - next-experiment impact: suggests future few-step PQ receiver refinement
     only if the one-step masked receiver shows headroom.
   - role: inspiration.

6. Song et al., "Consistency Models," ICML 2023.
   https://proceedings.mlr.press/v202/song23a.html
   - blocker helped: motivates one-step distilled receivers rather than
     multi-step decode procedures.
   - mechanism/design idea: consistency under different corruption levels.
   - next-experiment impact: the first one-step masked-PQ receiver collapsed
     or reproduced deterministic L2, so deeper consistency stacks should not be
     prioritized without a new feature surface.
   - role: inspiration and pruning support.

7. Zandieh et al., "TurboQuant," arXiv 2025.
   https://arxiv.org/abs/2504.19874
   - blocker helped: establishes strong modern quantization/compression
     comparators.
   - mechanism/design idea: rotated/scalar/residual quantization for LLM
     tensors and caches.
   - next-experiment impact: LatentWire should frame systems value as boundary
     traffic and source-private control, not as a universal compressor beating
     TurboQuant-style tensor compression.
   - role: baseline and systems framing.

8. Fu et al., "Cache-to-Cache Transfer for LLMs," arXiv 2026.
   https://arxiv.org/abs/2510.03215
   - blocker helped: closest cross-model communication competitor.
   - mechanism/design idea: transfer projected/fused source KV states to the
     target.
   - next-experiment impact: product-codebook packets should be compared as a
     rate-capped/private alternative to cache transfer, with byte and exposure
     accounting, not as the same mechanism.
   - role: competitor and framing.

## Cycle Conclusion

The literature supports product-codebook packets as an interpretable discrete
communication primitive, but not as a novel PQ algorithm. The product-codebook
model-mediated target-decoder smoke failed because the target model ignored the
packet/numeric receiver table. The masked-PQ consistency receiver reproduced
the deterministic L2 decoder only after weighting, and did not improve over it.
The next product-codebook branch should therefore be either a true learned
feature surface that changes the packet geometry, or a scoped systems result
that keeps deterministic PQ as the receiver.
