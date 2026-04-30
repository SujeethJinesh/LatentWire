# Product-Codebook Geometry References

Date: 2026-04-30

Purpose: frame the utility-balanced and OPQ-style product-codebook geometry
gate. This memo supports the decision to treat geometry changes as ablations
unless they beat canonical PQ under the existing source controls.

## Sources And Implications

1. Jegou, Douze, and Schmid, "Product Quantization for Nearest Neighbor
   Search," IEEE TPAMI 2011. https://doi.org/10.1109/TPAMI.2010.57
   - blocker helped: establishes canonical PQ as an existing baseline.
   - mechanism/design idea: split vectors into subspaces, send centroid
     indices, decode with asymmetric distance to public candidates.
   - next-experiment impact: canonical contiguous PQ remains the baseline that
     any geometry variant must beat.
   - role: baseline and theory support.

2. Ge et al., "Optimized Product Quantization," CVPR 2013.
   https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
   - blocker helped: motivates rotating the vector space before PQ.
   - mechanism/design idea: alternate PQ assignment/reconstruction with an
     orthogonal Procrustes rotation update.
   - next-experiment impact: implemented `opq_procrustes` and
     `utility_opq_procrustes` variants.
   - role: method inspiration and ablation.

3. Ash et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence
   and Lattice Codebooks," ICML 2024. https://arxiv.org/abs/2402.04396
   - blocker helped: shows rotations/codebooks are strong modern compression
     primitives and not novel by themselves.
   - mechanism/design idea: transform geometry can determine quantization
     error and hardware behavior.
   - next-experiment impact: reinforces that LatentWire should claim
     source-private task communication, not generic quantization novelty.
   - role: baseline/framing.

4. SpinQuant, "LLM Quantization with Learned Rotations," arXiv 2024.
   https://arxiv.org/abs/2405.16406
   - blocker helped: supports learned/protected rotations as a plausible
     geometry branch.
   - mechanism/design idea: optimize rotations to reduce quantization error.
   - next-experiment impact: if OPQ remains a near-miss, a stronger learned
     rotation objective would need source-control and scalar-WZ comparisons.
   - role: inspiration.

5. QuaRot, "Outlier-Free 4-Bit Inference in Rotated LLMs," arXiv 2024.
   https://arxiv.org/abs/2404.00456
   - blocker helped: systems reviewers may expect rotation-aware quantization
     baselines.
   - mechanism/design idea: orthogonal transforms can make quantization more
     uniform and hardware-friendly.
   - next-experiment impact: report OPQ/protected rotations as comparator
     attempts, not promoted wins.
   - role: systems/quantization baseline.

6. Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
   Information at the Decoder," IEEE TIT 1976.
   https://ieeexplore.ieee.org/document/1055508
   - blocker helped: grounds decoder-side public candidate state as a
     legitimate side-information assumption.
   - mechanism/design idea: rate-capped messages can be decoded with public
     side information.
   - next-experiment impact: keep evaluating packet bytes under target-side
     candidate state and source-destroying controls.
   - role: theory support.

## Cycle Conclusion

Utility-balanced regrouping and OPQ-Procrustes did not beat canonical PQ by the
`+0.03` promotion bar. OPQ did partially repair the known remap-107/budget-2
control failure (`0.512 -> 0.527`, controls clean), but the effect is too small
to become a new contribution. Future geometry work should use a stronger
source-control-trained rotation objective or be reported as a negative ablation.
