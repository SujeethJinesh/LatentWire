# References: ARC Transport Common-Basis Gate

Web check: 2026-05-02. Scope: primary-source boundary checks for
`results/source_private_arc_challenge_transport_common_basis_gate_20260502_tinyllama_disagreement/`.

## Local Artifact

- Script: `scripts/build_source_private_arc_challenge_transport_common_basis_gate.py`
- Result:
  `results/source_private_arc_challenge_transport_common_basis_gate_20260502_tinyllama_disagreement/`
- Pass gate: `False`
- Selected method/view: `whitened_procrustes / query_residual`
- Test matched/Qwen-substituted/cached-Tiny mean:
  `0.228753 / 0.317125 / 0.269345`
- Test matched minus Qwen-substituted mean: `-0.088372`
- CI95 lower bound versus Qwen-substituted: `-0.160677`

## Primary Sources And Boundaries

1. SVCCA
   - https://arxiv.org/abs/1706.05806
   - Boundary: SVCCA motivates low-rank CCA-style representation comparison
     and affine-invariant diagnostics. It does not by itself prove downstream
     source-private communication.

2. CKA
   - https://arxiv.org/abs/1905.00414
   - Boundary: CKA compares similarity structure between neural
     representations. For LatentWire it is useful telemetry for whether two
     views share geometry, not a packet method or receiver improvement claim.

3. Relative Representations
   - https://openreview.net/forum?id=SrC-nwieGJ
   - Boundary: relative coordinates can support zero-shot latent-space
     communication across models. The current ARC transport gate is narrower:
     it keeps a fixed `12B` packet boundary and tests downstream multiple-choice
     accuracy under destructive controls.

4. Latent Space Translation via Semantic Alignment
   - https://openreview.net/forum?id=pBa70rGHlr&noteId=9MWnfMIOv7
   - Boundary: semantic alignment and Procrustes-style maps are relevant
     neighbors. The negative ARC result says this shallow version did not
     transfer TinyLlama hidden/query signal into the public ARC basis.

5. Wasserstein Procrustes
   - https://arxiv.org/abs/1805.11222
   - Boundary: optimal-transport/procrustes alignment motivates geometric
     transport controls. LatentWire should not claim optimal transport unless
     the learned coupling and objective are explicitly optimized and audited.

6. QJL
   - https://arxiv.org/abs/2406.03482
   - Boundary: QJL uses JL projection plus sign-bit quantization for KV/cache
     inner-product fidelity. The ARC gate borrows only the sign-sketch idea as
     a destructive/byte-aware control, not as a KV-cache compression result.

7. TurboQuant
   - https://arxiv.org/abs/2504.19874
   - Boundary: TurboQuant combines online vector quantization with QJL-style
     residual correction for vector/KV compression. It is a systems
     compression neighbor, not source-private cross-model reasoning evidence.

8. Perceiver IO
   - https://arxiv.org/abs/2107.14795
   - Boundary: learned latent queries over large inputs are the strongest next
     connector inspiration because static transport failed. Any LatentWire use
     must still emit a strict discrete packet before claiming source privacy.

9. Flamingo and BLIP-2/Q-Former
   - https://arxiv.org/abs/2204.14198
   - https://arxiv.org/abs/2301.12597
   - Boundary: these papers show narrow learned connector/query bottlenecks
     can bridge frozen pretrained modules. They are architectural inspiration,
     not prior evidence for LatentWire's fixed-byte ARC packet.

10. Sparse autoencoders and sparse crosscoders
    - https://arxiv.org/abs/2309.08600
    - https://transformer-circuits.pub/2024/crosscoders/index.html
    - https://arxiv.org/abs/2502.03714
    - Boundary: shared sparse dictionaries remain the highest-value fallback
      after the static transport failure. Do not claim monosemanticity or a
      universal language unless audited features and held-out transfer support
      that claim.

## Method Decision

The negative result rules out static nearest-neighbor, sign-sketch, and
orthogonal Procrustes repairs over the current TinyLlama ARC hidden/query
caches. The next method branch should change the receiver representation
structure: a small learned query bottleneck, a nonlinear sparse crosscoder, or
a stronger true cross-family source. Keep QJL/TurboQuant-style sketches as
byte-frontier controls rather than the main positive-method claim.
