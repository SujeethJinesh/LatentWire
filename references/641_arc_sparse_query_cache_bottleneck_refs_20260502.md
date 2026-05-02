# References: ARC Sparse-Query Cache-Bottleneck Gate

Web check: 2026-05-02. Scope: primary-source boundary checks for
`results/source_private_arc_challenge_sparse_query_cache_bottleneck_gate_20260502_tinyllama_disagreement/`.

## Local Artifact

- Script:
  `scripts/build_source_private_arc_challenge_sparse_query_cache_bottleneck_gate.py`
- Result:
  `results/source_private_arc_challenge_sparse_query_cache_bottleneck_gate_20260502_tinyllama_disagreement/`
- Pass gate: `False`
- Selected view/bottleneck: `hidden_query_residual / PCA16 / RFF32 / top16`
- Test matched/Qwen-substituted/cached-Tiny mean:
  `0.248203 / 0.317125 / 0.269345`
- Test matched minus Qwen-substituted mean: `-0.068922`
- Test matched minus cached Tiny mean: `-0.021142`
- CI95 lower bound versus Qwen-substituted: `-0.138531`

## Primary Sources And Boundaries

1. Random Features for Large-Scale Kernel Machines
   - https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines
   - Boundary: random Fourier features motivate the nonlinear finite feature
     expansion used in this gate. The LatentWire result is not a kernel-method
     contribution; it is a downstream packet falsification of one nonlinear
     connector family.

2. Perceiver IO
   - https://arxiv.org/abs/2107.14795
   - Boundary: Perceiver IO motivates learned latent queries over large inputs.
     The current gate is a cheap fixed random-query approximation, not a
     trained Perceiver-style receiver.

3. Flamingo and BLIP-2/Q-Former
   - https://arxiv.org/abs/2204.14198
   - https://arxiv.org/abs/2301.12597
   - Boundary: these papers show that narrow connector/query bottlenecks can
     bridge frozen pretrained modules. LatentWire differs by enforcing a
     fixed-byte discrete packet and destructive controls on downstream ARC
     accuracy rather than exposing dense multimodal features to a trained LLM.

4. Cache-to-Cache
   - https://arxiv.org/abs/2510.03215
   - Boundary: C2C is the closest direct model-to-model communication neighbor,
     but it communicates cache/state rather than a source-private fixed-byte
     packet. LatentWire must compare bytes, source exposure, and downstream
     quality against C2C before making systems claims.

5. Relative Representations
   - https://openreview.net/forum?id=SrC-nwieGJ
   - Boundary: relative coordinates are a direct common-basis inspiration. This
     gate keeps the stricter packet boundary and tests whether a TinyLlama
     hidden/query map helps a Qwen receiver on held-out multiple-choice rows.

6. Sparse autoencoders and sparse crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - https://arxiv.org/abs/2502.03714
   - https://arxiv.org/abs/2603.05805
   - Boundary: sparse dictionaries remain a live path for a common language
     across model activations. The current result says a shallow random sparse
     query bottleneck is not enough; it does not falsify trained sparse
     crosscoders.

7. QJL and TurboQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - Boundary: these are byte-frontier and vector-compression references. They
     are systems baselines and sketching inspirations, not source-private
     latent reasoning methods under the LatentWire packet contract.

8. QuaRot
   - https://arxiv.org/abs/2404.00456
   - Boundary: rotation-based quantization supports the broader systems point
     that basis choice matters for compact numeric transfer. It does not solve
     cross-model semantic alignment by itself.

9. VQ-VAE
   - https://arxiv.org/abs/1711.00937
   - Boundary: VQ-VAE motivates learned discrete latent codebooks. LatentWire
     still needs to show that any discrete code helps a different model under
     source-private controls.

10. Consistency Models
    - https://arxiv.org/abs/2303.01469
    - Boundary: consistency/distillation ideas motivate one-step refinement
      alternatives to iterative diffusion-style latent repair. They are future
      connector ideas, not evidence for the current ARC sparse-query gate.

## Method Decision

The negative result rules out a low-data random Fourier sparse-query
bottleneck over the current TinyLlama hidden/query caches. Novelty remains in
the fixed-byte source-private packet protocol and destructive controls, not in
the random feature machinery itself. The next method branch should either add
real learned connector capacity with more matched activations or replace
TinyLlama with a stronger true non-Qwen source.
