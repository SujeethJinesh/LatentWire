# Reference Memo 716: Target-Error Syndrome And Decoder Side Information

Date: 2026-05-04

## Current Paper Status

Paper readiness: not ICLR-ready. The target-error syndrome branch has useful
oracle headroom, but no learned positive method yet.

Current story: LatentWire should be framed as a source-private,
target-loss-conditioned packet protocol. The packet is not a generic compressed
latent; it is a tiny per-example syndrome decoded against target-side
information.

Exact blocker: convert the source-unique top-2 repair mass into held-out
accuracy gains over fixed packet, target-only, target-side top-2, same-byte,
and source-destroying controls.

## Primary Sources And Boundaries

1. Slepian and Wolf, "Noiseless Coding of Correlated Information Sources,"
   IEEE Transactions on Information Theory, 1973.
   https://www.mit.edu/~6.454/www_fall_2001/kusuma/slepwolf.pdf
   - Establishes distributed coding for correlated sources, where separate
     encoders can be jointly decoded below naive independent rates.
   - LatentWire is not novel because it uses correlated side information; the
     novelty must be the controlled LLM instantiation and task-loss packet.

2. Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
   Information at the Decoder," IEEE Transactions on Information Theory, 1976.
   https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
   - Establishes lossy compression when only the decoder has side information.
   - This is the cleanest theoretical analogy: minimize downstream target
     decision distortion, not source-state reconstruction error.

3. Dietterich and Bakiri, "Solving Multiclass Learning Problems via
   Error-Correcting Output Codes," JAIR, 1995.
   https://arxiv.org/abs/cs/9501101
   - Establishes multiclass prediction through distributed output codewords.
   - LatentWire should not claim output coding itself is new. Our boundary is
     per-example source-conditioned packet codes, decoded by another model
     under source-private controls.

4. Allwein, Schapire, and Singer, "Reducing Multiclass to Binary: A Unifying
   Approach for Margin Classifiers," JMLR, 2000.
   https://jmlr.csail.mit.edu/papers/v1/allwein00a.html
   - Establishes margin-based decoding for output-code reductions.
   - Useful for receiver design: decode candidate actions by margin/cost, not
     only by exact label-copy.

5. Chow, "On Optimum Recognition Error and Reject Tradeoff," IEEE Transactions
   on Information Theory, 1970.
   https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff
   - Establishes the error/reject tradeoff that underlies selective use.
   - LatentWire's receiver should be selective: defer to source packet only
     when expected gain beats expected harm.

6. Mozannar and Sontag, "Consistent Estimators for Learning to Defer to an
   Expert," ICML, 2020.
   https://proceedings.mlr.press/v119/mozannar20b.html
   - Establishes learning a predictor plus rejector/defer policy from expert
     decisions.
   - Direct overlap: learned deferral. Novel only if the "expert" is not a full
     model decision but a source-private fixed-byte packet.

7. Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
   Communication," ICLR, 2023.
   https://openreview.net/forum?id=SrC-nwieGJ
   - Establishes anchor-relative coordinates as a common-basis strategy across
     representation spaces.
   - Useful for future common-language packets. Not enough by itself unless it
     improves target-error repair under destructive controls.

8. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
   Image Encoders and Large Language Models," ICML, 2023.
   https://proceedings.mlr.press/v202/li23q.html
   - Establishes learned query bottlenecks between frozen encoders and frozen
     language models.
   - Architectural precedent only. LatentWire's novelty is not "use queries";
     it is fixed-byte source-private LLM-to-LLM repair.

9. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   https://arxiv.org/abs/2510.03215
   - Closest dense latent/KV communication competitor.
   - LatentWire should contrast extreme-rate source-private packets with cache
     transfer/fusion rather than claiming generic latent communication novelty.

## Reviewer-Safe Framing

Use: "target-loss-conditioned Wyner-Ziv/ECOC syndrome packet."

Avoid: "we invented syndrome coding," "soft latents are new," or "we beat
cache-transfer systems." The current evidence supports an oracle surface and a
research branch, not a positive method.

## Next Method Gate

Train a harm-controlled receiver only on fixed-hybrid-error/source-top2-headroom
patterns from official train. Evaluate on frozen held-out Qwen-to-Phi slices.
Promote only if it:

- beats fixed Qwen-hybrid by at least `0.005` with positive paired CI;
- beats candidate-only and Phi target-only;
- beats Phi-local top-2 target-side controls or cleanly isolates source-unique
  rows;
- beats source-row-shuffle, candidate-roll, random same-byte, target-derived,
  source-index/rank/score, and label-permuted controls;
- helps more than it harms and is nonnegative on each slice.
