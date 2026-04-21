# 390. Unsaturated Canonicalization And Quantized Residual References

Date: 2026-04-21

This memo captures the highest-signal recent ideas that still look unsaturated
relative to the current LatentWire lane. The useful split is:

1. make the shared basis more canonical and gauge-stable per example
2. make the transmitted coefficients more compressible without breaking that
   symmetry

## Primary sources

1. RECON
   - Link: https://openreview.net/forum?id=bpWzTPDybh
   - Date: January 2026
   - Read: The useful move is a lightweight test-time canonicalization layer.
     For LatentWire, this suggests a per-example canonicalization step before
     quotient matching rather than only a family-level canonical gauge.

2. Curvature Meets Bispectrum
   - Link: https://openreview.net/forum?id=GnjpMOXIkV
   - Date: October 2025
   - Read: Gauge-invariant correspondence scoring is the important idea here.
     It points directly at replacing the current match score with a more
     symmetry-aware comparison that should be less brittle across examples.

3. Transporting Task Vectors across Different Architectures without Training
   - Link: https://arxiv.org/abs/2602.12952
   - Date: February 2026
   - Read: Functional residual transport can matter more than raw activation
     matching. This is a plausible next low-shot adaptation layer on top of the
     current shared-basis branch.

4. SPARC
   - Link: https://arxiv.org/abs/2507.06265
   - Date: July 2025, revised March 2026
   - Read: Global TopK support sharing plus cross-reconstruction is the most
     relevant part. It suggests a stronger shared sparse dictionary objective
     than the current local spherical-kmeans-style route.

5. CommVQ
   - Link: https://arxiv.org/abs/2506.18879
   - Date: June 2025
   - Read: The key idea is codebook design that respects algebraic structure.
     For LatentWire, the useful question is whether coefficient compression can
     commute with the shared-basis rotational structure better than scalar
     quantization or byte sidecars.

6. LoRaQ
   - Link: https://arxiv.org/abs/2604.18117
   - Date: April 2026
   - Read: Quantized low-rank compensation is a closer fit to the current
     “repair is inert” blocker than another unconstrained repair module.

7. Reasoning with Latent Tokens in Diffusion Language Models
   - Link: https://arxiv.org/abs/2602.03769
   - Date: February 2026
   - Read: Small latent planning channels can buy global structure under a
     tight token budget. This remains relevant if the current sequence-aligned
     sidecar still fails to move real benchmarks enough.

8. Latent-DARM
   - Link: https://arxiv.org/abs/2603.09184
   - Date: March 2026
   - Read: The relevant lesson is not “use diffusion,” but “a tiny latent plan
     channel may outperform richer textual coordination.”

## Most useful next ablations

1. Per-example canonicalization plus gauge-invariant matching
   - Replace the current global quotient match score with a gauge-invariant
     score and a lightweight per-example canonicalization step.
   - Track head-match stability, `1-2` shot MSE, and real-pair route stability.

2. Functional residual transport on shared-basis coefficients
   - Keep quotient + GPA + sparse dictionary fixed.
   - Fit a closed-form residual transport on coefficient deltas as the only
     low-shot adaptation layer.

3. Support-tied sparse dictionary
   - Add Global TopK support tying and cross-reconstruction to the shared
     sparse basis.
   - Track atom Jaccard, dead-atom rate, atom entropy, and downstream accuracy.

4. Symmetry-compatible coefficient codebook
   - Replace scalar/byte-only coefficient compression with a codebook that
     preserves the shared-basis algebra better.
   - Compare against the current byte-sidecar lane at matched bytes.

5. Quantized low-rank correction
   - Add a rank-`r` quantized residual on top of the shared-basis bridge.
   - Compare “bytes spent on correction” versus “bytes spent on sidecar or
     more atoms”.

## Current prioritization

If only two new branches are worth opening after the frozen GSM8K smoke, the
best candidates are:

1. per-example canonicalization plus gauge-invariant matching
2. symmetry-compatible codebook compression plus quantized low-rank correction
