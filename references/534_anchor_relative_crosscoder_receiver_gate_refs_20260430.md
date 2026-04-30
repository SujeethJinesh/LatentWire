# Anchor-Relative Crosscoder Receiver Gate References

Date: 2026-04-30

## Claim Boundary

The current positive paper story should be "source-private residual coding with
decoder side information," not a broad claim that anchor-relative
representations or sparse crosscoders already solve cross-model latent
reasoning.

## Primary Sources

- Relative representations: https://arxiv.org/abs/2209.15430 and
  https://openreview.net/forum?id=SrC-nwieGJ
  - Role: inspiration and baseline for anchor-similarity packet spaces.
  - Effect on our work: anchor-relative coordinates are not novel by
    themselves; novelty must come from source-private rate caps, destructive
    controls, and systems accounting.

- Model stitching: https://arxiv.org/abs/2106.07682
  - Role: representation-compatibility baseline and diagnostic.
  - Effect on our work: a learned receiver should be compared against
    stitching/linear-map style interfaces before claiming cross-model latent
    transfer.

- Sparse crosscoders: https://transformer-circuits.pub/2024/crosscoders/index.html
  - Role: inspiration for shared and model-specific feature decompositions.
  - Effect on our work: top-feature knockout and shared/private feature
    attribution are mandatory diagnostics.

- Universal sparse autoencoders: https://arxiv.org/abs/2502.03714
  - Role: support for shared feature dictionary hypotheses.
  - Effect on our work: motivates a future shared-dictionary receiver, but does
    not make the current anchor-relative packet positive.

- Feature-space universality with SAEs: https://arxiv.org/abs/2410.06981
  - Role: support for cross-model feature reuse and universality tests.
  - Effect on our work: reviewers will expect synonym/ontology and cross-family
    held-out tests, not just same-table accuracy.

- Wyner-Ziv coding with decoder side information:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  - Role: theory framing.
  - Effect on our work: the target has public prompt/candidate state, so the
    source should communicate only residual private evidence.

- Slepian-Wolf coding: https://www.scholarpedia.org/article/Slepian-Wolf_coding
  - Role: theory framing for correlated source/target observations.
  - Effect on our work: supports separated source encoding plus joint decoding,
    but only as framing, not as novelty.

- KIVI: https://openreview.net/forum?id=L057s2Rq8O
  - Role: KV-cache compression baseline.
  - Effect on our work: systems claims must compare against compressed KV
    baselines and report hardware transfer floors.

- KVQuant: https://arxiv.org/abs/2401.18079
  - Role: KV-cache quantization baseline.
  - Effect on our work: "why not just compress KV?" is a required reviewer
    comparison.

- TurboQuant: https://arxiv.org/abs/2504.19874
  - Role: quantization/systems inspiration for mixed-precision residual packets.
  - Effect on our work: motivates the next TurboResidual/PQ packet gate.

- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
  - Role: serving baseline and TTFT/TPOT/goodput framing.
  - Effect on our work: Mac-local byte accounting is useful, but production
    claims need serving telemetry.

- DistServe: https://arxiv.org/abs/2401.09670
  - Role: serving disaggregation and SLO metric framing.
  - Effect on our work: future NVIDIA runs should report TTFT, TPOT, goodput,
    and batch effects.

## Reviewer-Expected Ablations

1. Leakage controls: zero-source, shuffled-source, random same-byte,
   answer-only, answer-masked, public-only sidecar, target-derived sidecar,
   feature-ID permutation, exact ordered-ID parity, and label/slot remapping.
2. Representation controls: raw hashed, anchor-relative, learned
   anchor-relative, sparse dictionary/crosscoder, top-feature knockout, random
   feature knockout, synonym/ontology stress, and held-out family directions.
3. Systems controls: matched-byte structured text, full structured text oracle,
   KV-cache compression baselines, packet cache-line/DMA floors, latency,
   TTFT/TPOT where available, and batch amortization.
