# PQ Model-Receiver Literature Scout

Date: 2026-04-30

## Status

Current paper readiness: strong scoped positive-method evidence, not
comfortable full ICLR. The current story is source-private packet
communication with public decoder side information: a source sends a tiny
residual/evidence packet, and the target selects among public candidates. The
submission blocker is still a less protocol-shaped, product-codebook-specific
model-mediated receiver or an explicit narrowed claim boundary.

Local evidence read before this scout:

- Product-codebook packets pass source-control and paired-uncertainty gates on
  the current remapped surfaces; deterministic public-table PQ L2 remains the
  best PQ receiver.
- Geometry-mitigated PQ/OPQ/Hadamard reduces lookup-risk or improves
  public-mean sensitivity without removing the protocol-shaped caveat.
- Frozen Qwen binary-logprob verification is strong for 2-byte diagnostic
  packets, but the PQ-specific prompt/model receiver ignored numeric packet
  evidence or merely reproduced deterministic L2.

## Primary Sources

- Cache-to-Cache, arXiv:2510.03215:
  https://arxiv.org/abs/2510.03215
  - Closest broad cross-model semantic communication competitor; sends and
    fuses source KV/cache state and reports accuracy and latency gains over text.
  - Threat: can subsume broad "latent communication" claims.

- KVComm, arXiv:2510.03346:
  https://arxiv.org/abs/2510.03346
  - Selective KV-pair sharing with attention-importance layer selection.
  - Threat: reviewer will ask why packet bytes are preferable to selected KV.

- Q-KVComm, arXiv:2512.17914:
  https://arxiv.org/abs/2512.17914
  - Adaptive layer-wise compressed KV communication.
  - Threat: systems comparison must count bytes, selected layers, metadata,
    and source-KV exposure explicitly.

- DroidSpeak, arXiv:2411.02820:
  https://arxiv.org/abs/2411.02820
  - Cross-LLM KV cache reuse for same-architecture models, with throughput and
    prefill-speed claims.
  - Threat: systems reviewers will compare against KV reuse/serving, not only
    against text relay.

- Communicating Activations Between Language Model Agents, arXiv:2501.14082:
  https://arxiv.org/abs/2501.14082
  - Activation-level inter-agent communication by injecting one model's
    intermediate activation into another model.
  - Threat: direct activation communication covers the broad "models talk via
    latents" framing.

- Relative Representations, ICLR 2023 / arXiv:2209.15430:
  https://openreview.net/forum?id=SrC-nwieGJ
  https://arxiv.org/abs/2209.15430
  - Anchor/similarity representations for latent-space communication invariant
    to isometries/rescalings.
  - Use: cite as geometry/anchor precedent; do not claim anchor-like public
    signatures are new.

- Product Quantization, TPAMI 2011:
  https://doi.org/10.1109/TPAMI.2010.57
  - Base subspace codebook index primitive.

- Optimized Product Quantization, CVPR 2013:
  https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html
  - Rotation/space-decomposition optimization for PQ.

- TurboQuant, arXiv:2504.19874:
  https://arxiv.org/abs/2504.19874
  - Modern rate-distortion and inner-product quantization baseline.

- QJL, arXiv:2406.03482:
  https://arxiv.org/abs/2406.03482
  - 1-bit JL/sign sketch for KV inner-product estimation.

- KIVI, ICML 2024 / arXiv:2402.02750:
  https://arxiv.org/abs/2402.02750
  - 2-bit asymmetric KV-cache quantization with throughput/memory claims.

- First-token MCQ mismatch, ACL Findings 2024 / arXiv:2402.14499:
  https://arxiv.org/abs/2402.14499
  - Warns that first-token option likelihood can misrepresent final answers.
  - Directly supports keeping choice-token likelihood as a diagnostic, not the
    headline receiver.

- Grammar-Aligned Decoding, NeurIPS 2024 / arXiv:2405.21047:
  https://arxiv.org/abs/2405.21047
  - Constrained decoding can distort model probabilities unless the constrained
    distribution is handled carefully.

- JSONSchemaBench, arXiv:2501.10868:
  https://arxiv.org/abs/2501.10868
  - Structured/constrained decoding is now a standard target-consumption
    baseline with real efficiency/coverage/quality tradeoffs.

- CRANE, ICML 2025:
  https://openreview.net/forum?id=wKs9fHYxCV
  - Strict final-answer constraints can harm reasoning; reasoning-augmented
    constrained decoding is a safer analogy for candidate selection.

## Novelty Boundary

Novel:

- The evidence object is a 2-4 byte source-private packet, not source text,
  source activations, or source KV/cache.
- The receiver uses public candidate side information; destructive controls
  test whether gain vanishes under shuffled, random, permuted-code,
  answer-masked, and same-byte text replacements.
- The systems table accounts for boundary bytes, cache-line/DMA floors,
  receiver compute, source-text exposure, and source-KV exposure.
- Product-codebook packets are interpretable learned discrete residual
  packets evaluated for task-causal candidate selection, not ANN recall or KV
  reconstruction.

Not novel:

- PQ/OPQ/codebook indices, Hadamard/rotation quantization, QJL/sign sketches,
  constrained decoding, verifier/reranking, and KV-cache communication are all
  established primitives.

## Subsumption Risks

- C2C/KVComm/Q-KVComm subsume any broad cross-model latent/cache communication
  claim unless the paper centers source-private byte-scale packets and exposure
  accounting.
- PQ/OPQ/TurboQuant subsume the codec if framed as better vector
  quantization. The paper must frame PQ as a communication packet under
  decoder side information and source-causal controls.
- Constrained-decoding/verifier work subsumes the target-consumption mechanism
  unless LatentWire shows packet-specific causality and hard same-byte controls.
- Query-aware text/full-log relay can subsume the method at higher bytes when
  private text exposure is allowed; the claim is a privacy/rate operating
  point, not universal superiority.

## Cheap Mac Ablations

1. PQ binary verifier over public distance/rank facts:
   ask one yes/no question per candidate using binned PQ distance/rank/margin
   features; score `yes-no` logprob with fallback threshold. Compare to
   deterministic L2, target, shuffled packet, permuted-code, random same-byte,
   wrong-codebook, and same-byte text.

2. Distance-table wording sweep:
   run n32/n64 prompt variants that expose only ordinal ranks, binned distances,
   signed margin buckets, or top-byte contribution buckets. Kill any wording
   that collapses to option priors before n256.

3. Choice-token sanity guard:
   keep `choice_logprob` as a negative control and report option-prior entropy,
   because first-token MCQ scoring is known to be fragile.

4. Public-table derangement for PQ:
   keep the real packet but rotate candidate distance/rank rows. A receiver
   that follows evidence should fail or select the deranged candidate; a
   target-prior receiver should not move.

5. Collision generalization:
   evaluate only repeated-payload subsets for protected Hadamard/utility
   Hadamard PQ and report collision-subset accuracy vs target/control. This is
   the cheapest anti-lookup stress.

6. OPQ/Hadamard seed repeat:
   rerun n256/n500 for one held-out remap seed with the existing deterministic
   receiver. This is cheaper and more reviewer-useful than another prompt-only
   PQ receiver.

7. Tiny listwise score adapter with hard controls:
   train a CPU linear/MLP candidate scorer over public PQ features plus packet
   code embeddings, but require controls to stay within target +0.03 and
   require deterministic-L2 preservation or improvement. If it only reproduces
   L2, prune as already observed.

## Decision

The product-codebook/PQ receiver is unique only under the scoped
source-private, byte-limited, decoder-side-information protocol. The next
highest-value gate is not free-form target generation; it is a constrained
binary/logprob PQ verifier or packet-conditioned score denoiser that consumes
public distance/rank evidence and passes the existing destructive controls.
