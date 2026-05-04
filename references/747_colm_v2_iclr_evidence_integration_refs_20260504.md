# Reference Memo 747: COLM_v2 / ICLR Evidence Integration Boundaries

Date: 2026-05-04

## Purpose

This memo records the primary-source boundaries for the COLM_v2 evidence table
and the next ICLR method gate. The live repo evidence says the obvious
Mac-local learned bottleneck/resampler variants are saturated. The paper should
therefore backport the current controlled evidence into COLM_v2 while reserving
the ICLR claim for a genuinely new source-causal interface.

## Direct Competitors

1. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   - OpenReview: https://openreview.net/forum?id=LeatkxrBCi
   - arXiv/PDF: https://openreview.net/pdf?id=LeatkxrBCi
   - Boundary: C2C projects and fuses source KV-cache into the receiver cache
     with a learned gate. LatentWire should not claim to beat C2C on raw
     accuracy, latency, HBM, or energy without native measurement. The
     defensible distinction is low-rate, source-private packet transfer with
     strict destructive controls and utility per byte.

2. DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM
   Serving.
   - arXiv: https://arxiv.org/abs/2411.02820
   - Boundary: cache sharing/reuse is a strong serving-side comparison. This
     raises novelty risk for any same-family cache-reuse framing. LatentWire
     should emphasize source exposure, packet bytes, and cross-model semantic
     transfer rather than compatible-cache reuse.

3. KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based
   Multi-agent Systems.
   - arXiv: https://arxiv.org/abs/2510.12872
   - Boundary: multi-agent KV communication is prior art. LatentWire's
     differentiator must be a source-private low-rate packet protocol evaluated
     against source-destroying controls, not just "agents communicate through
     cache."

## Learned Bottleneck Connector Priors

4. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders
   and Large Language Models.
   - arXiv: https://arxiv.org/abs/2301.12597
   - Boundary: Q-Former-style learned query bottlenecks between frozen systems
     are established. A LatentWire learned resampler is only a component, not
     the novelty.

5. Flamingo: a Visual Language Model for Few-Shot Learning.
   - arXiv: https://arxiv.org/abs/2204.14198
   - Boundary: Perceiver Resampler plus gated cross-attention is established as
     a fixed latent-token bridge into a frozen LM. LatentWire must differentiate
     via model-to-model source-private communication, not by the connector shape
     alone.

6. Perceiver IO: A General Architecture for Structured Inputs & Outputs.
   - arXiv: https://arxiv.org/abs/2107.14795
   - Boundary: latent query arrays are prior art for scalable input/output
     interfaces. They are useful motivation for future interfaces but not a
     standalone contribution.

7. LLaVA: Visual Instruction Tuning.
   - arXiv: https://arxiv.org/abs/2304.08485
   - Boundary: simple projection from a frozen encoder space into an LLM input
     space is established. LatentWire needs stricter source-causal evidence than
     "a projector improves downstream behavior."

## Low-Rate / Quantized Cache Baselines

8. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   - arXiv: https://arxiv.org/abs/2504.19874
   - Boundary: TurboQuant is a strong rate-distortion and KV quantization
     baseline. LatentWire can borrow rate-distortion language, but should
     compare against TurboQuant/KV byte floors and avoid unmeasured throughput
     claims.

9. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization.
   - arXiv: https://arxiv.org/abs/2401.18079
   - Boundary: KVQuant compresses one model's cache. LatentWire transfers
     source evidence across models, so the access model differs; byte and
     exposure accounting must make this distinction explicit.

## Implication For The Next Gate

The next ICLR method gate should not be another local variant of chunk encoders,
query resamplers, source-hidden soft prefixes, residual slots, score selectors,
or scalar integrity thresholds. Those are already represented in the ledger.

The next method needs at least one genuinely new source-causal information path,
for example:

- tokenwise extraction rather than mean hidden summaries;
- supervised cross-model alignment from source traces to target intervention
  points with strict wrong-row and target-derived controls;
- an information-theoretic packet with explicit redundancy/error detection that
  changes target behavior only when the packet is source-causal;
- a benchmark where source correctness and target uncertainty are separable
  before training the receiver.

Until that exists, the highest-value paper work is COLM_v2 integration: core
positive packet rows, systems byte accounting, and negative saturation rows that
make the claim boundaries credible.
