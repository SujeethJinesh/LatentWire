# Control-Regularized JEPA Receiver References

- date: `2026-04-30`
- purpose: primary-source framing for the control-regularized JEPA/Q-Former
  receiver gate, novelty pressure test, and systems caveats.

## Sources And Use

1. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture", arXiv `2301.08243`,
   https://arxiv.org/abs/2301.08243.
   - Blocker: learned receiver needs anti-collapse motivation.
   - Mechanism idea: predict compact latent targets without reconstructing full
     inputs.
   - Next experiment change: keep rank/entropy/context-variance diagnostics,
     but do not describe the receiver as JEPA unless it passes controls.
   - Role: inspiration and framing.

2. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
   Image Encoders and Large Language Models", arXiv `2301.12597`,
   https://arxiv.org/abs/2301.12597.
   - Blocker: need a non-hand-coded connector family.
   - Mechanism idea: small query bottleneck over source-side features.
   - Next experiment change: query-resampler remains a fair learned-connector
     baseline, but it needs stronger features or whole-pool training.
   - Role: architecture inspiration and baseline family.

3. Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning",
   arXiv `2204.14198`, https://arxiv.org/abs/2204.14198.
   - Blocker: reviewers may ask why the method is not just a Perceiver-style
     connector.
   - Mechanism idea: resampled source states can condition a frozen target.
   - Next experiment change: keep connector comparisons explicit, but separate
     high-rate feature connectors from byte-capped evidence packets.
   - Role: architecture baseline and framing.

4. Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of
   Large Language Models", ACL Anthology, EMNLP 2023,
   https://aclanthology.org/2023.emnlp-main.825/.
   - Blocker: source-private packets must beat or trade off against compressed
     text relay.
   - Mechanism idea: compare against query-aware extractive text, not only raw
     hidden logs.
   - Next experiment change: systems tables should continue reporting
     query-aware shortest text and prompt-token rows.
   - Role: baseline and reviewer-risk comparison.

5. Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
   Language Models", OpenReview, https://openreview.net/pdf?id=LeatkxrBCi.
   - Blocker: closest direct cross-model communication competitor.
   - Mechanism idea: learned source-to-target cache fusion shows richer
     high-rate communication is possible.
   - Next experiment change: do not claim general latent transfer without
     C2C/KV comparisons; position packets as a lower-rate/private operating
     point.
   - Role: baseline and novelty threat.

6. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", arXiv
   `2402.02750`, https://arxiv.org/abs/2402.02750.
   - Blocker: systems claim needs careful KV compression caveat.
   - Mechanism idea: KV cache can be aggressively quantized, so byte-floor
     comparisons must state access assumptions.
   - Next experiment change: report KV byte floors as accounting rows, not as
     native wins over KV compression.
   - Role: systems baseline/caveat.

7. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization", arXiv `2401.18079`, https://arxiv.org/abs/2401.18079.
   - Blocker: long-context KV systems baselines are strong.
   - Mechanism idea: outlier-aware/non-uniform KV quantization is a serious
     production competitor on memory pressure.
   - Next experiment change: keep LatentWire claim to source-private task
     communication, not generic KV-cache reduction.
   - Role: systems baseline/caveat.

8. "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization", arXiv
   `2406.03482`, https://arxiv.org/abs/2406.03482.
   - Blocker: reviewers may ask for mathematical compression baselines.
   - Mechanism idea: JL/sketching motivates protected residual comparators and
     rate-distortion plots.
   - Next experiment change: use as a systems/math comparator, not a direct
     source-private packet competitor.
   - Role: systems baseline and inspiration.

## Gate Implication

The control-regularized JEPA-Q result is a clean negative. It improves the
paper by pruning a plausible learned-connector story, but it does not solve the
hand-coded receiver objection. The next learned-receiver gate should either use
stronger activation features or train on whole candidate pools with destructive
controls, instead of further threshold tuning.
