# ICLR Strengthening Scout: Systems Baselines and Third-Method Branches

- date: `2026-04-29`
- blocker: the current source-private packet evidence is strong in scoped
  settings, but reviewers can still call it a protocol-coded evidence lookup
  unless we add stronger systems comparisons and a deeper target-conditioned
  latent method.
- role: literature scout memo for next gates; not a claim of completed
  experiments.

## Systems And Quantization Baselines

1. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
   (`https://arxiv.org/abs/2504.19874`).
   - blocker helped: prevents overclaiming packet bytes as generic KV/cache
     compression superiority.
   - mechanism/design idea: data-oblivious vector quantization plus QJL-style
     residual correction suggests a strong compressed-KV byte floor.
   - next experiment change: add a KV-byte lower-bound table for fp16/int8/4-bit
     /2.5-bit/3.5-bit cache relay versus the 2-byte packet.
   - use: baseline and systems framing.

2. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead (`https://arxiv.org/abs/2406.03482`).
   - blocker helped: tests whether random sign-sketch transport explains the
     packet gains.
   - mechanism/design idea: JL sign sketches can preserve inner products; use
     same-byte random/sign sketches as candidate-feature baselines.
   - next experiment change: add a QJL-style sign-sketch packet ablation only if
     the sparse innovation receiver becomes live.
   - use: baseline and ablation.

3. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
   (`https://arxiv.org/abs/2402.02750`) and KVQuant
   (`https://arxiv.org/abs/2401.18079`).
   - blocker helped: establishes that cache compression is mature and should be
     treated as a competitor axis, not ignored.
   - mechanism/design idea: report cache payload bytes at 2-bit and outlier-
     aware quantization rates for the same prompts.
   - next experiment change: Mac-local accounting table now has higher value
     than trying to reimplement kernels.
   - use: systems baselines.

4. SnapKV (`https://openreview.net/forum?id=poE54GOq2l`) and CacheGen
   (`https://arxiv.org/abs/2310.07240`).
   - blocker helped: separates source-private residual communication from
     query-aware cache pruning/streaming.
   - mechanism/design idea: selected-token cache relay is a strong higher-byte
     comparator; it should be shown on the byte axis.
   - next experiment change: add a cache-streaming lower-bound row to the
     systems frontier.
   - use: baseline and paper framing.

5. NVIDIA GenAI-Perf metric conventions
   (`https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2540/user-guide/docs/perf_analyzer/genai-perf/README.html`).
   - blocker helped: clarifies that current Mac endpoint timings are proxy
     evidence, not serving-throughput proof.
   - mechanism/design idea: mirror TTFT, ITL/TPOT, output throughput, request
     throughput, prompt tokens, payload bytes.
   - next experiment change: normalize endpoint artifact fields before GPU
     serving is available.
   - use: systems evaluation framing.

## Latent-Prediction And Geometry Branches

6. I-JEPA (`https://arxiv.org/abs/2301.08243`), V-JEPA
   (`https://arxiv.org/abs/2404.08471`), V-JEPA 2
   (`https://arxiv.org/abs/2506.09985`), and V-JEPA 2.1
   (`https://arxiv.org/abs/2603.14482`).
   - blocker helped: provides a principled non-generative latent-prediction
     route beyond hand-coded evidence packets.
   - mechanism/design idea: predict masked target-state innovations, not
     answers or full reconstructions.
   - next experiment change: implement a masked source-private innovation gate
     only after the audit ledger/KV-byte table is done.
   - use: inspiration and objective design.

7. DiT (`https://arxiv.org/abs/2212.09748`), SiT
   (`https://arxiv.org/abs/2401.08740`), and Flow Matching
   (`https://arxiv.org/abs/2210.02747`).
   - blocker helped: suggests a cheap one-step residual/velocity predictor
     rather than an expensive diffusion sampler.
   - mechanism/design idea: source packet predicts the target-private
     correction vector from target prior to target evidence state.
   - next experiment change: bounded `source_private_masked_innovation_flow`
     gate with zero/shuffled/target-prior controls.
   - use: method inspiration.

8. Relative Representations
   (`https://openreview.net/pdf?id=SrC-nwieGJ`), model stitching
   (`https://papers.nips.cc/paper/2021/file/01ded4259d101feb739b06c399e9cd9c-Paper.pdf`),
   sparse crosscoders (`https://transformer-circuits.pub/2024/crosscoders/index.html`),
   and cross-architecture crosscoders (`https://openreview.net/forum?id=YXB8uigyOg`).
   - blocker helped: targets the cross-family failure of the learned receiver.
   - mechanism/design idea: transmit only sparse source-private innovation
     atoms relative to target prior and shared anchors.
   - next experiment change: next learned method should be a fold-heldout sparse
     innovation dictionary receiver, not raw coordinate regression.
   - use: method design and interpretability.

## Decision

The most defensible near-term path is:

1. finish the pass/fail audit ledger and sync it into `final/`;
2. add KV/cache byte lower-bound accounting so the systems contribution is
   honestly scoped against TurboQuant/QJL/KIVI/KVQuant/SnapKV/CacheGen;
3. then run one sparse or masked innovation receiver gate with explicit
   target-prior, zero-source, shuffled-source, random same-byte, answer-only,
   and same-byte text controls.

Do not claim generic KV compression superiority or broad cross-family latent
transfer until those gates pass.
