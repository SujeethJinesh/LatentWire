# References: Next LatentWire Decision Literature Pass

Date: 2026-05-02

## Current Status

- Current paper readiness: COLM plausible; ICLR blocked.
- Current story: fixed-byte source-private packets, public-basis ARC/OpenBookQA,
  byte/exposure accounting, and a falsification ladder.
- Exact gap: a positive learned/cross-family connector or a stronger true
  non-Qwen source beating controls.

## Local Evidence Read

- `paper/source_private_iclr_colm_readiness_update_20260502.md`
- `paper/reviewer_feedback.md`
- `results/source_private_pass_fail_ledger_20260429/pass_fail_ledger.md`
- `results/source_private_native_readiness_ledger_20260501/native_readiness_ledger.md`

Current saturated or weakened branches:

- scalar receiver/source confidence routing on ARC disagreement rows;
- cached candidate-syndrome connector;
- Mac-local Phi-3 source packets;
- TinyLlama hidden/query PCA-ridge, Procrustes/transport, and RFF sparse-query
  connectors;
- HellaSwag receiver-improvement claims over packet-only.

Still alive:

- stronger true non-Qwen source under the ARC/OpenBookQA packet contract;
- learned query/cache connector trained against target loss;
- systems boundary/accounting claims that compare bytes and source exposure,
  not native GPU throughput.

## Primary Sources Checked

### Direct Cross-Model Communication And Cache Transfer

1. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Primary: https://openreview.net/forum?id=LeatkxrBCi
   - ArXiv: https://arxiv.org/abs/2510.03215
   - Relevance: closest direct competitor. It learns projection/fusion over
     source and target KV caches, with target-layer gating.
   - Decision impact: any learned LatentWire connector must be compared against
     C2C at matched task, bytes/exposure, and training budget.

2. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Primary: https://openreview.net/forum?id=F7rUng23nw
   - ArXiv: https://arxiv.org/abs/2510.03346
   - Relevance: selective layer/KV-pair sharing baseline for multi-LLM
     communication.
   - Decision impact: cite as the training-free KV-sharing peer. LatentWire's
     novelty must be source-private fixed-byte packets or a smaller learned
     connector, not "KV is a communication medium."

3. KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs
   - Primary: https://arxiv.org/abs/2604.13226
   - Relevance: current cache-reuse work using lightweight trainable soft-token
     adapters around immutable KV packets.
   - Decision impact: if LatentWire uses "packet" language for systems claims,
     distinguish fixed task packets from cache packets plus soft-token adapters.

4. RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse
   - Primary: https://arxiv.org/abs/2603.13289
   - Relevance: recent multi-agent cache reuse baseline that reuses decoding KV
     caches and selectively recomputes sparse deviations.
   - Decision impact: useful systems comparator for collaboration settings, but
     it still transfers/reuses KV state rather than source-private task packets.

### Latent Alignment, Common Bases, And Local Structure

5. Relative representations enable zero-shot latent space communication
   - Primary: https://openreview.net/forum?id=SrC-nwieGJ
   - ArXiv: https://arxiv.org/abs/2209.15430
   - Relevance: anchor-relative representation baseline for invariance to
     latent isometries and rescalings.
   - Decision impact: LatentWire should not claim anchor-relative novelty unless
     it is downstream-task, source-private, and benchmark-backed.

6. From Bricks to Bridges: Product of Invariances to Enhance Latent Space
   Communication
   - Primary: https://openreview.net/forum?id=vngVydDWft
   - Relevance: product-space invariance construction for latent communication
     and zero-shot stitching across modalities/models.
   - Decision impact: static invariance/common-basis repair is well-covered
     prior art; the new branch needs learned downstream communication.

7. The Platonic Representation Hypothesis
   - Primary: https://arxiv.org/abs/2405.07987
   - Relevance: motivates possible cross-model representation convergence.
   - Decision impact: only use as motivation, not evidence.

8. Revisiting the Platonic Representation Hypothesis: An Aristotelian View
   - Primary: https://arxiv.org/abs/2602.14486
   - Relevance: argues calibrated global similarity can disappear while local
     neighborhood similarity remains.
   - Decision impact: supports cutting global-linear/common-basis claims after
     the TinyLlama Procrustes/RFF failures; if revived, use local-neighborhood
     or query-conditioned losses.

### Sparse Dictionaries, SAEs, And Crosscoders

9. Scaling and evaluating sparse autoencoders
   - Primary: https://arxiv.org/abs/2406.04093
   - Relevance: SAE scaling and evaluation baselines for sparse bottlenecks.
   - Decision impact: a sparse connector should be evaluated as a bottleneck and
     interpretability object, not just a compression trick.

10. Quantifying Feature Space Universality Across Large Language Models via
    Sparse Autoencoders
    - Primary: https://arxiv.org/abs/2410.06981
    - Relevance: SAE feature-space universality across LLMs.
    - Decision impact: useful motivation for shared sparse features, but not a
      downstream communication result.

11. Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning
    - Primary: https://arxiv.org/abs/2504.02922
    - Relevance: crosscoder best practices, including BatchTopK, for shared
      dictionaries and model-specific artifacts.
    - Decision impact: if LatentWire uses a crosscoder, include artifact
      controls so model-specific latents are not overclaimed.

12. Cross-Architecture Model Diffing with Crosscoders
    - Primary: https://arxiv.org/abs/2602.11729
    - Relevance: cross-architecture crosscoder method, directly relevant to
      non-Qwen source differences.
    - Decision impact: promising analysis layer for a learned connector, but it
      is model diffing, not proof of communication.

13. Sparse Crosscoders for diffing MoEs and Dense models
    - Primary: https://arxiv.org/abs/2603.05805
    - Relevance: crosscoders jointly model multiple activation spaces and expose
      shared versus model-specific features.
    - Decision impact: relevant if the next true non-Qwen source is MoE or if
      LatentWire claims dense-to-MoE transfer.

### Query Bottlenecks And Connector Architecture

14. Perceiver IO: A General Architecture for Structured Inputs & Outputs
    - Primary: https://openreview.net/forum?id=fILj7WpI-g
    - ArXiv: https://arxiv.org/abs/2107.14795
    - Relevance: flexible learned querying over arbitrary inputs/outputs.
    - Decision impact: architectural precedent for fixed query bottlenecks
      instead of global latent alignment.

15. Flamingo: a Visual Language Model for Few-Shot Learning
    - Primary: https://arxiv.org/abs/2204.14198
    - Relevance: Perceiver Resampler plus gated cross-attention bridges frozen
      pretrained modules.
    - Decision impact: supports a small learned resampler as a serious connector
      branch.

16. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders
    and Large Language Models
    - Primary: https://proceedings.mlr.press/v202/li23q/li23q.pdf
    - ArXiv: https://arxiv.org/abs/2301.12597
    - Relevance: Q-Former uses learned queries as an information bottleneck
      between frozen modules.
    - Decision impact: strongest prior for a trainable query/cache connector
      that outputs a small fixed interface.

### Systems, Quantization, And Serving Baselines

17. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Primary: https://arxiv.org/abs/2406.03482
    - Relevance: QJL transform plus sign-bit quantization for KV-cache inner
      product preservation.
    - Decision impact: mandatory byte-floor and sketch baseline; do not claim
      systems wins without comparing against QJL-style cache compression.

18. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Primary: https://arxiv.org/abs/2504.19874
    - Relevance: online vector quantization with random rotation and residual
      QJL for KV-cache quality at low bit rates.
    - Decision impact: mandatory systems citation for rate-distortion and
      cache-compression baselines.

19. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
    - Primary: https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf
    - Relevance: Hadamard rotations remove outliers and enable end-to-end
      low-bit inference including KV cache.
    - Decision impact: cite for rotation/quantization geometry, but do not
      imply LatentWire's public-basis packets are native low-bit inference.

20. KV Cache Transform Coding for Compact Storage in LLM Inference
    - Primary: https://iclr.cc/virtual/2026/poster/10008708
    - Relevance: PCA/decorrelation, adaptive quantization, and entropy coding
      for reusable KV-cache storage.
    - Decision impact: include in the next systems memo as a cache-storage
      competitor, especially for offloaded cache scenarios.

21. Efficient Memory Management for Large Language Model Serving with
    PagedAttention
    - Primary: https://arxiv.org/abs/2309.06180
    - Relevance: vLLM/PagedAttention serving substrate and KV-memory management.
    - Decision impact: native NVIDIA rows should target vLLM-compatible
      measurements, but Mac-local claims should stay accounting-only.

22. SGLang: Efficient Execution of Structured Language Model Programs
    - Primary: https://arxiv.org/abs/2312.07104
    - Relevance: RadixAttention and structured runtime for KV cache reuse.
    - Decision impact: cite as second native serving substrate for future
      measurements.

23. LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference
    - Primary: https://arxiv.org/abs/2510.09665
    - Relevance: open-source KV cache offloading and cross-engine/cache
      transfer for vLLM and SGLang.
    - Decision impact: useful systems baseline for cache movement/exposure,
      separate from accuracy communication.

### Privacy And Source Exposure

24. I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant
    LLM Serving
    - Primary: https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/
    - Relevance: KV-cache sharing can create prompt leakage side channels in
      multi-tenant serving.
    - Decision impact: supports a source-exposure table as a real systems
      contribution, separate from throughput.

25. Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference
    - Primary: https://arxiv.org/abs/2508.08438
    - Relevance: SafeKV frames the efficiency/privacy tradeoff for KV-cache
      sharing.
    - Decision impact: cite when claiming LatentWire exposes no source KV state
      across the boundary.

26. OptiLeak: Efficient Prompt Reconstruction via Reinforcement Learning in
    Multi-tenant LLM Services
    - Primary: https://arxiv.org/abs/2602.20595
    - Relevance: cache-based prompt leakage risk may be stronger than earlier
      estimates.
    - Decision impact: reinforces why "source KV exposed" should be a first-
      class systems column.

## Answers For The Next Decision

### 1. What Would Be Novel

A stronger true non-Qwen source is novel only if it beats the Qwen-substituted
packet/control rows on frozen test disagreement surfaces while preserving
source privacy and fixed-byte accounting. Merely showing another source can
emit labels or score-like packets is not enough after the Phi-3 failure.

A learned query/cache connector is novel only if it is:

- target-loss trained, not just static Procrustes/CCA/PCA/RFF transport;
- fixed-rate, with a rate-distortion curve over query/byte budgets;
- source-private by construction, with no raw source KV crossing the boundary;
- cross-family or at least same-family plus one strict cross-family
  falsification;
- compared against C2C, KVComm, KV Packet/cache-reuse, and KV quantization byte
  floors.

The clearest publishable shape is a Q-Former/Perceiver-style bottleneck over
source hidden/query/cache summaries that emits a small public packet or
receiver-consumable query state, trained on target decision loss with
destructive source controls.

### 2. Mandatory Competitor Baseline Or Citation

The next memo must include C2C and KVComm as direct communication baselines, and
TurboQuant/QJL as systems byte-floor baselines. If space allows, add KV Packet,
KVTC, LMCache, and RelayCaching to show awareness of the newest cache-reuse and
cache-storage systems literature.

### 3. What To Cut Or Not Claim

- Do not claim HellaSwag receiver improvement over packet-only.
- Do not claim robust cross-model latent reasoning from TinyLlama hidden/query
  or sparse-query connector gates.
- Do not claim global latent alignment/common-basis success.
- Do not claim native GPU throughput, HBM traffic, or vLLM/SGLang wins without
  native NVIDIA measurements.
- Do not call source-private fixed-byte packets equivalent to KV/cache transfer;
  the boundary difference is the point.

### 4. Systems-Side Win Without Native NVIDIA

A reviewer-acceptable systems-side win without native NVIDIA is a strict
boundary/accounting result:

- bytes crossing the boundary;
- source KV exposure flag;
- minimum source-state byte floor for C2C/KVComm/KV quantizers;
- packet byte range and source-private status;
- Mac-local packed-record or endpoint proxy only as a transport sanity check.

Frame it as "LatentWire transmits a fixed-byte task packet with no source KV
exposure, whereas cache-transfer and quantization baselines move or store
source/internal KV state." Do not frame it as a throughput win.

## Decision

Highest-priority next branch: train a fixed-rate query/cache connector on a
strict ARC/OpenBookQA disagreement surface, or rerun the source-family gate with
a materially stronger true non-Qwen source if NVIDIA access makes that cheaper.

Do not widen to new benchmarks until one of those clears paired uncertainty,
source-destroying controls, and a strict cross-family falsification pair.
