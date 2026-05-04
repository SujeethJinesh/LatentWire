# Sparse Resonance Packet Quantization, Confidence Coding, And Systems References

Date: 2026-05-04

## Current Gate Read

- Paper readiness: ICLR is still blocked. Sparse Resonance Packets have a clean
  interface and byte/accounting story, but the current sparse PCA,
  common-basis SAE, quantized score, and denoising syndrome gates do not beat
  strict paired controls.
- Current story: LatentWire should be positioned as a low-rate,
  source-private, interpretable packet interface, not as dense cache fusion.
- Exact blocker: a packet receiver must beat target-only, candidate/source
  index, source-score/rank, target-derived, same-byte text/random,
  source-row-shuffle, atom-shuffle, candidate-roll, and cross-family
  substitution controls with paired uncertainty.

## Primary Sources

### Low-Bit Vector And KV Quantization

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  https://arxiv.org/abs/2504.19874
  - Use: random-rotation plus scalar quantization for packet coefficients; QJL
    residual sign bits as an inspiration for a small bias-correction sideband.
  - Boundary: byte-floor and format inspiration only. Do not claim superiority
    without matched native runs.

- ICLR 2026 TurboQuant poster.
  https://iclr.cc/virtual/2026/poster/10006985
  - Use: venue/current-frontier citation for near-optimal vector quantization.

- Google Research TurboQuant blog.
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
  - Use: official applied framing around KV bottlenecks and vector search.
  - Boundary: do not cite blog speed/quality numbers as local evidence.

- Revisiting RaBitQ and TurboQuant.
  https://arxiv.org/abs/2604.19528
  - Use: reproducibility caution around TurboQuant runtime/recall/KV results.

- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. https://arxiv.org/abs/2406.03482
  - Use: sign-bit JL residual sketches and unbiased inner-product estimation.

- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
  https://arxiv.org/abs/2402.02750
  - Use: practical low-bit KV byte floor and asymmetric key/value format idea.

- KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
  Quantization. https://arxiv.org/abs/2401.18079
  - Use: non-uniform per-layer formats and dense/sparse outlier handling.

- BlockDialect: Block-wise Fine-grained Mixed Format Quantization for
  Energy-Efficient LLM Inference. https://arxiv.org/abs/2501.01144
  - Use: packet "formatbook" design: choose compact formats per block/field
    instead of forcing one scalar quantizer everywhere.

### Sparse / Hardware-Friendly Packet Formats

- NVIDIA Ampere structured sparsity overview.
  https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/
  - Use: hardware-friendly sparsity means regular patterns such as 2:4, with
    index metadata and constrained layout. Unstructured sparse packets are not
    automatically accelerator-friendly.

- SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks.
  https://arxiv.org/abs/1708.04485
  - Use: compressed sparse formats reduce movement/storage only when the
    dataflow can exploit them.

- SIGMA: A Sparse and Irregular GEMM Accelerator with Flexible Interconnects.
  https://doi.org/10.1109/hpca47549.2020.00015
  - Use: irregular sparsity needs hardware/dataflow support; packet sparsity
    alone is not a speed claim.

### Communication, Error-Correction, And Confidence Coding

- Slepian-Wolf: Noiseless coding of correlated information sources.
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - Use: source-private packet framing with receiver side information.

- Wyner-Ziv: Rate-distortion with side information at the decoder.
  https://doi.org/10.1109/TIT.1976.1055508
  - Use: conditional residual packets; encode what the receiver lacks.

- DISCUS: Distributed source coding using syndromes.
  https://doi.org/10.1109/TIT.2002.808103
  - Use: syndrome-style parity sidebands and list decoding against receiver
    side information.

- Error-Correcting Output Codes.
  https://arxiv.org/abs/cs/9501101
  - Use: redundant class/candidate codewords and Hamming-distance decoding
    against target evidence.

- On Calibration of Modern Neural Networks.
  https://proceedings.mlr.press/v70/guo17a.html
  - Use: temperature-scaled source/target confidence bits; margin bins should
    be calibrated on train only.

- Selective Classification for Deep Neural Networks.
  https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
  - Use: risk/coverage style receiver override gates rather than always-on
    packet decoding.

- A Tutorial on Conformal Prediction.
  https://jmlr.org/beta/papers/v9/shafer08a.html
  - Use: confidence sets/list packets with train-calibrated coverage.

### Dense Communication And Native Systems Baselines

- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  https://openreview.net/forum?id=LeatkxrBCi
  - Use: closest dense learned KV-cache communication competitor.

- KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
  https://arxiv.org/abs/2510.03346
  - Use: selective KV-pair sharing baseline and attention-importance control.

- KVCOMM: Online Cross-context KV-cache Communication.
  https://arxiv.org/abs/2510.12872
  - Use: cache-reuse native serving comparator.

- DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM
  Serving. https://arxiv.org/abs/2411.02820
  - Use: same-architecture cache reuse competitor.

- vLLM / PagedAttention. https://arxiv.org/abs/2309.06180
  - Use: native serving substrate and KV memory-management baseline.

- vLLM metrics documentation.
  https://docs.vllm.ai/en/stable/design/metrics/
  - Use: required fields include request success, prompt/generation tokens,
    e2e latency, TPOT, TTFT, running/waiting requests, cache use, and prefix
    cache hits.

- SGLang / RadixAttention. https://arxiv.org/abs/2312.07104
  - Use: second serving substrate for KV reuse and structured decoding.

- SGLang production metrics.
  https://sgl-project.github.io/references/production_metrics.html
  - Use: estimated read/write byte counters are observability signals, not
    hardware counters.

- NVIDIA GenAI-Perf.
  https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2520/user-guide/docs/perf_analyzer/genai-perf/README.html
  - Use: independent endpoint metrics for TTFT, inter-token latency, output
    throughput, and request throughput.

- FlashAttention.
  https://arxiv.org/abs/2205.14135
  - Use: systems claims must be IO-aware: HBM/SRAM reads and writes matter more
    than abstract byte arithmetic.

- NVIDIA Nsight Systems GPU metrics.
  https://docs.nvidia.com/nsight-systems/2021.2/pdf/UserGuide.pdf
  - Use: native HBM/PCIe/NVLink/SM utilization claims require profiler rows.

## Packet Techniques To Try Next

1. Rotated coefficient quantization: random orthogonal/Hadamard rotation before
   scalar quantizing sparse coefficients; compare against existing uniform
   signed coefficients.
2. QJL-style residual sign sketch: add a 1-bit-per-small-sketch residual
   sideband after top-k atom coding; controls must include sign flip, residual
   row shuffle, and zero residual.
3. Confidence-coded packet header: reserve 2-4 bits for calibrated source
   margin / source-target disagreement / expected-value bin; train receiver
   overrides only where calibration predicts positive value.
4. Syndrome/list packet: transmit candidate id plus parity/top2/rival bits and
   decode against target side information; compare to candidate-only,
   same-byte random, candidate-roll, and parity-shuffle.
5. Formatbook sweep: packet fields choose among 1B candidate, 2B score-shape,
   4B PQ/residual, 8B calibrated score, and 12B train-donor formats selected
   only on official-train calibration.
6. Hardware-friendly sparse layout: sorted atom ids, delta-coded indices,
   block-top-k or N:M-compatible atom groups; report decode branches, bytes,
   cache-line/DMA-rounded bytes, and local ring transport.

## Mac-Feasible Ablations

- Re-run current HellaSwag validation `1024:2048` Qwen-to-Phi receiver surface
  with rotated quantizers and QJL residual bits.
- Add confidence header variants to the failed quantized-score and denoising
  syndrome gates; evaluate slice stability and paired CI.
- Implement Hamming/ECOC-style top2 candidate codewords for 4-choice tasks;
  decode by target score plus Hamming-distance penalty.
- Sweep packet formats at fixed framed bytes: 4B, 5B, 7B, 11B, 16B.
- Add top-atom knockout, coefficient shuffle, source-row shuffle, and
  same-byte random for every new packet.
- Extend the Mac packet-ring ledger with cache-line rounded bytes and
  coefficient of variation for the same packet formats.

## No-Claim Boundary

Without native GPU/hardware measurement, do not claim TTFT/TPOT/goodput
improvement, HBM or PCIe/NVLink reduction, energy savings, vLLM/SGLang
integration, C2C/KVComm/DroidSpeak/TurboQuant/KIVI/KVQuant superiority,
structured-sparsity acceleration, or production serving wins. Current Mac
packet-ring evidence supports only byte/accounting and local packed-record
movement.
