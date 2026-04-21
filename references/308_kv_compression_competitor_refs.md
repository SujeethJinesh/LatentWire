# KV Compression Competitors for LatentWire

Primary-source memo for baselines and benchmark inspiration. This is ordered by how useful each method is as a comparator for LatentWire, not by citation count.

## Short read

- The closest apples-to-apples baseline for **cross-model communication** is **C2C** and, secondarily, **KVComm**.
- The rest are mostly **KV-cache compression / selection / quantization** baselines. They are useful as efficiency comparators and failure-mode probes, but they do not directly test semantic transfer between two different models.
- For LatentWire, the important thing is to report both **quality** and **communication cost**: retained KV fraction, bytes moved, cache bytes per answer, latency, and layer/head retention traces.

## Comparator matrix

| Method | Primary source | What it measures / optimizes | Apples-to-apples for cross-model communication? | Runnable repo under `references/repos` | LatentWire comparator / telemetry action |
|---|---|---|---|---|---|
| **C2C / Cache-to-Cache** | [Paper](https://arxiv.org/abs/2510.03215), [code](https://github.com/thu-nics/C2C) | Neural projection + gating to fuse source and target KV caches for direct semantic transfer | **Yes, closest match** | `references/repos/C2C` | Use as the main external semantic-communication baseline; log paired accuracy, cache transfer bytes, projection FLOPs, and source/target asymmetry. |
| **KVComm** | [Paper](https://arxiv.org/abs/2510.12872), [code](https://github.com/Zephyroam/KVComm) | Selective KV sharing across contexts / agents for online communication | **Mostly yes** | `references/repos/KVComm` | Treat as the main multi-agent cross-context baseline; compare with the same source-target pair and record token-level communication budget. |
| **Quest** | [Paper](https://arxiv.org/abs/2406.10774), [code](https://github.com/mit-han-lab/quest) | Query-aware page sparsity: which KV pages to load at decode time | No, but strong query-aware selection baseline | `references/repos/Quest` | Use for query-aware selection comparisons; log page budget, page hit rate, and whether LatentWire’s runtime head routing matches or beats Quest-like selection under equal bytes. |
| **KVzip** | [Paper](https://arxiv.org/abs/2505.23416), [code](https://github.com/snu-mllab/KVzip) | Query-agnostic compression with context reconstruction; preserves future-query utility | No direct semantic comm, but strong compression comparator | `references/repos/KVzip` | Compare at matched compression ratio and matched prompt budget; log reconstruction loss proxy, retained-KV fraction, and whether LatentWire remains stable under query shift. |
| **H2O** | [Paper](https://arxiv.org/abs/2306.14048), [code](https://github.com/FMInference/H2O) | Heavy-hitter + recent-token cache eviction; measures token persistence/importance | No | `references/repos/kvpress`, `references/repos/Quest`, `references/repos/r-kv` | Baseline for heavy-hitter assumptions; log heavy-hitter overlap, sink/recent retention, and whether LatentWire’s routing concentrates on the same early tokens. |
| **StreamingLLM** | [Paper](https://arxiv.org/abs/2309.17453), [code](https://github.com/mit-han-lab/streaming-llm) | Sink-token + recent-window retention for infinite streaming | No | `references/repos/kvpress`, `references/repos/Quest`, `references/repos/r-kv`, `references/repos/DeltaKV_sparse_vllm` | Use as a fixed-window control; log sink-token dependence, recency bias, and failure under long-range retrieval. |
| **SnapKV** | [Paper](https://arxiv.org/abs/2404.14469), [code](https://github.com/FasterDecoding/SnapKV) | Prefill attention-based token selection; “what you are looking for” | No | `references/repos/kvpress`, `references/repos/KVzip`, `references/repos/r-kv`, `references/repos/DeltaKV_sparse_vllm` | Compare query-attention selection against LatentWire routing; log attention entropy, top-k mass, and retained-token overlap with route atoms. |
| **PyramidKV** | [Paper](https://arxiv.org/abs/2406.02069), [code](https://github.com/Zefan-Cai/PyramidKV) | Layer-wise pyramidal cache funneling; measures layer importance | No | `references/repos/kvpress`, `references/repos/KVzip`, `references/repos/DeltaKV_sparse_vllm` | Use for layer-budget schedules; compare against LatentWire’s runtime head selection ratio and log per-layer retention curves. |
| **AdaKV** | [Paper](https://arxiv.org/abs/2407.11550), [implementation in `kvpress`](https://github.com/NVIDIA/kvpress) | Head-wise compression wrapper for any scorer-based policy | No | `references/repos/kvpress`, `references/repos/KVzip` | Useful as a head-budget wrapper; log whether LatentWire’s head routing is just an AdaKV-style redistribution or something more structural. |
| **Scissorhands** | [Paper](https://arxiv.org/abs/2305.17118), [code](https://github.com/JingWu321/Scissorhands) | Persistence-of-importance hypothesis for pruning | No | No standalone local repo found | Use as a theory check for token persistence; log stability of token importance across decode steps and whether the same tokens remain selected. |
| **Keyformer** | [Paper](https://arxiv.org/abs/2403.09054), [code](https://github.com/d-matrix-ai/keyformer-llm) | Key-token selection to reduce cache size | No | No standalone local repo found | Benchmark selection sharpness; log selected-token sparsity, score skew, and whether LatentWire can match the same budget with better transfer fidelity. |
| **rKV / R-KV** | [Paper](https://arxiv.org/abs/2505.24133), [code](https://github.com/zefan-cai/r-kv) | Redundancy-aware decode-time compression for reasoning traces | No, but relevant for reasoning-model decode budgets | `references/repos/r-kv` | Compare on long CoT / reasoning traces; log redundancy ratio, reasoning-step retention, and whether LatentWire’s communication method keeps less redundant state. |
| **DeltaKV** | [Paper](https://arxiv.org/abs/2602.08005), [code/runtime](https://github.com/CURRENTF/Sparse-vLLM) | Residual-based hybrid compression for long-context inference | No | `references/repos/DeltaKV_sparse_vllm` | Use as the main hybrid compression/runtime baseline; log dense-vs-compressed residual split, cache layout, and throughput/quality trade-offs. |

## Local repo inventory that is already runnable here

- `references/repos/C2C`
- `references/repos/KVComm`
- `references/repos/KVzip`
- `references/repos/Quest`
- `references/repos/kvpress`
- `references/repos/r-kv`
- `references/repos/DeltaKV_sparse_vllm`

## Concrete LatentWire actions

1. Add a benchmark table with three groups: direct semantic communication (`C2C`, `KVComm`), query-aware compression (`Quest`, `KVzip`, `SnapKV`, `H2O`, `StreamingLLM`, `PyramidKV`, `AdaKV`), and reasoning/decode compression (`rKV`, `DeltaKV`).
2. Report paired quality/cost curves, not a single operating point: exact-match or task accuracy, bytes moved, retained-KV fraction, and end-to-end latency.
3. Log interpretability telemetry for every run: per-layer retention, head-selection entropy, top-1 mass, source/target asymmetry, and route overlap with runtime heads.
4. For cross-model claims, run the same source-target pair and prompt budget across `target_alone`, `rotalign`, `route_atom`, and the best semantic-communication baseline (`C2C` / `KVComm`).
5. Treat `kvpress`, `Quest`, `KVzip`, and `DeltaKV_sparse_vllm` as reusable harnesses for quick sanity checks before launching full LatentWire runs.
