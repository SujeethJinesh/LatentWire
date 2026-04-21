# Architecture-Level Lateral Inspiration for LatentWire

This memo collects recent architecture papers that do not solve cross-model KV
communication directly, but do isolate the same design axes LatentWire keeps
running into: dynamic token budgeting, recursive reuse, persistent memory,
hybrid attention/SSM schedules, and residual correction.

| Source | Primary link | Relevance to LatentWire | Concrete LatentWire ablation |
|---|---|---|---|
| **Mixture-of-Depths** (Raposo et al., 2024) | [arXiv](https://arxiv.org/abs/2404.02258) | Top-k routing under a fixed FLOPs budget; token identities can change by layer instead of staying tied to static positions. | Replace static grouped transport with token-importance routing at fixed bytes/latency; compare norm-based, attention-based, and learned routers while holding the transport budget constant. |
| **Attention Is All You Need for Mixture-of-Depths Routing** (2024) | [OpenReview](https://openreview.net/forum?id=1uDP4ld3eZ) | Uses the previous layer's attention map to drive routing, which is a cheap and directly testable selector for what to compute or transport. | Swap the current selector for previous-layer attention-derived routing; test whether attention alone can choose which source tokens deserve K/V export. |
| **Mixture-of-Depths Attention** (2026) | [arXiv](https://arxiv.org/abs/2603.15619) | Lets heads attend to both current-layer KVs and depth-KVs from preceding layers, which is very close to "communicate with prior latent states" as a primitive. | Add depth-KV memory to LatentWire bridges: current-only vs depth-only vs current+depth; measure whether depth-KVs repair mismatch better than wider current-layer transport. |
| **Mixture-of-Recursions** (2025) | [arXiv](https://arxiv.org/abs/2507.10524) | Reuses a shared stack across recursion steps, caches only active tokens' KVs, and includes a KV-sharing variant. | Run 1/2/3 recursion passes through the same bridge; cache only active tokens between recursions; compare shared-first-recursion KV reuse against full recomputation. |
| **Retentive Network (RetNet)** (2023) | [arXiv](https://arxiv.org/abs/2307.08621) | Shows a clean split between parallel training and recurrent/chunkwise inference with a fixed-size persistent state. | Replace part of the bridge with a recurrent summary state; compare fixed-size memory against explicit token-wise KV carry on long-context and repeated-query slices. |
| **Titans: Learning to Memorize at Test Time** (2025) | [arXiv](https://arxiv.org/abs/2501.00663) | Introduces a separate long-term memory module that helps short-term attention, which maps well to a bridge-plus-memory design. | Add a tiny persistent memory sidecar for cross-model communication; compare reset-every-example vs carry-across-chunks and inspect contamination from stale memory. |
| **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (2023) | [arXiv](https://arxiv.org/abs/2312.00752) | The base SSM result: input-dependent gating plus a recurrent state can replace quadratic attention in many settings. | Compress source KVs into an SSM state before transport; compare all-attention, all-SSM, and hybrid source-compression variants at equal wall-clock cost. |
| **Jamba: A Hybrid Transformer-Mamba Language Model** (2024) | [arXiv](https://arxiv.org/abs/2403.19887) | Interleaves Transformer and Mamba blocks and shows that hybrid schedules can preserve quality while cutting memory footprint. | Sweep attention:Mamba ratios inside the bridge; test early-attention, late-attention, and alternating schedules instead of one monolithic translator. |
| **Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models** (2025) | [arXiv](https://arxiv.org/abs/2504.03624) | A stronger recent hybrid: most self-attention layers are replaced with Mamba layers while keeping accuracy competitive and inference faster. | Use Nemotron-style "mostly-SSM, some attention" placement as a bridge control; measure whether a small late attention repair block recovers accuracy better than a pure KV bridge. |
| **Gated Delta Networks: Improving Mamba2 with Delta Rule** (2024) | [arXiv](https://arxiv.org/abs/2412.06464) | Shows gating and delta updates are complementary, and that hybridizing with sliding-window attention or Mamba2 layers can improve retrieval and long-context behavior. | Transport deltas instead of absolute latent states; compare gate-erase-update bridges against direct state transfer, especially on examples where the source/target mismatch is mostly residual. |

## Five LatentWire Ablations To Run First

1. Static position groups vs token-importance routing at a fixed byte budget, using MoD/A-MoD/MoDA as the selector family.
2. Single-pass bridge vs recursive bridge reuse, using MoR to test whether multiple cheap refinement passes beat one expensive transport.
3. Pure KV transport vs transport plus persistent memory sidecar, using RetNet and Titans as the memory baseline.
4. All-attention translator vs mostly-SSM translator with late attention repair, using Mamba, Jamba, and Nemotron-H as hybrid schedules.
5. Absolute-state transport vs delta transport with erase/update gating, using Gated DeltaNet as the correction baseline.
