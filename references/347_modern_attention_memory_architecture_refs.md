# 347 Modern Attention and Memory Architecture References for LatentWire

Primary-source memo on attention/memory architectures that are the closest semantic neighbors to LatentWire-style cross-model communication. The main lesson is not "use more context"; it is that routing, memory, recurrence, and KV sharing are distinct levers and should be ablated separately.

| Source | Primary links | Core idea | LatentWire ablation | Telemetry fields | Claim risks |
|---|---|---|---|---|---|
| **Mamba** | [paper](https://arxiv.org/abs/2312.00752), [repo](https://github.com/state-spaces/mamba) | Selective SSMs can replace attention in many settings by making state updates input-dependent while keeping inference linear in sequence length. | Replace the bridge attention block with a selective-SSM transport block and compare `attention-only`, `SSM-only`, and `hybrid` transport. | `accuracy`, `throughput_tok_s`, `peak_kv_bytes`, `hidden_state_drift`, `route_entropy` | A win may just reflect lower compute or fewer parameters, not better communication. |
| **Mamba-2** | [paper](https://arxiv.org/abs/2405.21060) | Structured State Space Duality links Transformers and SSMs and yields a faster refinement of the Mamba core layer. | Test whether a duality-style bridge layer improves transport stability relative to a vanilla recurrent/attention bridge. | `accuracy`, `wall_clock_ms`, `state_update_norm`, `latent_reconstruction_loss`, `context_length` | Faster inference is not the same as better cross-model transfer; match budget and length. |
| **Jamba** | [paper](https://arxiv.org/abs/2403.19887) | Interleaving Transformer and Mamba layers, plus MoE, gives a hybrid architecture with strong long-context quality and smaller memory footprint than a vanilla Transformer. | Compare `all-attention`, `all-SSM`, and `interleaved` bridge stacks, with an MoE router only on the bridge path. | `active_parameter_count`, `gate_entropy`, `accuracy`, `throughput_tok_s`, `memory_footprint` | Hybrid gains can come from hidden capacity changes; keep active parameters fixed. |
| **Titans** | [paper](https://arxiv.org/abs/2501.00663) | A neural long-term memory module is trained at test time to store persistent context, while attention acts as short-term memory. | Add a writeable test-time memory bank to the LatentWire path and compare it against a pure sliding-context bridge. | `memory_write_rate`, `memory_read_hit_rate`, `accuracy_by_distance`, `update_norm`, `latency_ms` | Test-time memory can look good only because it expands effective context; control for length and writes. |
| **Memorizing Transformers** | [paper](https://arxiv.org/abs/2203.08913) | Approximate kNN over past key/value pairs lets a model memorize new information at inference time without weight updates. | Compare `no_memory`, `kNN_memory`, and `retrieval_plus_bridge` when the source model sees novel facts or symbols. | `retrieval_precision`, `memory_size`, `novel_fact_recovery`, `accuracy_long_tail`, `kv_reuse_rate` | Retrieval memory can overfit easy lookup tasks and understate reasoning gaps. |
| **Mixture-of-Depths** | [paper](https://arxiv.org/abs/2404.02258) | Tokens are routed top-k into self-attention and MLP computation so FLOPs are allocated non-uniformly across positions and depth. | Replace uniform bridge compute with top-k token routing and compare to a fixed-depth bridge at the same FLOP budget. | `active_token_fraction`, `route_entropy`, `flops_per_token`, `accuracy`, `latency_ms` | A routing win may just be a compute win unless FLOPs are explicitly matched. |
| **Mixture-of-Recursions** | [paper](https://arxiv.org/abs/2507.10524) | Shared layers are reused across recursion steps, and token-specific recursion depth plus selective KV caching give both parameter efficiency and adaptive computation. | Test recursive bridge depth `1 / 2 / 4` and a KV-sharing variant that reuses early recursion state. | `recursion_depth_histogram`, `kv_share_ratio`, `accuracy`, `throughput_tok_s`, `memory_access_count` | Recurrence can hide extra compute in the loop count; report budget-normalized quality. |
| **StreamingLLM / attention sinks** | [paper](https://arxiv.org/abs/2309.17453), [code](https://github.com/mit-han-lab/streaming-llm) | Keeping a few sink tokens plus the recent window largely restores performance when KV cache is truncated. | Compare `sink_tokens + recent_window` against `recent_window_only` and `full_cache` in the bridge. | `sink_mass`, `window_size`, `perplexity_drift`, `accuracy_over_length`, `kv_bytes` | Sink behavior can be model-specific and may fail outside chat-style or pretrained settings. |
| **PagedAttention / vLLM** | [paper](https://arxiv.org/abs/2309.06180) | KV cache should be managed like virtual memory: paged allocation reduces fragmentation and enables flexible KV sharing within and across requests. | Benchmark a paged bridge payload against a contiguous-cache payload and a page-sharing payload. | `fragmentation_ratio`, `page_count`, `peak_kv_bytes`, `throughput_tok_s`, `latency_p95_ms` | Serving-system gains do not imply an architecture gain unless the model path is the same. |
| **GQA** | [paper](https://arxiv.org/abs/2305.13245) | Grouped-query attention generalizes multi-query attention by using fewer KV heads than query heads, preserving quality better than plain MQA. | Sweep `MHA`, `GQA`, and `MQA` on the bridge to see whether fewer KV heads are enough for cross-model transfer. | `kv_head_count`, `accuracy`, `decode_latency_ms`, `memory_bandwidth`, `quality_delta` | KV-head reductions can hide quality loss that only appears on long-context or reasoning-heavy prompts. |
| **Multi-Query Attention** | [paper](https://arxiv.org/abs/1911.02150) | Sharing keys and values across all heads reduces memory bandwidth and speeds incremental decoding with only minor quality loss. | Use MQA as the aggressive low-KV baseline for the bridge before trying more elaborate sharing schemes. | `kv_head_count`, `bandwidth_gbps`, `accuracy`, `decode_latency_ms`, `memory_savings` | MQA is a strong efficiency baseline but may undercut fidelity on heterogeneous tasks. |
| **HShare** | [forum](https://openreview.net/forum?id=Tb5PY5vwp6), [paper pdf](https://openreview.net/pdf?id=Tb5PY5vwp6) | Critical KV indices are similar across neighboring queries, layers, and heads, so they can be shared hierarchically instead of reselected each step. | Compare `query-specific selection`, `fixed sharing`, and `hierarchical sharing` for bridge-side KV reuse. | `sharing_ratio`, `self_attention_speedup`, `end_to_end_throughput`, `accuracy`, `critical_token_overlap` | Sharing can look better only because the selected prompt family has high token stability; test heterogeneous pairs. |

## Highest-priority LatentWire ablations

1. **Attention vs selective-SSM transport**
   - Compare `attention-only`, `Mamba-style selective SSM`, and `hybrid` bridge blocks under the same byte and compute budget.

2. **Test-time memory vs sliding cache**
   - Compare a writable memory bank against a sink-plus-window bridge and a plain truncation baseline.

3. **Retrieval memory vs direct cache transfer**
   - Compare `kNN memory`, `KV sharing`, and `direct cache fusion` on novelty-heavy prompts.

4. **Adaptive compute vs uniform depth**
   - Compare `Mixture-of-Depths` and `Mixture-of-Recursions` routing against a fixed-depth bridge with the same FLOPs.

5. **KV sharing granularity**
   - Sweep `MQA`, `GQA`, and hierarchical sharing to find the smallest KV footprint that still preserves transfer quality.

## Telemetry To Standardize

- `accuracy`
- `throughput_tok_s`
- `latency_ms`
- `peak_kv_bytes`
- `fragmentation_ratio`
- `memory_bandwidth`
- `route_entropy`
- `gate_entropy`
- `kv_head_count`
- `critical_token_overlap`
- `reconstruction_loss`
- `accuracy_over_length`

## Interpretation Guardrails

- Match byte budget before comparing bridge variants.
- Match wall-clock or FLOPs before comparing recurrence or routing.
- Report the effective context length whenever memory is added.
- Do not treat serving-system wins as communication wins unless the model path is unchanged.
- Treat any gain from sharing or routing as provisional until it holds on both long-context and reasoning-heavy prompts.
