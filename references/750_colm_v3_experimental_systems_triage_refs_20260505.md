# COLM_v3 Experimental Systems Triage References

- created: 2026-05-05
- purpose: primary-source anchors for the three side experiments in `experimental/`
  and for COLM_v3 systems claim boundaries.

## HybridKernel

| Source | Link | Relevance |
|---|---|---|
| IBM Granite 4.0 announcement | https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models | Granite 4.0-H uses a hybrid Mamba-2/Transformer architecture with a 9:1 Mamba-to-attention ratio; useful target family for transition-overhead mapping. |
| Nemotron-H project page | https://research.nvidia.com/labs/adlr/nemotronh/ | Nemotron-H is a hybrid Mamba-Transformer family; useful target and competitor context for hybrid serving. |
| Nemotron-H paper | https://arxiv.org/abs/2504.03624 | Primary paper for hybrid Mamba-Transformer architecture and efficiency claims. |
| vLLM hybrid SSM disaggregated serving blog | https://vllm.ai/blog/hybrid-ssm-disagg | Strong systems baseline/risk: vLLM has recent hybrid SSM/FA state-transfer work. Phase 1 must distinguish boundary-kernel fusion from state-transfer layout work. |

## SinkAware

| Source | Link | Relevance |
|---|---|---|
| StreamingLLM | https://arxiv.org/abs/2309.17453 | Primary source for attention sinks; establishes that retaining initial-token KV helps streaming. SinkAware must show a kernel-level static-prior path, not merely sink retention. |
| FlashAttention-3 PyTorch blog | https://docs.pytorch.org/blog/flashattention-3/ | Hardware-aware attention baseline. Phase 1 should inspect whether FA3 or downstream kernels special-case fixed sink tokens. |
| IndexCache / DSA discussion | https://arxiv.org/abs/2603.12201 | Recent sparse-attention context describing DeepSeek Sparse Attention-style lightning indexers; useful for distinguishing static sinks from dynamic top-k indexers. |

## ThoughtFlow-FP8

| Source | Link | Relevance |
|---|---|---|
| LongFlow OpenReview | https://openreview.net/forum?id=rz6WybXjgk | Direct reasoning-KV compression competitor; fused attention/importance/eviction makes ThoughtFlow novelty difficult unless Phase 1 finds a concrete failure mode. |
| LongFlow arXiv | https://arxiv.org/abs/2603.11504 | Paper source for LongFlow method and reported throughput/KV compression claims. |
| PM-KVQ OpenReview | https://openreview.net/forum?id=Vem6FQvRvq | Long-CoT KV quantization competitor; directly pressures ThoughtFlow's FP8/mixed-precision angle. |
| PM-KVQ arXiv | https://arxiv.org/abs/2505.18610 | Public paper source for progressive mixed-precision KV cache quantization. |
| KIVI OpenReview | https://openreview.net/forum?id=L057s2Rq8O | Established 2-bit KV quantization baseline; useful for the COLM_v3 systems boundary and ThoughtFlow comparisons. |

## Readout For COLM_v3

These side experiments should not enter the COLM_v3 claim set until they produce
measured artifacts. For the current workshop paper, they are best treated as
systems follow-up lanes and as reminders that any hard GPU latency/HBM/energy
claim needs native NVIDIA evidence.
