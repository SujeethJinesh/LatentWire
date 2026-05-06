# HybridKernel Quick Phase 1 Audit

- date: 2026-05-05
- status: quick_audit_partial_proceed_to_architecture_map
- scope: primary-source web/docs/code audit without SSH, GPU, global installs,
  large clones, or model-weight downloads.

## Question

Does current public work already implement a fused compute kernel across an
attention/SSM layer-type boundary, such as attention output directly feeding
Mamba/SSM input in one fused kernel?

## Short Answer

I did not find evidence of an existing fused attention-to-SSM layer-boundary
compute kernel in the quick audit. However, vLLM has very recent, sophisticated
hybrid SSM serving work that substantially narrows the novelty. The project
should proceed only to a Granite-focused Phase 2 architecture map, with the
claim reframed from "hybrid support gap" to the narrower "per-layer compute
boundary fusion gap."

## Source-Backed Notes

| Source | Evidence | Boundary interpretation |
|---|---|---|
| IBM Granite 4.0 docs and release notes | Granite 4.0 uses a hybrid Mamba-2/Transformer architecture; IBM describes a sequential 9:1 Mamba-2/Transformer pattern and ecosystem support in vLLM, llama.cpp, NexaML, and MLX. Sources: <https://www.ibm.com/granite/docs/models/granite/> and <https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models>. | Confirms target relevance. Does not describe boundary-fused kernels; model support is broader than the proposed compute fusion. |
| Local Granite configs | `phase0/configs/ibm-granite-4.0-h-tiny.config.json` and `phase0/configs/ibm-granite-4.0-h-small.config.json` expose `layer_types` with four attention layers among 40 layers. | Gives a concrete Mac-local Phase 2 surface: 8 Mamba/attention boundaries per forward pass. |
| vLLM hybrid SSM disaggregated serving blog | vLLM describes HMA shared-tensor layout, dual descriptor views, DS conv layout, no intermediate buffers/no reshuffling for P/D transfer, and a Nemotron example with 52 layers alternating Mamba and FA. Source: <https://vllm.ai/blog/hybrid-ssm-disagg>. | Strong novelty pressure. vLLM optimizes heterogeneous state transfer/layout, not obviously a single layer-boundary compute kernel. |
| vLLM RFC #17140 | The RFC says SSM state management was less mature, incompatible with prefix caching/offloading/disaggregated serving at that time, and tracks V1 native Mamba/SSM/hybrid support. Source: <https://github.com/vllm-project/vllm/issues/17140>. | Confirms the ecosystem was actively closing hybrid-support gaps; any claim must be current and specific. |
| vLLM `ssm_conv_transfer_utils` docs | Source docs show Mamba2 conv/SSM state decomposition for transfer descriptors and explicitly require DS conv state layout for 3-read transfer. Source: <https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils/>. | This is transfer/layout plumbing, not the proposed attention-output-to-SSM-input fused compute boundary. It is still the highest-risk related implementation. |
| vLLM `mamba_mixer2` docs | Source docs describe `MambaMixer2`, `forward_cuda`, `get_attn_backend`, state shape/dtype, and Mamba2 attention backend plumbing. Source: <https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/mamba_mixer2/>. | Indicates Mamba2 has its own backend and kernels. Quick audit did not find attention-layer output fused with the next SSM layer. |
| vLLM SSD chunk scan docs | Source docs expose Triton-style SSD chunk scan operations. Source: <https://vllm.website.cncfstack.com/api/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan/>. | Confirms optimized SSM internals exist; does not establish cross-layer attention/SSM fusion. |
| FlashInfer README | FlashInfer provides attention, GEMM, MoE, communication, RoPE, normalization, and activation kernels, and powers vLLM/SGLang/TensorRT-LLM/TGI. Source: <https://github.com/flashinfer-ai/flashinfer>. | Strong serving-kernel baseline. The README does not list Mamba/SSM or attention-to-SSM boundary fusion as a core operator; any deeper source audit should wait until native profiling shows a real boundary signal. |
| state-spaces/mamba README | Mamba repo lists Mamba, Mamba-2, and Mamba-3 blocks, including CUDA/NVIDIA requirements and source files such as `mamba3.py`; Mamba-3 is installed from source. Source: <https://github.com/state-spaces/mamba>. | Confirms Mamba-3 kernels are a standalone SSM stack. Quick audit did not find hybrid attention-boundary fusion there. |
| Nemotron-H paper | Nemotron-H replaces most transformer attention layers with Mamba layers and reports inference speed improvements. Source: <https://arxiv.org/abs/2504.03624>. | Confirms hybrid-model motivation. Does not itself claim a boundary-fused compute kernel. |
| Bamba documentation/blog | Bamba is a hybrid Mamba2 model with vLLM support and reported inference speedups. Sources: <https://huggingface.co/docs/transformers/v5.0.0rc2/en/model_doc/bamba> and <https://huggingface.co/blog/bamba>. | Relevant competitor/model family, but quick audit found no boundary-fusion claim. |
| Qwen3-Next NVIDIA/model-card sources | Qwen3-Next is a hybrid Transformer-Mamba/Gated DeltaNet-style MoE model with long-context support and vLLM/NIM/SGLang deployment. Sources: <https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/> and <https://build.nvidia.com/qwen/qwen3-next-80b-a3b-thinking/modelcard>. | Useful as a future config-only target, but its GDN/linear-attention boundary differs from Granite Mamba2. |

## Boundary Fusion Table

| Work/system | Hybrid attention/SSM? | Optimizes SSM internals? | Optimizes hybrid state layout/transfer? | Found fused attention->SSM layer-boundary compute kernel? |
|---|---:|---:|---:|---:|
| Granite 4.0 H | yes | model-dependent | ecosystem support noted | no evidence in quick audit |
| Nemotron-H / Nemotron 3 hybrid serving | yes | yes, via vLLM/Mamba kernels | yes, vLLM HMA/NIXL path | no evidence in quick audit |
| vLLM V1 hybrid support | yes | yes | yes | no evidence in quick audit |
| FlashInfer | supports serving kernels broadly | not found in README | communication kernels exist | no evidence in quick audit |
| state-spaces/mamba / Mamba-3 | SSM stack; not hybrid serving | yes | no | no evidence in quick audit |
| Bamba | yes | via model/runtime support | vLLM support | no evidence in quick audit |
| Qwen3-Next | yes, but GDN-style | likely runtime-specific | vLLM/NIM/SGLang support | no evidence in quick audit |

## COLM_v3 Usefulness

No COLM_v3-useful systems artifact emerged yet. The most useful artifact is a
negative/triage artifact: vLLM already covers a large part of the hybrid serving
systems story through HMA state layout and disaggregated transfer, so any future
HybridKernel claim must be much narrower and must show measured per-layer
compute-boundary overhead.

## Recommendation

Phase 2 has since completed and this recommendation is now a historical gate.
The current project framing is:

> Does a per-layer attention/SSM compute-boundary fusion save measurable memory
> traffic or launch overhead after vLLM's hybrid state-layout work is already in
> place?

Kill if native NVIDIA/vLLM profiling fails to show a separable boundary-local
signal. Deeper vLLM/FlashInfer source audit is deferred until native profiling
shows a real boundary signal worth implementing.

## Current Gate

`phase2/architecture_map.md` has already been generated from the fetched
Granite and Qwen3-Next configs. The remaining gate is native NVIDIA/vLLM
profiling with server-side Nsight traces:

- pass `phase2/check_profiler_run_artifacts.py`;
- reduce the exact packet with `phase2/analyze_profiler_metrics.py`;
- promote only if repeated same-model/config rows clear the 3% recoverable-gain
  gate.
