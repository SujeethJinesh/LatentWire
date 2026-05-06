# HybridKernel Source-Line Audit Table

- date: 2026-05-06
- status: reviewer-risk hardening, not proof of absence
- scope: public docs and local artifacts available on the Mac; no SSH, GPU, or
  large runtime clone.

## Claim Boundary

This table does not prove that no production runtime has an attention-to-SSM
boundary fusion. It records the exact source surfaces checked before native
profiling. The current claim remains narrower:

> no fused attention-output-to-next-SSM compute-boundary kernel was found in
> this audit, while vLLM already covers major hybrid state layout and transfer
> paths.

## Audit Table

| Surface checked | Version / path | What it covers | What it rules out | What it does not rule out |
|---|---|---|---|---|
| IBM Granite model docs | `https://www.ibm.com/granite/docs/models/granite/` and local configs in `phase0/configs/` | Confirms Granite 4.0 H hybrid Mamba2/Transformer layout and 8 immediate layer-type boundaries in Tiny/Small configs. | Does not advertise a boundary-fused serving kernel. | Vendor/runtime private kernels. |
| vLLM hybrid SSM disaggregated serving blog | public vLLM blog, 2026-04-21 | HMA shared tensors, dual descriptor views, DS conv-state layout, 3-descriptor conv transfer, no staging buffers/no reshuffling for hybrid SSM/FA transfer. | Broad claim that hybrid state transfer/layout is unsolved. | Per-layer compute-boundary fusion inside model execution. |
| vLLM RFC #17140 | `https://github.com/vllm-project/vllm/issues/17140` | Tracks native Mamba/SSM/hybrid support gaps and V1 work. | Confirms ecosystem actively targets hybrid serving. | Current post-RFC implementation details not present in issue discussion. |
| vLLM `ssm_conv_transfer_utils` docs | public API docs | Decomposes Mamba2 conv state into x/B/C regions under DS layout for transfer descriptors. | Claim that conv-state transfer layout is the open contribution. | Attention output to next SSM input fusion. |
| vLLM `mamba_mixer2` docs | public API docs | Mamba2 backend, state shape/dtype, CUDA path, and attention backend plumbing. | Claim that Mamba2 internals lack optimized backend support. | Cross-layer attention/SSM boundary fusion. |
| vLLM SSD chunk scan docs | public API docs | Optimized SSM/chunk-scan internals. | Claim that SSM internals are unoptimized. | Boundary-local hidden-state handoff overhead. |
| FlashInfer README | `https://github.com/flashinfer-ai/flashinfer` | Attention/GEMM/MoE/communication/RoPE/norm/activation kernels. | No README-level evidence of a named hybrid attention-to-SSM boundary operator. | Operators deeper in source or downstream integrations. |
| state-spaces/mamba README | `https://github.com/state-spaces/mamba` | Mamba/Mamba-2/Mamba-3 standalone SSM kernels and install path. | Claim that Mamba kernels are absent. | Hybrid serving boundary fusion outside the Mamba repo. |
| Local HybridKernel toy primitive | `phase3/reference/boundary.py`, `phase4/kernel/boundary_triton.py` | A tiny semantic/indexing preflight for future boundary-local math. | Nothing about runtime performance. | Real attention-to-Mamba/Gated-DeltaNet fusion. |

## Reviewer Readout

The audit supports a profiler gate, not a method result. A reviewer can still
object that the source search is incomplete; the paper should therefore avoid
novelty claims stronger than "no evidence found in audited public surfaces."
