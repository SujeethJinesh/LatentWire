# SinkAware Phase 1 Quick-Kill Source Audit

Date: 2026-05-05

Scope: Mac-only primary-source/code audit for whether existing systems already
implement the SinkAware kill-criterion path:

`output += sink_bias_precomputed; for non_sink_tokens: compute_scores(); softmax_with_bias()`

## Decision

Quick-kill criterion: not triggered for fixed-position BOS/sink KV tokens.

However, novelty risk is high. FlashInfer, FlashMLA, and GPT-OSS already expose
learned/per-head attention sink terms that enter the softmax denominator. This
means generic "sink-aware attention kernel" wording is already occupied. The only
remaining wedge is narrower: fixed early-position/BOS KV tokens as a reusable
static or semi-static contribution, with a proof or measurement that avoids
recomputing those token scores without changing output quality.

## Evidence Table

| Source | Primary-source evidence | Sink/static handling | Effect on SinkAware |
| --- | --- | --- | --- |
| FlashInfer / TRT-LLM FMHA | `external/flashinfer/include/flashinfer/trtllm/fmha/kernelParams.h:66-67` calls attention sinks an additional per-head value in the softmax denominator. `external/flashinfer/csrc/fmha_v2/fmha/fragment.h:1021-1033` and `:1205-1217` add `exp(attention_sink_value_ - max[i])` to the running softmax sum. `external/flashinfer/csrc/fmha_v2/fmha/warpspec/epilogue.h:977-1000` adds the sink to `global_sum` before output scaling. `external/flashinfer/tests/test_helpers/sink_attention_reference.py:23-36` appends sink logits then drops the synthetic sink column from output weights. | Learned/per-head synthetic sink mass in denominator; no inspected path computes fixed BOS/sink K/V contribution as a precomputed output term. | Does not kill the exact fixed-token prior idea, but kills broad novelty wording around "attention sinks in kernels." |
| FlashAttention | `external/flash-attention/flash_attn/flash_attn_interface.py:1037-1046` documents causal/window masks for packed attention. `:1099-1122` documents causal and sliding-window masking for KV-packed attention. CK wrappers set `has_sink=false`, `sink_ptr=nullptr`, and `sink_size=0` in `external/flash-attention/csrc/flash_attn_ck/mha_varlen_fwd.cpp:40-48`, `:132-138`, `:228-235`, `:316-318`. | Standard causal/local masks and optional bias paths; CK sink plumbing appears disabled in the inspected wrapper. | Leaves a gap for a fixed-token sink optimization, but FlashAttention alone is not the novelty baseline anymore because downstream kernels added sink terms. |
| StreamingLLM | README describes retaining recent tokens plus attention sinks in `external/streaming-llm/README.md:22`, `:48-52`. `external/streaming-llm/streaming_llm/kv_cache.py:23-34` defines `start_size + recent_size`; `:46-64` and `:66-90` concatenate initial K/V and recent K/V. LLaMA path still computes `query @ key`, adds mask, softmaxes, then multiplies by V in `external/streaming-llm/streaming_llm/pos_shift/modify_llama.py:94-131`. | Keeps initial sink K/V in cache; does not precompute away their score or value contribution. | Supports the motivation, but is an algorithmic/cache baseline rather than a kernel-level kill. |
| FlashMLA / DeepSeek sparse kernels | `external/FlashMLA/flash_mla/flash_mla_interface.py:85-90` documents sparse indices and `attn_sink` as output scaling by `exp(lse)/(exp(lse)+exp(attn_sink))`; `:151-160` passes `attn_sink` to sparse decode; `:176-199` documents sparse prefill with `attn_sink` and `topk_length`. Kernel code at `external/FlashMLA/csrc/sm90/prefill/sparse/phase1.cuh:201-207` scales by `1/(rL + exp2(attn_sink-rM))`; `:467-486` loads selected token indices. | Sparse selected-token attention plus learned denominator sink; not a fixed early-position precomputed K/V prior. | Strong novelty risk; proceed only with a narrower fixed-token decomposition claim. |
| DeepSeek-V3.2-Exp DSA | README frames DeepSeek Sparse Attention as long-context efficiency in `external/DeepSeek-V3.2-Exp/README.md:42-50` and points high-performance sparse kernels to FlashMLA in `:81-83`. `external/DeepSeek-V3.2-Exp/inference/model.py:435-487` builds an FP8 indexer and returns dynamic `topk_indices`. `:574-589` and `:599-605` scatter top-k masks into prefill/decode scores before softmax. | Dynamic top-k sparse mask/indexing. No inspected static BOS/sink special case. | Related-work baseline, not a kill. It raises bar for any efficiency claim. |
| Native Sparse Attention | `external/native-sparse-attention/native_sparse_attention/ops/naive.py:13-24` mean-pools compressed K/V blocks. `:196-208` builds top-k selected block indices from compressed attention. `:158-164` separately computes sliding-window attention. Triton path `external/native-sparse-attention/native_sparse_attention/ops/parallel.py:419-456` selects top-k blocks; `:1390-1429` combines compressed, selected, and sliding-window attention using FlashAttention for local windows. | Compression + selected blocks + local windows. `bos_token_id` exists in config, but no inspected static BOS/sink prior path. | Related-work baseline for sparse selection/compression, not a kill. |
| GPT-OSS reference implementations | `external/gpt-oss/gpt_oss/torch/model.py:153-173` concatenates learned sinks into attention logits and drops the synthetic sink column from output weights. `:187-190` declares per-head `self.sinks`. Triton file states "learned sinks and banded attention" at `external/gpt-oss/gpt_oss/triton/attention.py:1-7`; `:44-55` loads sink and initializes max state; `:94-96` adds sink to the denominator; `:165-203` reference implementation mirrors denominator-only sink handling. | Learned attention sinks plus banded/sliding attention. No fixed early-token K/V contribution path in inspected code. | Most important novelty warning. Any paper-facing claim must avoid sounding like learned sink support or banded attention. |

## Quick COLM_v3 Usefulness

Useful artifact: yes, but only as a systems triage artifact. The audit can support
a sentence that side experiments did not reveal an immediate paper-ready systems
claim and that existing kernels already cover learned sink-denominator handling.

Not useful as current COLM_v3 evidence: no measurements, no local kernel, no GPU
run, and no proof that fixed-token sink work can be skipped safely.

## Recommendation

Pivot/narrow, do not kill outright.

Proceed only to a tiny Phase 2 math/reference gate: implement a CPU-only reference
decomposition that separates fixed sink-token contribution from non-sink tokens
and checks exactness or approximation error on synthetic tensors. If the
contribution remains query-dependent enough that no reuse is possible without
recomputing `QK_sink`, kill the branch. If it is only a denominator-only learned
sink variant, kill because GPT-OSS/FlashInfer/FlashMLA already cover it.
