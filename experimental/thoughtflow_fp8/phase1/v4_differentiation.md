# DeepSeek V4 Differentiation

Date: 2026-05-05

Status: **HISTORICAL PRE-FALSIFICATION SCOPING**. This document records the
original positive-method target before the current sparse-cache signal family
was stopped. It is superseded by
`../phase2/current_decision_manifest_20260506.md` and
`../paper/reviewer_pack.md`; the current camera-ready contribution is the
falsification ladder, not a retrofit FP8/phase-retention method.

## What DeepSeek V4 Does

The April 25, 2026 LMSYS/SGLang post describes DeepSeek V4 as a trained hybrid
sparse-attention model with:

- 1.6T Pro and 284B Flash variants;
- hybrid sparse attention where each layer mixes sliding-window attention over
  the last 128 raw tokens with either:
  - C4: top-k attention over 4:1 compressed KV, or
  - C128: dense attention over 128:1 compressed KV;
- separate SWA, C4, C128 KV pools plus compression-state pools;
- ShadowRadix prefix caching to maintain coherent virtual-to-physical mappings
  across those pools;
- Flash Compressor, an IO-aware fused compressor for compressed attention;
- Lightning TopK for the C4 indexer over very large candidate sets;
- FlashMLA integration for hybrid attention;
- training backend changes for compressed attention, indexer replay, context
  parallelism, FP8 rollout/training, and numerical stability controls.

## Why V4 Is Not A Direct Retrofit Baseline

DeepSeek V4's compression is architectural. It assumes the model was built and
trained around compressed attention, indexers, compression-state pools, and
hybrid attention metadata. The SGLang post explicitly describes training backend
changes for the new compressed-attention module and indexer paths.

ThoughtFlow-FP8 should not claim to be "better DeepSeek V4." The correct
distinction is:

> V4 is a production architecture for newly trained models. ThoughtFlow-FP8 is a
> retrofit compression policy for existing reasoning models whose attention
> architecture and weights are fixed.

## Historical Target Only: What ThoughtFlow Would Have Needed To Claim

This was the target before the branch was falsified. It is not a current claim:

1. Retrofit: no model pretraining or architecture-specific indexer training.
2. Bias-controlled retention: explicit anchor/fair-span/phase-transition keep
   rules derived from Pitfalls and LongFlow review failures.
3. FP8 byte-budgeting: use FP8 to keep more protected context under the same
   memory budget, rather than as a standalone quantization claim.
4. Model portability: target GPT-OSS/Qwen/Apriel/Nemotron-like existing models
   rather than only V4-family compressed-attention models.

## What ThoughtFlow Cannot Claim Yet

- Production systems superiority over DeepSeek V4/SGLang.
- Better 1M-token serving without a real GPU end-to-end run.
- Novelty from compressed attention alone.
- Novelty from fusing compression kernels alone.

## Superseded Recommended Framing

This historical framing is superseded by the current falsification-methodology
framing. The current paper should say:

> ThoughtFlow-FP8 is a preregistered sparse-cache falsification ladder: it
> shows how attractive retrofit utility signals can pass one frozen surface,
> fail stricter reproduction, and be stopped before GPU/kernel work.

## Proceed Gate Against V4

Proceed only if Phase 2/3 can define a portable policy that:

- does not require training compressed-attention indexers;
- can be implemented as a KV-cache manager around standard attention;
- has telemetry V4 papers/blogs do not address for retrofits: span keep-rate,
  phase-transition keep-rate, recurrence misses, and FP8 numerical drift.
