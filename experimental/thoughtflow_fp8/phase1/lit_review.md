# ThoughtFlow-FP8 Phase 1 Literature Readout

Date: 2026-05-05

## Question

Can ThoughtFlow-FP8 claim a sharp COLM_v3-useful wedge over current reasoning KV
compression methods by combining FP8 KV quantization, sink/anchor protection,
and reasoning-phase-aware eviction for existing models without retraining?

## Primary Findings

### LongFlow

LongFlow targets long-output reasoning generation. Its core claim is an
importance metric derived from intermediate attention computation using only the
current query, plus a Triton kernel that fuses FlashAttention, importance
estimation, and token eviction. The paper reports up to 11.8x throughput and 80%
KV compression with minimal accuracy loss.

OpenReview reviews materially weaken the paper's acceptance story:

- Reviewers saw LongFlow as a speed/accuracy trade-off rather than a clear
  Pareto improvement over R-KV.
- They asked for end-to-end inference speedups in production frameworks such as
  vLLM or SGLang.
- They flagged missing Pareto sweeps over compression/reasoning budget.
- They challenged the theoretical simplification from all future generations to
  the immediate next step.
- They flagged approximation risks: current-query proxy, denominator
  simplification, omitted running max for qk scores, and possible numerical
  instability from outlier key channels.
- They noted efficiency was evaluated too narrowly, especially on Qwen3-1.7B
  with fixed lengths/budgets.

Implication: LongFlow is vulnerable on quality, systems realism, and numerical
stability. A ThoughtFlow paper must not merely add another fused kernel. It must
show reviewer-targeted fixes on those axes.

### Pitfalls of KV Cache Compression

Pitfalls argues that "negligible loss" claims hide realistic failure modes in
multi-instruction prompts. The key finding is eviction bias: compression
policies disproportionately evict entries from some instructions, causing those
instructions to be ignored. The paper uses system prompt leakage and directive
following as concrete examples. The authors propose whitelisting/fair eviction
and discuss automated span identification.

Implication: ThoughtFlow's anchor protection should be framed as a general
structured-retention policy, not just "keep the first few sink tokens." The
right Phase 1 wedge is: LongFlow's single-step current-query importance is
likely biased against low-current-attention but globally important anchors,
instructions, phase transitions, and safety/problem-statement spans.

### DeepSeek V4

The April 25, 2026 LMSYS/SGLang post says DeepSeek-V4 has hybrid sparse
attention: each layer mixes sliding-window attention over the last 128 raw tokens
with either C4 top-k over 4:1 compressed KV or C128 dense attention over 128:1
compressed KV. The stack requires coherent management of SWA/C4/C128 pools and
compression-state pools. It includes ShadowRadix prefix caching, Flash
Compressor, Lightning TopK, FlashMLA hybrid attention integration, and training
backend changes for compressed attention, indexer replay, FP8 rollout/training,
and numerical stability.

Implication: DeepSeek V4 raises the novelty bar for "compressed attention as a
system." The remaining wedge is retrofit: ThoughtFlow-FP8 should target existing
open-weight reasoning models that were not pretrained with V4-style compressed
attention/indexers.

### ThinKV

ThinKV is a close competitor. It is thought-adaptive, combines quantization and
eviction, assigns precision by thought importance, progressively evicts less
critical thoughts, and extends PagedAttention to reuse evicted slots. It reports
near-lossless accuracy with less than 5% original KV cache and up to 5.8x
throughput over baselines.

Implication: ThoughtFlow cannot claim phase awareness broadly unless it
distinguishes from ThinKV. The plausible distinction is stronger anchor/fairness
protection plus FP8/system-kernel design, but ThinKV already occupies much of
the "thought adaptive" space.

### R-KV and R-KVHash

R-KV targets redundant reasoning tokens and reports near-full or better-than-full
accuracy with 10-16% KV, plus memory and throughput gains. R-KVHash observes
that R-KV's pairwise cosine/Gram computation is expensive and replaces it with
SimHash/LSH-style redundancy estimation, avoiding attention-based importance and
reporting up to 2x higher decoding throughput than R-KV.

Implication: R-KV is the quality bar reviewers used against LongFlow. ThoughtFlow
must beat or complement R-KV-style redundancy retention, not only LongFlow.

### RaaS

RaaS identifies milestone tokens and phoenix tokens in reasoning decode. It keeps
prefill tokens to protect phoenix tokens and uses LRU-like retention of milestone
tokens until they become irrelevant, targeting O(L) time and O(L) memory with
accuracy comparable to Quest.

Implication: RaaS already covers reasoning-aware token lifecycle. ThoughtFlow's
phase-transition anchors should acknowledge milestone/phoenix behavior and
explicitly test whether phase markers protect recurring critical tokens better
than LRU/milestone policies.

### LazyEviction

LazyEviction identifies token-importance recurrence: tokens can regain high
attention after multiple decode steps. It uses an observation-window lagged
eviction mechanism to retain latent recurring tokens and reports 50-70% KV
reduction with comparable accuracy.

Implication: naive phase-based eviction can fail if a planning token becomes
important again during execution. Any Phase 2 analysis must measure recurrence,
not just early/late phase labels.

### ForesightKV

ForesightKV is training-based. It constructs Golden Eviction using future
attention scores, distills pairwise rankings, and applies GRPO to mitigate
losses on low-entropy tokens. It reports outperformance under half cache budget
on AIME2024/AIME2025.

Implication: ForesightKV owns the "learn future contribution" axis. ThoughtFlow's
retrofit story should stay training-free or low-calibration; otherwise it enters
ForesightKV's lane.

### PM-KVQ

PM-KVQ targets long-CoT KV quantization. It argues naive quantization hurts due
to cumulative error and short-context calibration under RoPE. It uses progressive
mixed precision, block-wise memory allocation, and positional-interpolation
calibration.

Implication: FP8 alone is not novel. ThoughtFlow needs FP8 as the byte-budget
enabler inside a retention policy, and must test cumulative-error effects.

## Quick COLM_v3-Useful Artifact

Status: **HISTORICAL PRE-FALSIFICATION SCOPING**. The checklist below is
preserved as reviewer context for why the branch needed strict gates. It is not
the current paper framing; the current contribution is the falsification ladder
documented in `../phase2/current_decision_manifest_20260506.md`.

Yes: a reviewer-grounded failure matrix emerged. The most useful systems artifact
is not an implementation yet; it is a concrete design checklist:

1. Show end-to-end SGLang/vLLM-style speed or explicitly call Phase 0 Mac-only
   results non-systems.
2. Plot Pareto curves versus R-KV/ThinKV/LongFlow, not single points.
3. Include a vanilla reasoning-budget truncation baseline.
4. Preserve anchors/fair spans and report eviction-bias/keep-rate telemetry.
5. Quantify numerical error from FP8 and LongFlow-like approximations separately.
6. Test recurrence and phase-transition tokens, not just first/recent tokens.

## Superseded Recommendation

Historical recommendation before the RDU/PSI/VWAC stop ladder: pivot, not kill.
Do not proceed as "LongFlow + FP8 + phase awareness." The field is too crowded
and ThinKV already claims thought-adaptive quantization/eviction. That positive
method branch is now stopped; the current camera-ready contribution is the
falsification ladder rather than:

> Retrofit, bias-controlled reasoning KV compression for existing models:
> FP8 byte budget plus explicit anchor/fair-span/phase-transition protection,
> evaluated against LongFlow's documented reviewer failures.
