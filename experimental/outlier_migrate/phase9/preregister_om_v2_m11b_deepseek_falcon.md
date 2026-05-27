# V2 M11b Cross-Architecture Baseline Vetting Preregistration

**Frozen on**: 2026-05-27
**Frozen by**: Codex, after Stage 1 integration, citation additions, math
formalization, and before any DeepSeek/Falcon M11b score cache was inspected.
**Status**: Frozen baseline-vetting preregistration.

## Purpose

Nemotron-3 M11b top-10 passed after the corrected static-top-1% salvage. This
run tests whether M11b top-10 transfers to the two other measured architectures:
a pure Transformer control and a parallel Falcon-H1 hybrid. The result decides
whether M11b is a broader positive method or a Nemotron-specific scoped remedy.

## Models

1. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
   - HuggingFace snapshot commit:
     `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
2. `tiiuae/Falcon-H1-0.5B-Instruct`
   - HuggingFace snapshot commit:
     `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368`

## Trace Set

- Source: AIME-2025
- Count: 12 deterministic prompts, indices `0-11`
- Prompt file: `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- The canonical prompt hash must match the existing Phase 9 AIME slice.

## Reused Artifacts

The run may reuse BF16 target traces from the Stage 1 E1 packet:

`experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z`

Allowed reuse:

- DeepSeek BF16 traces from the `deepseek_r1_distill_qwen_1_5b` subdirectory.
- Falcon-H1 BF16 traces from the `falcon_h1_0_5b` subdirectory.

Dense activation trajectories and all M11b/static score caches must be freshly
computed by the V2 M11b runner unless a completed V2 packet is being rechecked.

## Quantization and Scoring

- Weight quantization: symmetric per-output-channel INT4 represented as
  dequantized tensors for framework compatibility.
- Activations: FP16 under the existing scoring autocast path.
- Protected channels remain in the high-precision path.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- M11b update cadence: every 100 decode positions.
- M11b alpha: `0.3`.
- Bootstrap samples: `1000`.
- Bootstrap seeds:
  - DeepSeek: `20260622`
  - Falcon-H1: `20260623`

## Regimes

Each model packet must evaluate:

1. BF16 baseline.
2. Static top-1% protected W4A16 baseline from position `100`.
3. M11b EMA-smoothed top-1%.
4. M11b EMA-smoothed top-5%.
5. M11b EMA-smoothed top-10%.
6. Static top-10% matched-budget control from position `100`.

## Metric

For each trace and non-BF16 regime:

`recovery = 1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_top1 - perplexity_BF16)`

Only traces with a positive recoverable static top-1% gap are included in the
primary recovery median. The packet must still report no-gap counts.

## Per-Model Decision Rule

Use the established M11b budget-scaling rule:

- PASS if either M11b top-5% or top-10% has median recovery at least `0.30` and
  beats static top-10% by at least `0.15`.
- KILL if top-5% and top-10% are both within `0.05` median recovery of top-1%.
- AMBIGUOUS otherwise.
- FAIL_INFRA for incomplete packets or runner/checker failures.

## Cross-Model Interpretation Rule

- If M11b top-10 passes on both models, M11b becomes a broadly transferring
  positive method across the measured small-model set.
- If M11b top-10 passes on one of two, report partial generalization and retain
  architecture-dependence framing.
- If M11b top-10 kills or is non-positive on both, the Nemotron result is
  treated as a model-specific scoped remedy.

Headline-changing outcome: both models pass with top-10 median recovery above
`0.60`. Surface this before reframing.

## Forbidden Actions

- Changing trace indices after seeing V2 score caches.
- Reusing any DeepSeek/Falcon M11b score cache from outside this V2 packet.
- Changing alpha, budgets, or decision thresholds after observing data.
- Dropping static top-10 matched-budget control.
- Using ParoQuant rotations, AWQ-style scaling, or SmoothQuant scale folding.
- Modifying model source code.
