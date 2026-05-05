# ARC Tokenwise Source-Evidence Preflight

- created UTC: `2026-05-04T23:59:00Z`
- COLM_v2 readiness effect: negative/guardrail backport only
- ICLR readiness effect: still blocked
- gate: `arc_n32_tokenwise_source_evidence_preflight`
- status: `blocked_on_mps_qwen3_prefix_scoring_with_negative_cpu_smoke`

## Current Story

The live ICLR direction is still to distill dense C2C-style transfer into
sparse, source-private packets. The immediate question was whether token-level
source traces can drive a target-loss soft-prefix receiver better than the
source-choice and same-byte controls that have invalidated prior shallow rows.

## What Ran

I used the existing ARC/OpenBookQA soft-prefix preflight with:

- `source_feature_mode=hf_choice_token_hidden_pool_residual`
- Qwen2.5-0.5B-Instruct as the source feature model
- Qwen3-0.6B as the frozen target model
- answer-key-forbidden ARC validation source cache
- rank-3 token-pool source summaries
- target-loss soft-prefix receiver
- strict controls including target-only, zero-source, shuffled source,
  source-row shuffle, target-derived prefix, candidate roll, atom shuffle,
  source-index/rank/score, and same-byte visible text

## MPS n32 Attempt

The intended n32 MPS run reached target scoring but failed inside MPSGraph with
a Qwen3 prefix-embedding attention shape error:

```text
LLVM ERROR: Failed to infer result type(s):
"mps.matmul"(...) : (tensor<1x16x90x128xf32>, tensor<1x8x128x90xf32>) -> ( ??? )
```

I patched padded choice scoring to use the unbatched scorer on MPS. That reduced
the failing tensor from a padded four-choice batch to a single-choice call, but
Qwen3/MPS still failed with prefix `inputs_embeds`. This is a Mac execution
blocker, not a scientific pass/fail.

## CPU n8 Smoke

Artifact:
`results/source_private_arc_tokenwise_source_evidence_preflight_20260504_n8_cpu_qwen05_to_qwen3/arc_openbookqa_soft_prefix_preflight.json`

Headline:

| Metric | Value |
|---|---:|
| pass gate | `False` |
| fit rows | `4` |
| eval rows | `4` |
| source tensor rank | `3` |
| source pool size | `16` |
| source token count mean | `26.875` |
| matched soft-prefix accuracy | `0.000` |
| target-only accuracy | `0.250` |
| best control | `packet_only_source_index` |
| best-control accuracy | `0.750` |
| same-byte visible text accuracy | `0.500` |
| matched minus best-control accuracy | `-0.750` |
| matched minus best-control margin | `-4.598476` |

## Interpretation

This weakens the current token-pooled source-to-soft-prefix receiver. It does
not rule out tokenwise source evidence, because the target prefix receiver is
itself unstable and the full n32 MPS run is blocked. It does rule out promoting
the current n8 CPU token-pool residual soft-prefix smoke as positive evidence.

In lay terms: we tried to feed the receiver a compact set of token-level clues
from the source model. On the tiny CPU smoke, the learned soft prompt did worse
than simply copying the source model's favorite answer option. That means this
specific receiver is not yet carrying useful source reasoning.

## Decision

- `weakened`: Qwen3 target soft-prefix receiver over token-pooled source traces.
- `alive`: tokenwise source evidence as a diagnostic signal.
- `blocked`: MPS/Qwen3 prefix `inputs_embeds` path for the full n32 gate.
- `not promoted`: broad Sparse Resonance Packets or C2C-effect distillation.

## Next Exact Gate

Completed in `paper/arc_tokenwise_repair_readout_preflight_20260504.md`.

The readout failed: matched accuracy was `0.3125`, the source-index control was
`0.4375`, and matched tied zero-source, wrong-row, source-row shuffle,
same-source-choice wrong-row, atom-shuffle, and coefficient-shuffle controls.
This rules out the row-level token-pool repair readout as a source-causal
preflight.

The next gate is now `arc_candidate_local_source_evidence_repair_preflight`:
preserve candidate-local source hidden structure rather than flattening a
row-level token pool.
