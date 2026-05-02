# ARC Sparse-Query Cache-Bottleneck Gate

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: ARC/OpenBookQA public-basis packets and systems byte/exposure
  accounting are the defensible core, while ARC source-family repair remains
  unsolved.
- Exact gap: a nonlinear TinyLlama hidden/query cache bottleneck does not beat
  Qwen-substituted packets or cached TinyLlama packets on frozen ARC
  disagreement rows.

## Gate

New script:
`scripts/build_source_private_arc_challenge_sparse_query_cache_bottleneck_gate.py`

Artifact:
`results/source_private_arc_challenge_sparse_query_cache_bottleneck_gate_20260502_tinyllama_disagreement/`

This gate is the nonlinear follow-up to the negative TinyLlama hidden/query
PCA/ridge and static transport gates. It reuses cached TinyLlama hidden/query
features from the answer-key-forbidden ARC prompts, trains/selects only on the
`144` ARC validation rows where TinyLlama and Qwen-0.5B source packets
disagree, then evaluates once on the frozen `473` ARC test-disagreement rows.

The communicated object remains the same `12B` sparse signed ARC packet. The
new connector:

- projects row-centered hidden/query residual views with train-only PCA;
- expands them with random Fourier features;
- keeps only the largest sparse query coordinates;
- ridge-decodes those coordinates into the public ARC Fourier/anchor receiver
  residual basis;
- emits the normal fixed-byte packet rather than raw hidden states, KV cache,
  source text, or dense vectors.

## Result

- pass gate: `False`
- selected view: `hidden_query_residual`
- selected PCA/RFF/active/gamma/ridge: `16 / 32 / 16 / 1.0 / 1000.0`
- validation disagreement rows: `144`
- test disagreement rows: `473`
- frontier candidates: `216`
- test matched mean: `0.248203`
- test Qwen-substituted packet mean: `0.317125`
- test cached Tiny packet mean: `0.269345`
- test matched minus Qwen-substituted mean: `-0.068922`
- test matched minus cached Tiny mean: `-0.021142`
- test CI95 lower bound versus Qwen-substituted: `-0.138531`
- test CI95 lower bound versus cached Tiny: `-0.087791`
- candidate-roll control mean: `0.260465`
- content-rotation control mean: `0.281607`
- spectral-permutation control mean: `0.251163`
- final fit alignment cosine mean: `0.184738`
- final sparse activation fraction: `0.500000`

The selected validation row looked superficially promising on held-out
validation rows: matched `0.383333` versus Qwen-substituted `0.361111` and
cached Tiny `0.277778`. Its validation CI lower bound versus Qwen-substituted
was still negative at `-0.277778`, and the frozen test result did not transfer.

## Decision

Rule out this Mac-local nonlinear sparse-query cache bottleneck over the
current TinyLlama ARC hidden/query caches. The selected connector is worse than
Qwen-substituted packets and worse than cached TinyLlama packets on frozen
test-disagreement rows.

This weakens the branch that TinyLlama's cached hidden/query means contain an
easy low-data nonlinear map into the public ARC packet basis. It does not rule
out larger trainable query bottlenecks, sparse crosscoders trained on more
matched activations, stronger non-Qwen source models, or NVIDIA-scale connector
training.

## Lay Explanation

This experiment gave TinyLlama a small learned translator before it sent its
usual 12-byte ARC hint. The translator asked sparse nonlinear questions of
TinyLlama's internal hidden/query state, converted those answers into the same
public packet coordinate system used by Qwen, and then sent only the tiny
packet. On new hard rows, that translated hint was still worse than simply
using Qwen's own packet.

## Next Gate

Stop spending Mac-local cycles on TinyLlama hidden/query PCA, ridge,
Procrustes, nearest-neighbor transport, sign-sketch transport, or random
Fourier sparse-query bottlenecks for this ARC disagreement surface. The next
highest-value gate is either:

- a stronger true non-Qwen source-family run on NVIDIA; or
- a larger trainable query/cache connector or sparse crosscoder with more
  matched activations, strict `1-12B` discrete packets, seed repeats, and
  zero-source, wrong-row, candidate-roll, content-rotation, same-byte text, and
  Qwen-substituted controls.
