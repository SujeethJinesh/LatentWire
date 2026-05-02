# ARC Transport Common-Basis Gate

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: ARC/OpenBookQA public-basis packets and systems byte/exposure
  accounting are the defensible core, while ARC source-family repair remains
  unsolved.
- Exact gap: transport-style TinyLlama hidden/query alignment does not beat
  Qwen-substituted packets on frozen ARC disagreement rows.

## Gate

New script:
`scripts/build_source_private_arc_challenge_transport_common_basis_gate.py`

Artifact:
`results/source_private_arc_challenge_transport_common_basis_gate_20260502_tinyllama_disagreement/`

This gate reuses the cached TinyLlama hidden/query features from the previous
ARC hidden/query common-basis failure. It trains/selects only on the `144` ARC
validation rows where TinyLlama and Qwen-0.5B source packets disagree, then
evaluates once on the frozen `473` ARC test-disagreement rows.

The communicated object remains the same `12B` sparse signed ARC packet. The
new connector variants are:

- local nearest-neighbor barycentric transport in TinyLlama feature space;
- QJL-style sign-projected nearest-neighbor transport;
- whitened orthogonal Procrustes alignment.

## Result

- pass gate: `False`
- selected validation method: `whitened_procrustes`
- selected view: `query_residual`
- selected transform: `raw`
- selected parameter: `dim=32`
- validation disagreement rows: `144`
- test disagreement rows: `473`
- frontier candidates: `48`
- test matched mean: `0.228753`
- test Qwen-substituted packet mean: `0.317125`
- test cached Tiny packet mean: `0.269345`
- test matched minus Qwen-substituted mean: `-0.088372`
- test matched minus cached Tiny mean: `-0.040592`
- test CI95 lower bound versus Qwen-substituted: `-0.160677`
- candidate-roll control mean: `0.251163`
- spectral-permutation control mean: `0.257082`

## Decision

Rule out this Mac-local transport/common-basis connector family for the ARC
TinyLlama-vs-Qwen source-family repair. The best validation-selected row is
worse than both the cached Tiny packet and the Qwen-substituted packet on the
frozen test disagreement surface.

This weakens the idea that the current TinyLlama hidden/query caches contain a
linearly or locally translatable signal into the public ARC receiver basis. It
does not rule out richer learned query bottlenecks, nonlinear sparse
crosscoders, stronger source models, or NVIDIA-scale connector training.

## Lay Explanation

The experiment asked whether we could make TinyLlama's internal hints easier
for Qwen to use by translating them into a shared coordinate system. We tried
three simple translators: copying from nearby examples, copying from nearby
random sign sketches, and rotating the two spaces into alignment. None helped
on new held-out rows.

## Next Gate

Do not spend more time on shallow ARC hidden/query PCA, ridge, Procrustes, or
static nearest-neighbor transport over the current TinyLlama caches. The next
highest-value method gate is a trainable query/cache bottleneck or nonlinear
sparse crosscoder that emits a strict `1-12B` discrete packet and is evaluated
against Qwen-substituted packets, candidate-roll, zero-source, wrong-row, and
same-byte text controls. If NVIDIA access arrives first, repeat the
source-family gate with a stronger true non-Qwen source.
