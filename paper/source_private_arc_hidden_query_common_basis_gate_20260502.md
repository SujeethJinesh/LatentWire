# ARC Hidden/Query Common-Basis Gate

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: ARC/OpenBookQA public-basis packets and HellaSwag
  hidden-innovation packets are useful fixed-byte source-private evidence, but
  ARC source-family transfer is not solved.
- Exact gap: the current TinyLlama hidden/query connector does not beat
  Qwen-substituted packets on frozen ARC disagreement rows.

## Gate

New script:
`scripts/build_source_private_arc_challenge_hidden_query_common_basis_gate.py`

Artifact:
`results/source_private_arc_challenge_hidden_query_common_basis_gate_20260502_tinyllama_disagreement/`

This gate trains on the `144` ARC validation rows where TinyLlama and
Qwen-0.5B source packets disagree, then evaluates once on the frozen `473`
ARC test-disagreement rows from the TinyLlama source-family falsification
artifact.

The connector extracts TinyLlama per-choice hidden means and attention-query
projection means from answer-key-forbidden prompts, builds row-centered
candidate residual views, compresses them through a train-only PCA/ridge map
into the public ARC Fourier/anchor receiver basis, and emits the same `12B`
sparse signed packet used by the ARC public-basis gates.

## Result

- pass gate: `False`
- selected validation view: `hidden_residual`
- selected PCA/ridge: `32` / `100.0`
- validation dev delta versus Qwen-substituted packet: `-0.016667`
- test matched mean/min: `0.229598 / 0.211416`
- test Qwen-substituted packet mean: `0.317125`
- test cached Tiny packet mean: `0.269345`
- test matched minus Qwen-substituted mean/min: `-0.087526 / -0.105708`
- test CI95 lower bound versus Qwen-substituted: `-0.159672`
- test matched minus cached Tiny mean: `-0.039746`
- candidate-roll control mean: `0.256660`
- receiver spectral-permutation control mean: `0.251163`
- hidden/query extraction: validation `65.59s`, test `218.24s` on CPU
- raw packet: `12B`; framed record: `15B`

## Decision

Rule out this Mac-local TinyLlama hidden/query common-basis connector as the
ARC source-family repair. It is weaker than both the cached Tiny packet and the
Qwen-substituted packet on the strongest available disagreement surface.

This does not kill the whole paper. It narrows the claim:

1. ARC Fourier/anchor-syndrome remains a positive public-coordinate packet
   method with Qwen source caches.
2. Qwen-1.5B remains a strong same-family source-strength diagnostic on frozen
   test, but not a cross-family claim.
3. TinyLlama/Phi-3 plus cached routers, cached candidate connectors, scalar
   confidence, and this hidden/query map are now ruled out for ARC source-family
   repair.

## Lay Explanation

We looked only at questions where TinyLlama and Qwen wanted to send different
tiny hints. Then we asked whether TinyLlama's internal activations could be
translated into the same coordinate system used by our successful ARC packet.
They could not: the translated TinyLlama hint was worse than just using the
ordinary TinyLlama hint, and much worse than using Qwen's hint.

## Next Gate

Do not spend more Mac time on shallow ARC hidden/query PCA/ridge connectors.
The next high-value branches are:

- native systems boundary figure/table V3 for COLM/ICLR systems framing;
- NVIDIA run for a stronger true cross-family source or a trainable query/cache
  connector;
- HellaSwag hidden-innovation full-validation/native-systems consolidation if
  the paper needs a positive method branch before new GPU access.
