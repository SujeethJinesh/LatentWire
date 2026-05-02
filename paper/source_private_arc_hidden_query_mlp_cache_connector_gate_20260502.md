# ARC Hidden/Query MLP Cache Connector Gate

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: ARC/OpenBookQA fixed-byte public-basis packets and
  byte/exposure systems accounting are the defensible core, while learned
  cross-model/common-language transfer remains unsolved.
- Exact gap: a train-only MLP cache-to-packet connector over the current
  TinyLlama hidden/query caches does not beat Qwen-substituted packets or
  cached TinyLlama packets on frozen ARC disagreement rows.

## Gate

New script:
`scripts/build_source_private_arc_challenge_hidden_query_mlp_cache_connector_gate.py`

Artifact:
`results/source_private_arc_challenge_hidden_query_mlp_cache_connector_gate_20260502_tinyllama_disagreement/`

This gate is the bounded learned follow-up to the negative TinyLlama
hidden/query PCA/ridge, transport/common-basis, and sparse-query
cache-bottleneck gates. It reuses cached TinyLlama hidden/query features from
the answer-key-forbidden ARC prompts, trains/selects only on the `144` ARC
validation rows where TinyLlama and Qwen-0.5B source packets disagree, then
evaluates once on the frozen `473` ARC test-disagreement rows.

The communicated object remains the same `12B` sparse signed ARC packet. The
new connector:

- row-centers TinyLlama hidden/query candidate features;
- fits train-only PCA on validation fit rows;
- trains a one-hidden-layer MLP on CPU to decode into the public ARC
  Fourier/anchor receiver residual basis;
- selects by held-out validation disagreement packet accuracy against
  Qwen-substituted and cached-Tiny baselines;
- emits the standard fixed-byte packet rather than raw hidden states, KV cache,
  source text, source answer labels, or dense vectors.

This is not a full target-LM soft-prefix or C2C-style KV connector. It is the
cheapest Mac-local approximation of a learned query/cache connector that the
current caches support.

## Result

- pass gate: `False`
- selected view: `query_residual`
- selected PCA/hidden/weight decay: `16 / 16 / 0.001`
- validation disagreement rows: `144`
- test disagreement rows: `473`
- frontier candidates: `36`
- test matched mean: `0.231712`
- test Qwen-substituted packet mean: `0.317125`
- test cached Tiny packet mean: `0.269345`
- test target-only mean: `0.268499`
- test same-byte structured text mean: `0.257928`
- test matched minus Qwen-substituted mean: `-0.085412`
- test matched minus cached Tiny mean: `-0.037632`
- test CI95 lower bound versus Qwen-substituted: `-0.154334`
- test CI95 lower bound versus cached Tiny: `-0.103700`
- candidate-roll control mean: `0.261311`
- content-rotation/wrong-row control mean: `0.255391`
- receiver spectral-permutation control mean: `0.235941`
- final fit alignment cosine mean: `0.231324`
- final fit loss: `0.801121`

The selected validation row was not cleanly positive: it was selected because
it was the best held-out validation tradeoff, but the validation CI lower bound
against Qwen-substituted packets remained negative. Frozen test then fell below
Qwen-substituted, cached Tiny, target-only, and same-byte structured text.

## Decision

Rule out the current Mac-local TinyLlama hidden/query mean-cache connector
family for ARC:

- static PCA/ridge common-basis mapping is negative;
- nearest-neighbor/sign-sketch/Procrustes transport is negative;
- random Fourier sparse-query cache bottlenecks are negative;
- this train-only MLP cache-to-packet connector is negative.

This does not rule out a true query-bottleneck/soft-prefix method. It rules out
the particular low-data setup where only per-choice mean hidden/query vectors
are available and the connector must learn from `144` validation disagreement
rows. A true soft-prefix or cache-fuser branch needs tokenwise target-forward
infrastructure, more matched activation pairs, and likely NVIDIA hardware.

## Lay Explanation

This experiment tried a small learned translator instead of another hand-built
geometric map. TinyLlama looked at the ARC question and answer choices. The
translator compressed each answer option's cached internal signal into the same
public packet language used by the receiver. Only a tiny 12-byte hint crossed
the boundary. On new hard examples, that learned hint was still worse than
using Qwen's own tiny hint.

## Next Gate

Stop widening TinyLlama mean hidden/query cache-to-packet variants on this ARC
surface. The next exact gate should be one of:

- true tokenwise query-bottleneck/soft-prefix connector with frozen endpoints,
  source-destroying controls, and prompt/prefix/KV baselines; or
- native systems rows for LatentWire versus vLLM/SGLang/C2C/KVComm/KVCOMM/QJL/
  TurboQuant once NVIDIA hardware is available.

For Mac-only work, the highest-value remaining task is paper-critical
consolidation: update the contribution boundary, make the negative ladder
explicit, and prepare the runbook for the tokenwise/NVIDIA connector rather
than running more low-data TinyLlama cache variants.
