# Durable Source-Surface Ranking

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; no live method has yet
   cleared strict source controls, target preservation, seed stability, systems
   metrics, and cross-family falsification.
2. Current paper story: existing artifacts do contain real target-complementary
   source headroom, but prior receiver/router branches failed to convert it into
   controlled communication gains.
3. Exact blocker to submission: no positive method survives on a durable
   exact-ID source surface; MPS is still blocked by orphaned PID `31103`.
4. Current live branch or top candidates: no live method branch; top candidate
   method is zero-init gated latent side-information after source-surface
   selection.
5. Highest-priority gate for this cycle: make source-surface selection durable
   and rank by clean source-only IDs.
6. Scale-up rung: smoke / branch selection.

## What Changed

- Added `scripts/rank_source_contrastive_target_sets.py`.
- Added focused tests in `tests/test_rank_source_contrastive_target_sets.py`.
- Ran the ranker over every existing `source_contrastive_target_set.json`
  artifact.
- Added a recent latent-agent communication reference memo:
  `references/469_recent_latent_agent_communication_refs.md`.

## Evidence

The durable ranker selects `svamp70_live` as the only primary-ready surface:

- `svamp70_live`: clean source-only `6/70`, raw source-only `9/70`, target/source
  oracle gain `9/70`.
- `svamp32_qwen25math`: clean `4/32`, useful only as smoke/debug.
- `svamp70_chal241_310`: clean `4/70`, useful as adjacent falsifier.
- `svamp70_holdout`: clean `2/70`, still the canonical holdout despite weak
  clean headroom.

Clean source-only IDs for the primary surface:

`14bfbfc94f2c2e7b`, `2de1549556000830`, `41cce6c6e6bb0058`,
`4d780f825bb8541c`, `bd9d8da923981d69`, `ce08a3a269bf0151`.

## Literature Update

Recent primary sources make latent/activation communication a stronger
baseline threat than plain text relay alone:

- LatentMAS (`https://arxiv.org/abs/2511.20639`) frames latent working memory
  as both accuracy and systems competition.
- Interlat (`https://arxiv.org/abs/2511.09149`,
  `https://openreview.net/forum?id=rmYbgsehTd`) strengthens the case for
  fixed-budget latent compression and target-preserving latent transfer.
- Communicating Activations Between Language Model Agents
  (`https://openreview.net/forum?id=W6RPXUUFic`) is a direct activation-sharing
  baseline for same-family and projected cross-family gates.
- Thought Communication (`https://openreview.net/forum?id=d671ljgwfY`) supports
  shared/private latent-factor framing and source-difference controls.

These sources do not change the selected surface. They change the method bar:
the next learned branch should be zero-init target-preserving, fixed-budget,
and compared against activation/latent communication baselines as well as text
relay.

## Decision

Promote `svamp70_live` as the primary surface for the next method gate. Keep
`svamp70_holdout` as the canonical replay gate. Do not spend another cycle on
shallow receiver thresholds unless there is a new source-derived feature.

## Next Exact Gate

First recheck the MPS blocker:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run the stronger-source scout from
`paper/postkill_historical_cpu_audit_20260427.md`. If that scout gives at least
six clean source-only IDs and target/source oracle gain of at least six, run the
zero-init gated latent side-information smoke on `svamp70_live` with
zero-source, shuffled-source, target-only slots, random sidecar, slots-only,
text relay, and activation-combination baselines.
