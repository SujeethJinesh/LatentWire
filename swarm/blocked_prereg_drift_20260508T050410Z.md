# Blocked: preregistration-drift audit conflict

## Status

`/goal` is paused under the stop rule:

> Any preregistration file modified during execution (audit subagent detects).

The active queue entry when this was detected was `outlier_migrate_phase1`.
Its full Granite-4.0-H-Small packet collection was already running in:

`experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`

At the time of this handoff, the runner had completed prompts 0-13 and was
continuing prompt 14. The run has not been interpreted and no checker decision
has been made.

## Audit finding

`swarm/state.json` records:

- `started_at_sha`: `6e2410fc063f1313aab0376ac76e23488b9cfb97`
- `started_at`: `2026-05-07T21:23:49Z`

The periodic audit subagent ran:

```bash
git diff --name-status 6e2410fc063f1313aab0376ac76e23488b9cfb97 -- ':(glob)**/preregister*.md'
```

It found three preregistration files added after `started_at_sha`:

```text
A	experimental/decode_microkernel/phase0/preregister_dmc_phase0.md
A	experimental/decode_microkernel/phase1/preregister_dmc_phase1.md
A	experimental/decode_microkernel/phase2/preregister_dmc_phase2.md
```

## Why this needs a human decision

There is a real policy conflict:

- The revised `swarm/goal.md` says killed entries should produce diagnostics
  and, when a plausible positive-method pivot exists, fresh preregistrations
  should be authored before the pivot runs.
- The stop rule and audit predicate treat any `preregister*.md` change after
  `started_at_sha` as preregistration drift.

The Decode Microkernel preregistrations were created as fresh positive-method
pivot preregistrations after the HybridKernel kill, not as edits to an already
executed preregistration. However, the current audit has no way to distinguish
allowed fresh-pivot preregistrations from forbidden post-hoc prereg drift.

## Human decision required

Choose one:

1. Treat added pivot preregistrations as allowed when they are fresh branch
   contracts created before any rows for that branch are inspected. Then the
   audit policy needs an explicit exception or manifest for post-start fresh
   pivot preregistrations.
2. Treat all post-start preregistration additions as forbidden drift. Then
   Decode Microkernel should be invalidated or quarantined, and the swarm
   should restart from a new clean `started_at_sha` before continuing.

Until this is resolved, no further gate decisions should be made.

## Preservation note

The in-flight OutlierMigrate Phase 1 run was not killed mid-packet. It should
be treated as uninterpreted data unless and until the human explicitly resumes
the swarm and allows the checker to run.
