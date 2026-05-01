# Candidate-Local Cross-Family Gate

- date: `2026-04-30`
- status: reviewer-facing same-family vs cross-family separation artifact for
  the live candidate-local residual receiver
- code: `scripts/build_source_private_candidate_local_cross_family_gate.py`
- artifact:
  `results/source_private_candidate_local_cross_family_gate_20260430/`
- references:
  `references/552_candidate_local_cross_family_gate_refs_20260430.md`

## What Changed

The live n512 candidate-local residual evidence is now separated into
same-family and cross-family slices instead of being reported only as one
aggregate `9/9` row. The artifact reads the existing common-basis falsification
table and reports, for each method family, whether it passes
`core_to_holdout`, `holdout_to_core`, and `same_family_all`.

## Main Evidence

Summary artifact:
`results/source_private_candidate_local_cross_family_gate_20260430/candidate_local_cross_family_gate.md`

Headline:

- pass gate: `true`;
- live candidate-local residual cross-family rows: `6/6` pass;
- live candidate-local residual same-family rows: `3/3` pass;
- live cross-family matched accuracy minimum: `0.500`;
- live cross-family best-control maximum: `0.260`;
- RR anchor-coordinate cross-family rows: `3/6` pass;
- RR core-to-holdout rows: `3/3` pass;
- RR holdout-to-core rows: `0/3` pass;
- RR same-family rows: `3/3` pass;
- ICLR ready: `false`.

Key interpretation: the live candidate-local residual receiver is not merely a
same-family chart on this surface. It passes both cross-family directions across
seeds `47/53/59`, with controls near the target floor. The strongest clean
common-basis competitor, Relative Representations-style anchor coordinates, is
asymmetric: it passes core-to-holdout and same-family, but all holdout-to-core
rows collapse to the target floor. Public transport/common-basis rows remain
unsafe because destructive controls rise. Two simple RR repair probes have now
been checked: raw anchor-prior innovation passes only the core-to-holdout rows
and leaks controls, while ranked anchor-prior innovation passes no rows and
pushes holdout-to-core below target.

## Layman Explanation

This asks whether the method only works when train and test examples come from
the same kind of task. The current method still works when training on one
family and evaluating on the other in both directions. The strongest clean
shared-coordinate baseline only works in one direction, so it is real prior art
but not an all-direction replacement. The newest repair attempts tried to
subtract the receiver's local anchor prior; they did not make that one-way
baseline bidirectional.

## Safe Claim

On the current n512 held-out packet surface, the normalized candidate-local
residual receiver passes both same-family and bidirectional cross-family gates
under strict source-destroying controls. RR anchor coordinates are a clean
partial competitor, not a defeated baseline.

## Non-Claims

- This is not a new benchmark family beyond the existing synthetic candidate
  surface.
- This does not replace native C2C/KVComm/TurboQuant systems comparisons.
- This does not prove full dense latent-transfer RR/LSTIRP is defeated.
- This does not prove transfer across unrelated real LLM families.

## Remaining ICLR Gap

Comfortable ICLR still needs either a guarded RR/candidate-local stack that
fixes holdout-to-core without leaking controls through a new hypothesis, or a
very explicit limitation that RR is the clean one-way competitor. The simple RR
innovation repairs are already pruned. The systems side still needs native or
proxy C2C/KVComm rows and KV-compression byte floors.

## COLM Workshop Use

This is COLM-useful now: it turns the live result into a clearer claim that is
not same-family-only, while preserving the limitation that the current
cross-family surface is synthetic and RR remains alive.
