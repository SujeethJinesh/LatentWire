# Source-Private ARC-Challenge Bridge Contract, 2026-05-01

## Status

- artifact:
  `results/source_private_arc_challenge_bridge_contract_20260501/`
- code:
  `scripts/build_source_private_arc_challenge_bridge_contract.py`
- test:
  `tests/test_build_source_private_arc_challenge_bridge_contract.py`
- references:
  `references/567_arc_challenge_bridge_contract_refs_20260501.md`

## Update

The public benchmark gate is now frozen as an ARC-Challenge bridge contract.
The script materialized official `allenai/ai2_arc` `ARC-Challenge` splits into
the result directory and audited both the local smoke slices and official
splits.

Local smoke slices:

- `data/arc_challenge_gate_15.jsonl`: `15` rows,
  sha256 `3e7325279f747151e83b19d032e7612d314db8c66a92633bf596dd8411947b48`
- `data/arc_challenge_eval_35.jsonl`: `35` rows,
  sha256 `9b7644d5187cfdc2f170936026b124a26b2fc37ff1d984817f21adaf371add02`
- `data/arc_challenge_50.jsonl`: combined smoke slice with exactly the
  `15+35` local rows

Official ARC-Challenge materialized rows:

- train: `1119`
- validation: `299`
- test: `1172`

The local validation/eval smoke overlap is `0`, and official train,
validation, and test have no cross-split content overlap. The train split has
one duplicate content hash within train; this is recorded as a warning, not a
cross-split leakage failure.

## Contract

The public ARC method must use the fixed `12B` packet selected by the stable-gap
internal validation rule. The source packet builder must not consume `answer`,
`answerKey`, answer labels, gold rationales, or correct-option markers.

Required public controls:

- target-only / zero-source
- shuffled-source packet
- random same-byte packet
- target-derived sidecar
- answer-only text as a forbidden leakage oracle
- same-byte structured text
- label permutation
- candidate derangement

## Interpretation

This is not a positive ARC result yet. It closes the reproducibility and
leakage-control setup for the public benchmark bridge. The next gate is the
actual fixed-12B source-private ARC run with paired uncertainty against
target-only and every destructive control.

## Next Exact Gate

Implement/run the fixed-12B source-private packet on official ARC-Challenge
validation/test with label permutation, shuffled-source, same-byte text,
target-derived, random, and candidate-derangement controls. Do not consume
`answerKey` when building the source packet.
