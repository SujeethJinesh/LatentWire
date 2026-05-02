# Source-Private ARC Consistency Repair Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets and byte/exposure systems
  accounting are credible, but learned receivers still have not proven matched
  source necessity over target-public controls.
- Exact gap: the consistency-style repair receiver does not clear the ARC n8
  target-public/zero-source gate.

## What Changed

Added `target_consistency_repair` mode to
`scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`.

The receiver keeps the same frozen-base contract as `target_residual`:

1. train a target-public candidate scorer;
2. freeze target-public scores;
3. train a no-intercept source-dependent repair delta;
4. evaluate `target_public + repair_delta`.

The repair objective uses matched and masked source views to push gold-vs-
distractor residual margins upward. It also trains corrupted source views
(`shuffled_source`, `same_norm_noise`, `train_mean_source`,
`target_derived_source`, `candidate_roll_source`) toward zero residual. The
feature map includes only source-dependent terms: source norm/agreement,
source-public compatibility, and those quantities gated by frozen target-score
gaps. It does not include an intercept, raw public vectors, raw target scores,
candidate-index constants, or a `has_source` bit.

Lay explanation: the target first makes its own guess. The source can send a
small correction note. During training, real notes are supposed to fix the
answer, while broken notes are taught to do nothing. If erasing the source note
changes the target's original guess, the receiver is cheating; this gate checks
that it does not.

## Evidence

Focused tests:

- `./venv_arm64/bin/python -m py_compile scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`
- `./venv_arm64/bin/python -m pytest tests/test_run_source_private_arc_candidate_alignment_receiver_preflight.py -q`

All nine focused unit tests pass.

Mac-local ARC consistency-repair rows:

| Artifact | Sketch | Packet bytes | Matched | Target-public / zero-source | Best control | Margin delta | Zero-source exact | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_consistency_repair_hidden_public_innovation_sign16_n8/` | sign-16 | 8B | `2/4` | `3/4` | `3/4` | `+0.036` | True | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_consistency_repair_hidden_public_innovation_int8_n8/` | int8-16 | 66B | `2/4` | `3/4` | `3/4` | `-0.054` | True | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_consistency_repair_hidden_public_innovation_none_n8/` | float-16 | 256B | `2/4` | `3/4` | `3/4` | `-0.056` | True | False |

The new `target_derived_source` control stays below target-public on all three
rows (`1/4`). That is useful: the failed repair is not explained by public
features masquerading as source packets.

## Interpretation

This branch is weakened.

The consistency repair objective is stricter than the previous residual
receiver and gives a cleaner reviewer-facing control surface, but it does not
fix the target-public mistake on the n8 split. The compact sign-16 row improves
mean margin while still losing accuracy, and int8/unquantized diagnostics do
not recover accuracy. Because the unquantized 256B diagnostic also fails, the
problem is not only sketch quantization.

The dominant failure mode is accepted harm: the repair delta can move
target-correct rows off the right answer. That argues against widening this
linear feature repair branch. A viable next method needs either a true
permutation-equivariant candidate set receiver or a learned accept/abstain
policy with explicit help/harm telemetry.

## Decision

Do not widen `target_consistency_repair` as implemented.

Promote the control hygiene:

- `zero_source` exactly equals `target_public_only` for frozen-base modes;
- `target_derived_source` is now a pass-critical destructive control;
- corrupted source views are trained toward zero residual.

Next exact gate: implement a permutation-equivariant DeepSets/Set Transformer
candidate receiver over the same frozen target-public base, with the same
source-destroying controls and an explicit accept/abstain head. If that fails,
cut learned ARC candidate receivers and shift ICLR method work back to the
existing positive Fourier/anchor-syndrome packet row plus cross-family/native
systems validation.
