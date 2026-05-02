# Source-Private ARC Residual Receiver Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets and systems byte/exposure
  accounting are credible, but learned receivers have not yet proven
  matched-source necessity.
- Exact gap: the frozen-target-public residual receiver fails the n8 ARC gate.

## What Changed

Extended `scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`
with a `target_residual` receiver mode.

The mode changes the decision surface:

1. Train a public target candidate scorer first.
2. Freeze its scores.
3. Train only a source-dependent residual correction.
4. Add the residual to frozen target-public scores at evaluation.

The residual design intentionally excludes public-only features and uses a
no-intercept ridge solve. That makes `zero_source` exactly reproduce the
target-public scores, so any gain must come from source-dependent terms:
`dot(source, public)`, `source`, and `source * public`.

Lay explanation: the target model gets to make its own guess first. The source
is allowed to send a tiny correction. If erasing the source correction gives
the same result as the target's original guess, then we know the correction
channel is not secretly relearning the target's public guess.

## Evidence

Focused tests:

- `./venv_arm64/bin/python -m py_compile scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`
- `./venv_arm64/bin/python -m pytest tests/test_run_source_private_arc_candidate_alignment_receiver_preflight.py -q`

All six unit tests pass.

Mac-local ARC residual rows:

| Artifact | Fit policy | Sketch | Packet bytes | Matched | Target-public / zero-source | Best control | Margin delta | Pass |
|---|---|---|---:|---:|---:|---:|---:|---|
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_residual_hidden_public_innovation_sign16_n8/` | target errors | sign-16 | 8B | `2/4` | `3/4` | `3/4` | `-0.173` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_residual_hidden_public_innovation_int8_n8/` | target errors | int8-16 | 66B | `1/4` | `3/4` | `3/4` | `-0.039` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_residual_hidden_public_innovation_none_n8/` | target errors | float-16 | 256B | `1/4` | `3/4` | `3/4` | `-0.039` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_residual_hidden_public_innovation_sign16_all_n8/` | all fit rows | sign-16 | 8B | `2/4` | `3/4` | `3/4` | `-0.014` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_residual_hidden_public_innovation_none_all_n8/` | all fit rows | float-16 | 256B | `3/4` | `3/4` | `3/4` | `-0.033` | False |

## Interpretation

This branch is weakened.

The target-error-only setting is too data-starved on the n8 split: the frozen
public scorer makes only one fit-row mistake, so the residual over-corrects
held-out rows that target-public already solved. The all-fit fallback is more
stable, but still only ties target-public by accuracy and loses on margin even
with unquantized 256B diagnostic sketches.

The most important positive engineering result is control hygiene: zero-source
now matches target-public exactly. That means the negative result is cleaner
than the previous direct candidate-alignment gate.

## Decision

Do not widen the linear residual receiver.

The next highest-value method branch should be one of:

- a consistency-style packet repair receiver trained with corruptions
  (`wrong-row`, `candidate-roll`, `label-shuffle`, `same-norm noise`) and
  evaluated as one-step repair over target-public plus packet state;
- a true permutation-equivariant DeepSets/Set Transformer receiver that shares
  candidate weights and is trained with source-control contrastive losses.

Do not claim resonance/activation matching yet. The right claim boundary is
behavioral equivalence under source-destroying controls, with selective hidden
or logit probes only as diagnostics.
