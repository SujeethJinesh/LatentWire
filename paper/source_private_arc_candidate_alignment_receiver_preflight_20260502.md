# Source-Private ARC Candidate-Alignment Receiver Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible because the systems
  byte-amplification story is now concrete; ICLR full paper is still blocked.
- Current story: LatentWire has fixed-byte source-private packets and strong
  destructive-control discipline, but still needs a positive receiver proving
  matched-source necessity beyond tiny smoke slices.
- Exact gap: this candidate-alignment receiver does not yet beat target-only
  and source-destroying controls with both accuracy and margin.

## What Changed

Added `scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`
and `tests/test_run_source_private_arc_candidate_alignment_receiver_preflight.py`.

The new gate trains an external candidate scorer instead of injecting a prefix
into the target LM. It builds one source candidate slot per answer choice,
projects each slot into a fixed sketch, optionally quantizes that sketch, and
trains a ridge receiver to score candidates from:

- public target-side candidate features;
- source candidate sketch features;
- source/public compatibility terms;
- candidate index and choice-count metadata.

Controls include target-public-only, zero-source, shuffled-source, same-norm
noise, train-mean source, label-shuffled receiver, candidate-roll source,
candidate-deranged scores, same-byte visible text, and a source-label-copy
audit upper bound.

Lay explanation: every answer choice gets a tiny hint from the source model.
The receiver is a simple referee that tries to use those hints. We then break
the hints in several ways. If the real hints do not beat the broken hints, the
method has not shown real communication.

## Evidence

Focused tests:

- `./venv_arm64/bin/python -m py_compile scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`
- `./venv_arm64/bin/python -m pytest tests/test_run_source_private_arc_candidate_alignment_receiver_preflight.py -q`

All four unit tests pass.

Mac-local ARC smoke rows:

| Artifact | Rows | Sketch | Objective | Packet bytes | Matched | Best control | Margin delta | Pass |
|---|---:|---|---|---:|---:|---:|---:|---|
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_public_innovation_sign16_n8/` | 8 | sign-16 | pointwise | 8B | `1/4` | `3/4` | `-0.142` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_public_innovation_float16_n8/` | 8 | float-16 | pointwise | 256B | `3/4` | `2/4` | `-0.012` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_score_public_innovation_sign16_n8/` | 8 | sign-16 + score | pointwise | 8B | `2/4` | `2/4` | `-0.008` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_public_innovation_sign64_n8/` | 8 | sign-64 | pointwise | 32B | `1/4` | `2/4` | `-0.141` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_public_innovation_int8_pairwise16_n8/` | 8 | int8-16 | pairwise | 66B | `3/4` | `3/4` | `-0.038` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_public_innovation_none_pairwise16_n8/` | 8 | float-16 | pairwise | 256B | `3/4` | `3/4` | `-0.037` | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_hidden_public_innovation_none_pointwise16_n16/` | 16 | float-16 | pointwise | 256B | `1/8` | `3/8` | `-1.389` | False |

## Interpretation

The branch is weakened, not promoted.

The only encouraging row is unquantized hidden-only pointwise at n8: matched
gets `3/4` while candidate-roll and candidate-derangement both get `0/4`.
That says candidate slot alignment can matter in the tiny setting. But it is
not claimable: the mean gold margin remains below target-public-only, the
packet is a 256B float diagnostic rather than a compact sign/int8 packet, and
the same setup collapses on the n16 probe.

Hard sign sketches are especially bad here. The 8B sign-16 row falls to `1/4`
while zero-source reaches `3/4`, and sign-64 still fails. Int8 preserves more
information, but pairwise ranking also makes the target-public-only receiver
reach `3/4`, so matched source is not necessary.

## Decision

Do not widen this low-capacity external candidate receiver as-is. Log it as a
falsified/weak branch.

The next highest-value method branch is a residual receiver: freeze a
target-public scorer, train only a source correction against target errors, and
evaluate whether the correction adds examples that target-only misses while
candidate-roll/wrong-row controls fail. If that still fails, move to a true
permutation-equivariant DeepSets/Set Transformer receiver with source-control
contrastive training rather than another linear scorer.
