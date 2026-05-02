# Source-Private ARC Equivariant Set-Delta Accept Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets and systems byte/exposure
  accounting remain the defensible core, with the Fourier/anchor-syndrome ARC
  row as the current positive method evidence. Learned ARC candidate receivers
  remain negative under target-public and source-destroyed controls.
- Exact gap: the permutation-equivariant set-delta receiver with
  accept/abstain does not produce matched-source lift on the n16 ARC gate.

## What Changed

Added `equivariant_set_delta_accept` to
`scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`.

The receiver keeps the target-public base frozen and learns only a
source-dependent candidate delta. It differs from the previous linear repair
branch in three ways:

1. candidate features are permutation-equivariant over the answer-choice set;
2. all zero-source features are exactly zero, so erased-source packets exactly
   reproduce target-public;
3. a cross-validated accept threshold either applies the repair delta or
   abstains back to target-public.

Added unit coverage for permutation equivariance, zero-source cleanliness,
accept/abstain blocking, and a synthetic matched-source repair case.

Lay explanation: the target first makes its own answer-choice scores. The
source then sends one tiny note per answer choice. The new receiver is allowed
to change the target only when its proposed change is confident enough;
otherwise it leaves the target's original answer untouched.

## Evidence

Focused tests:

- `./venv_arm64/bin/python -m py_compile scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`
- `./venv_arm64/bin/python -m pytest tests/test_run_source_private_arc_candidate_alignment_receiver_preflight.py -q`

All 12 focused unit tests pass.

Mac-local ARC n16 set-delta accept rows:

| Artifact | Sketch | Packet bytes | Matched | Target-public / zero-source | Best control | Selected accept threshold | Matched accepts | Candidate-derangement accepts / help / harm | Zero-source exact | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_equivariant_set_delta_accept_hidden_public_innovation_sign16_n16/` | sign-16 | 8B | `3/8` | `3/8` | `3/8` | `3.0` | `0/8` | `8/8 / 2 / 3` | True | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_equivariant_set_delta_accept_hidden_public_innovation_int8_n16/` | int8-16 | 66B | `3/8` | `3/8` | `3/8` | `3.0` | `0/8` | `8/8 / 2 / 3` | True | False |
| `results/source_private_arc_candidate_alignment_receiver_preflight_20260502_arc_equivariant_set_delta_accept_hidden_public_innovation_none_n16/` | float-16 | 256B | `3/8` | `3/8` | `3/8` | `3.0` | `0/8` | `8/8 / 2 / 3` | True | False |

The headline metric is not just that matched equals target-public. The accept
head selected the largest threshold and accepted zero matched-source rows. In
other words, the safest cross-validated policy is complete abstention. The
candidate-derangement audit still changes every eval row and causes both help
and harm, so the fitted delta has the capacity to move scores; the gate fails
because the matched source is not reliable enough to justify accepting.

## Interpretation

This branch is now weakened enough to stop widening on Mac-local linear ARC
candidate receivers.

The failure is not explained by source sketch quantization: sign-16, int8-16,
and unquantized float-16 all converge to the same protected no-op. The failure
is also not target-cache leakage: `zero_source`, shuffled source, same-norm
noise, train-mean source, target-derived source, label-shuffled, and
candidate-roll source all remain at the target-public floor.

The receiver design follows known set-model and selective-classification
principles, so the negative result is reviewer-useful but not a technical
contribution by itself. It sharpens the claim boundary: simple linear
source-public set compatibility is not enough to recover cross-model ARC
reasoning under strict source-private controls.

## Decision

Cut learned ARC candidate receivers as a current headline branch.

Keep the implementation as a falsification and future-baseline tool, but do
not run more small linear receiver variants unless there is a new mechanism
with a clear reason to beat the frozen target-public floor.

For the ICLR path, consolidate around the three stronger contributions:

1. fixed-byte source-private evidence packets with source-destroying controls;
2. positive public-basis packet methods, especially the ARC
   Fourier/anchor-syndrome row and OpenBookQA shared-basis row;
3. systems byte/exposure accounting and native-serving runbooks versus
   C2C/KVComm/KV-quantization state-transfer baselines.

Next exact gate: strengthen the existing positive Fourier/anchor-syndrome
packet row with cross-family validation if possible on Mac, and otherwise
prepare the native NVIDIA systems runbook plus COLM-ready figures/tables.
