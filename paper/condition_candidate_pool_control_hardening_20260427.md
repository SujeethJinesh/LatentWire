# Condition Candidate Pool Control Hardening

- date: `2026-04-27`
- status: `harness_hardened_not_method_evidence`

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; no live positive method
   survives source-destroying controls, answer-masking, holdout, and
   target-self preservation.
2. Current paper story: source-derived side information is still the most
   defensible direction, but current CPU sidecars over stored surfaces are
   pruned. The immediate value is hardening controls before the next MPS gate.
3. Exact blocker to submission: no evidence-bearing CPU-only gate remains, and
   PID `31103` is still an orphaned MPS process in `STAT=UE`.
4. Current live branch: none while MPS is blocked. Top candidate after cleanup
   is stronger-source answer-masked surface discovery, then erasure-aware
   learned syndrome or zero-init query bottleneck.
5. Highest-priority gate: make condition-specific candidate-pool controls
   truly source-destroying before future receiver scoring.
6. Scale-up rung: harness/source-control hardening before smoke.

## What Changed

The code audit found that `label_shuffle_offset` in
`scripts/build_condition_likelihood_candidate_pools.py` was recorded in the
manifest but not used. The `label_shuffle` condition placed the same-example
source output into the target-labeled slot, so it was a label-swap control but
not a non-self source-destroying label-shuffle control. The same code path could
also self-donor for `shuffled_source` if `--shuffle-offset` was `0` or a
multiple of the number of examples.

Implemented:

- `_nonself_offset_index(total, index, offset)` helper.
- `shuffled_source` now uses a guaranteed non-self donor.
- `label_shuffle` now uses `--label-shuffle-offset` for the target-labeled
  source donor while keeping target content in the source-labeled slot.
- Regression coverage for zero offsets and direct donor-ID checks.

## Evidence Update

This is not positive-method evidence. It is provenance hardening for future
condition-specific likelihood/receiver gates. Existing generated
`condition_candidate_pools` artifacts remain interpretable as older artifacts;
new runs should regenerate them after this patch if used in a future claim.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_condition_likelihood_candidate_pools.py \
  tests/test_analyze_condition_likelihood_receiver_gate.py \
  tests/test_kvcomm_eval_controls.py -q
```

Result: `18 passed in 0.12s`.

```bash
./venv_arm64/bin/python -m py_compile scripts/build_condition_likelihood_candidate_pools.py
```

Result: passed.

## Decision

- hardened: condition-specific candidate-pool control builder.
- no method promoted.
- next exact gate remains:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, regenerate future condition candidate pools with this
patch before scoring any receiver gate. If it remains in `STAT=UE`, OS/session
cleanup is still required before evidence-bearing MPS work.
