# Semantic Sidecar Target-Pool Hardening

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; still missing a
   positive method with strict controls, target-self preservation, seed
   stability, systems metrics, and cross-family falsification.
2. Current paper story: byte-efficient source side-information remains the top
   branch, but the decoder must operate over target-side candidates rather than
   silently exposing source answers.
3. Exact blocker to submission: no frozen learned sidecar has cleared a
   target-side candidate gate; MPS is still blocked by PID `31103`.
4. Current live branch or top candidates: no live branch. Top candidate remains
   learned source-derived syndrome/innovation sidecar.
5. Highest-priority gate: close reviewer holes in the sidecar harness.
6. Scale-up rung: CPU smoke / harness hardening.

## Reviewer Hole Closed

The sidecar decoder previously built candidate pools from every loaded
artifact, including `source`. That made source-only values available to
`zero_source`, `target_only`, `slots_only`, and `random_sidecar` controls. It
also meant generic sidecar labels such as `source` could map to the current
example's source answer even under a shuffled-source sidecar.

Code changes:

- `_candidate_pool()` is now target-side by default and excludes `source`.
- Candidate-score sidecars can map an explicit `value`/`candidate_value` only if
  that value already appears in the target-side candidate pool.
- `random_sidecar` now preserves declared `sidecar_bits` and uses candidate-
  score-style features when learned sidecar JSONL is present.
- `target_only_sidecar` and `slots_only_sidecar` now test sidecar-shaped
  target/slot controls at the same byte scale.
- Sidecar JSONL loading now rejects duplicate IDs and fails if supplied sidecar
  IDs do not exactly match the target-set `reference_ids`.
- Summaries now include accepted help, fallback-correct count, accepted clean
  source help, and sidecar present/missing counts.

New tests:

- target-only candidate pools cannot recover source-only values.
- duplicate sidecar IDs fail.
- missing sidecar IDs fail.
- candidate-score sidecar and random control preserve the same byte budget.

## CPU Replay

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir .debug/semantic_predicate_decoder_targetpool_20260427 \
  --output-predictions-jsonl .debug/semantic_predicate_decoder_targetpool_20260427/predictions.jsonl
```

Result: `semantic_predicate_decoder_fails_smoke`.

Live:

- matched: `24/70`, clean `3`, accepted clean help `3`, accepted harm `1`
- random same-byte sidecar: `16/70`, clean `0`, accepted harm `9`
- target-only sidecar: `21/70`, clean `0`, accepted harm `0`
- slots-only sidecar: `21/70`, clean `0`, accepted harm `0`
- zero-source: `21/70`, clean `0`
- shuffled-source: `21/70`, clean `0`
- target-only: `21/70`, clean `0`
- slots-only: `22/70`, clean `0`
- control clean union: `0`

Holdout:

- matched: `9/70`, clean `0`, accepted harm `0`
- random same-byte sidecar: `12/70`, clean `0`, accepted harm `0`
- target-only sidecar: `8/70`, clean `0`, accepted harm `0`
- slots-only sidecar: `11/70`, clean `0`, accepted harm `0`
- zero-source: `8/70`, clean `0`
- shuffled-source: `8/70`, clean `0`
- target-only: `8/70`, clean `0`
- slots-only: `15/70`, clean `0`
- control clean union: `0`

Artifact hashes:

- `.debug/semantic_predicate_decoder_targetpool_20260427/semantic_predicate_decoder.json`:
  `55a1e73e061f03c51733d16009e7d1f6766d2d2f8807ad725df1d0cd47020d5f`
- `.debug/semantic_predicate_decoder_targetpool_20260427/semantic_predicate_decoder.md`:
  `7d5882c76b0d9e6e5dbae90084bd731945a8d72cfa9b948843dc290a502365bb`
- `.debug/semantic_predicate_decoder_targetpool_20260427/predictions.jsonl`:
  `48ba362b0f8f557ceb5c1eedd4674d097af1a464f4ea9cdfe1bdf2475fb7fdd8`

Updated sidecar-shaped control replay:

- `.debug/semantic_predicate_decoder_sidecar_controls_20260427/semantic_predicate_decoder.json`:
  `f1529702d17ae53eb9e5b1ad40e2d274a0c3724d0158c70c6c5c1408353691a2`
- `.debug/semantic_predicate_decoder_sidecar_controls_20260427/semantic_predicate_decoder.md`:
  `7d5882c76b0d9e6e5dbae90084bd731945a8d72cfa9b948843dc290a502365bb`
- `.debug/semantic_predicate_decoder_sidecar_controls_20260427/predictions.jsonl`:
  `48ba362b0f8f557ceb5c1eedd4674d097af1a464f4ea9cdfe1bdf2475fb7fdd8`

## Decision

This is a stronger kill of the old semantic-predicate branch. Once source-only
values are removed from the decoder pool, the live row no longer clears the
configured accuracy/harm gate. The branch remains useful only as a hardened
evaluator for future learned sidecars.

## Next Exact Gate

First check:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run the stronger-source MPS surface scout from
`paper/postkill_historical_cpu_audit_20260427.md`. If it clears source-mass
thresholds, generate frozen out-of-fold candidate-score sidecars and evaluate
them with the hardened target-pool sidecar gate.
