# Semantic Predicate Sidecar Harness

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; still missing a
   deployable positive method with live/holdout controls, seed stability,
   systems accounting, and cross-family falsification.
2. Current paper story: byte-efficient source side information remains the top
   method direction, but current SVAMP semantic predicates are falsification
   and harness work rather than positive evidence.
3. Exact blocker to submission: no learned source-derived sidecar has cleared
   holdout clean source-necessary gains under source-destroying controls;
   MPS remains blocked by PID `31103`.
4. Current live branch or top candidates: no live branch. Top candidate remains
   learned source-derived syndrome/innovation sidecar over target candidate or
   cache side information.
5. Highest-priority gate: make the sidecar gate executable and harden controls
   while MPS is blocked.
6. Scale-up rung: CPU smoke / harness preparation.

## Code Change

Updated `scripts/analyze_svamp_source_semantic_predicate_decoder.py` to support:

- hash-derived non-self `shuffled_source` and `label_shuffle` controls
- `random_sidecar` same-byte source-destroying control
- source-control provenance fields:
  `condition_source_example_id`, `condition_source_final`,
  `source_control_source_answers_overlap_target`
- optional learned sidecar JSONL inputs:
  `--live-sidecar-jsonl`, `--holdout-sidecar-jsonl`, and
  `--sidecar-format candidate_scores`

Expected learned sidecar JSONL schema:

```json
{
  "example_id": "...",
  "candidate_scores": [
    {"label": "target", "score": 0.1},
    {"label": "source", "score": 1.7}
  ],
  "confidence": 1.6,
  "sidecar_bits": 8
}
```

The adapter converts candidate-score sidecars into profile features such as
`sidecar_present`, `sidecar_top_label:*`, `candidate_eq_sidecar_top`, score,
margin, and confidence buckets. `zero_source` gets no sidecar;
`shuffled_source` and `label_shuffle` use hash non-self sidecar/source pairing;
`random_sidecar` keeps the byte budget while destroying source semantics.

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
  --output-dir .debug/semantic_predicate_decoder_sidecar_harness_20260427 \
  --output-predictions-jsonl .debug/semantic_predicate_decoder_sidecar_harness_20260427/predictions.jsonl
```

Result: `semantic_predicate_decoder_fails_smoke`.

Live surface:

- matched: `25/70`, clean source-necessary `3`, accepted `5`, accepted harm `0`
- random same-byte sidecar: `17/70`, clean source-necessary `1`, accepted `27`,
  accepted harm `7`
- zero-source: `21/70`, clean source-necessary `0`
- shuffled-source: `21/70`, clean source-necessary `0`
- target-only: `21/70`, clean source-necessary `0`
- slots-only: `22/70`, clean source-necessary `0`
- label-shuffle: `21/70`, clean source-necessary `0`
- control clean union: `1`

Holdout surface:

- matched: `9/70`, clean source-necessary `0`, accepted `2`, accepted harm `0`
- random same-byte sidecar: `9/70`, clean source-necessary `0`, accepted `10`,
  accepted harm `1`
- zero-source: `8/70`, clean source-necessary `0`
- shuffled-source: `8/70`, clean source-necessary `0`
- target-only: `9/70`, clean source-necessary `0`
- slots-only: `13/70`, clean source-necessary `0`
- label-shuffle: `8/70`, clean source-necessary `0`
- control clean union: `0`

Artifact hashes:

- `.debug/semantic_predicate_decoder_sidecar_harness_20260427/semantic_predicate_decoder.json`:
  `9cc4804426b6eb1b0f47f7f3fb091cb9185b763379e9fc7c94d08ee936591ed0`
- `.debug/semantic_predicate_decoder_sidecar_harness_20260427/semantic_predicate_decoder.md`:
  `ee8210a6d6029bf2de2594de2b70db6a48aa6e0075eb4d58214274b1e10e9144`
- `.debug/semantic_predicate_decoder_sidecar_harness_20260427/predictions.jsonl`:
  `2fa404726a3cb654ec2de6cba75cfceac3e20fccf7ca2b5b873a687020aa6aed`

## Decision

This is tooling evidence and a stronger kill note, not positive method
evidence. The old semantic-predicate branch now fails for two independent
reasons:

- holdout matched clean source-necessary recovery is `0`
- random same-byte sidecar recovers a live clean ID and causes target-self harm

Do not tune this branch further on current artifacts. Use the hardened harness
only to evaluate a genuinely new learned sidecar produced from a stronger
source surface or from frozen out-of-fold model-side signals.

## Next Exact Gate

First check:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run the stronger-source MPS surface scout from
`paper/postkill_historical_cpu_audit_20260427.md`. If that scout clears
source-mass and exact-ID gates, produce frozen out-of-fold candidate-score
sidecars and evaluate them with:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/.../live/source_contrastive_target_set.json \
  --holdout-target-set results/.../holdout/source_contrastive_target_set.json \
  --live-sidecar-jsonl results/.../learned_syndrome_live.jsonl \
  --holdout-sidecar-jsonl results/.../learned_syndrome_holdout.jsonl \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.25 \
  --harm-weight 4.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 3 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --output-dir results/.../learned_syndrome_sidecar_gate \
  --output-predictions-jsonl results/.../learned_syndrome_sidecar_gate/predictions.jsonl
```
