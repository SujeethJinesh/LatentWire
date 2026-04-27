# SVAMP Semantic Predicate Decoder

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; still missing one
   stable positive method plus medium/large controls, seed stability, C2C/text
   comparisons, systems metrics, and cross-family falsification.
2. Current paper story: source-side side information can be source-specific,
   but the current deployable decoders either damage target-correct examples or
   fail holdout transfer.
3. Exact blocker to submission: no target-preserving method recovers clean
   source-necessary IDs on both live and holdout; MPS is still blocked by PID
   `31103` in `STAT=UE`.
4. Current live branch: none. Candidate branch tested here: learned semantic
   predicates over target/source/text candidate pools with erasure-aware
   abstention.
5. Highest-priority gate: CPU-only live/holdout smoke with source-destroying
   controls and zero accepted harm.
6. Scale-up rung: smoke / branch falsification.

## Method

Added `scripts/analyze_svamp_source_semantic_predicate_decoder.py`, a CPU-only
artifact analyzer that:

- builds numeric candidate pools from existing target/source/text generations;
- extracts source predicate bits from generated source reasoning text:
  verified equations, operation cues, final-answer markers, numeric windows,
  and pairwise arithmetic closures;
- learns fold-local Laplace log-odds weights over candidate predicate
  compatibility;
- applies an erasure rule, preserving the target fallback unless the best
  source-supported candidate clears score and margin thresholds;
- evaluates `matched`, `zero_source`, `shuffled_source`, `label_shuffle`,
  `target_only`, and `slots_only` controls.

This is a bounded CPU probe, not a headline method.

## Primary Sources / Design Rationale

Recorded in `references/468_target_preserving_receiver_gate_refs.md`:

- Speculative sampling / decoding motivates receiver-side acceptance rather
  than direct source overwrite.
- DeepJSCC-WZ and Wyner-Ziv-style side information motivate conditional
  decoding with erasure.
- CRANE and constrained decoding motivate constraining final-answer selection
  without constraining the entire reasoning trace.
- Conformal abstention motivates explicit no-inject decisions.
- Self-consistency motivates candidate-pool semantic agreement as a diagnostic,
  not a claim by itself.

## Main Gate

Command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
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
  --output-dir results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427 \
  --output-predictions-jsonl results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/predictions.jsonl
```

Result:

- Status: `semantic_predicate_decoder_fails_smoke`.
- Live: `25/70`, accepted `5`, clean source-necessary `3`, accepted harm `0`,
  control clean union `0`.
- Holdout: `9/70`, accepted `2`, clean source-necessary `0`, accepted harm `0`,
  control clean union `0`.

Decision: do not promote. The strict gate gives target-safe live recovery but
does not transfer to holdout. This prunes generated-source-trace semantic
predicate decoding on the current Qwen2.5-Math -> Qwen3 SVAMP artifacts unless
a stronger source surface or model-collected receiver uncertainty features
change the hypothesis.

## Scratch Target-Likelihood Smoke

A CPU-only target receiver scoring smoke was also run on 8 live examples using
`Qwen/Qwen3-0.6B` on CPU. It completed, so target-side CPU likelihood scoring
is feasible, but the initial setup is not a positive method:

- including C2C as an internal candidate contaminates the receiver gate because
  long C2C text receives high continuation likelihood even when wrong;
- excluding C2C makes the target scorer often prefer source candidates, but the
  8-example slice includes several wrong source answers.

Use C2C as an external baseline, not an internal candidate for this method.

## Next Gate

No further CPU artifact mining is promoted from this branch. Recheck:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout recorded in
`paper/postkill_historical_cpu_audit_20260427.md`. If it remains present, the
hard blocker is OS/session-level cleanup or reboot before MPS experiments.
