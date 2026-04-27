# Condition-Specific Likelihood Receiver Harness

- date: `2026-04-27`
- status: `implemented_tested_not_yet_run_on_full_controls`
- scale-up rung: harness / smoke preparation

## Purpose

The target-likelihood receiver idea cannot be evaluated fairly by the older
likelihood sketch analyzer because that analyzer shuffles sketches and forces
some controls to target fallback. The new harness requires each condition to
provide its own receiver-scored sketch, so source-destroying controls mutate the
candidate pool before target-side likelihood scoring.

## Implementation

- Added `scripts/analyze_condition_likelihood_receiver_gate.py`.
- Added `tests/test_analyze_condition_likelihood_receiver_gate.py`.

The analyzer accepts condition-specific JSONL sketches:

- `matched`
- `zero_source`
- `shuffled_source`
- `label_shuffle`
- `target_only`
- `slots_only`

It fits the receiver gate only on live matched sketches, then applies the same
frozen rule to every condition and to holdout. Clean source-necessary IDs are
computed as matched clean IDs minus the clean union recovered by all controls.

## Verification

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_condition_likelihood_receiver_gate.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py tests/test_collect_source_likelihood_sketch.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_condition_likelihood_receiver_gate.py
```

Result: `12 passed in 0.12s`; compile passed.

## Next Gate

Collect receiver-scored condition sketches on CPU or MPS, then run:

```bash
./venv_arm64/bin/python scripts/analyze_condition_likelihood_receiver_gate.py \
  --live-condition-sketch matched=results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl \
  --live-condition-sketch zero_source=<live_zero_source_sketch.jsonl> \
  --live-condition-sketch shuffled_source=<live_shuffled_source_sketch.jsonl> \
  --live-condition-sketch label_shuffle=<live_label_shuffle_sketch.jsonl> \
  --live-condition-sketch target_only=<live_target_only_sketch.jsonl> \
  --live-condition-sketch slots_only=<live_slots_only_sketch.jsonl> \
  --live-target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-condition-sketch matched=<holdout_matched_sketch.jsonl> \
  --holdout-condition-sketch zero_source=<holdout_zero_source_sketch.jsonl> \
  --holdout-condition-sketch shuffled_source=<holdout_shuffled_source_sketch.jsonl> \
  --holdout-condition-sketch label_shuffle=<holdout_label_shuffle_sketch.jsonl> \
  --holdout-condition-sketch target_only=<holdout_target_only_sketch.jsonl> \
  --holdout-condition-sketch slots_only=<holdout_slots_only_sketch.jsonl> \
  --holdout-target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --outer-folds 5 \
  --accept-penalty 0.10 \
  --max-sidecar-bits 8 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 3 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-clean-control-union 0 \
  --max-accepted-harm 1 \
  --date 2026-04-27 \
  --output-json results/condition_likelihood_receiver_gate_20260427/gate.json \
  --output-md results/condition_likelihood_receiver_gate_20260427/gate.md \
  --output-predictions-jsonl results/condition_likelihood_receiver_gate_20260427/predictions.jsonl
```

If PID `31103` remains stuck, collect these sketches on CPU only and stop early
if live no-harm thresholds still cannot reach the gate.
