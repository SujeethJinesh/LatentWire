# SVAMP70 Source Likelihood Sketch Gate

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; still missing a
   deployable positive communication method that survives disjoint controls.
2. Current paper story: Qwen2.5-Math -> Qwen3 has real SVAMP70 source/C2C
   headroom, but previous deployable rows collapse under target-only,
   shuffled-source, or holdout controls.
3. Exact blocker to submission: no source-derived low-rate method has yet
   improved over target/text baselines while passing source-destroying controls
   on live and disjoint holdout surfaces.
4. Current live branch: source likelihood sketch, a conditional
   syndrome-style sidecar over the existing target/text/source candidate pool.
5. Highest-priority gate: collect source-model continuation likelihoods on
   `svamp70_live_source`, freeze a live-CV acceptance rule, then evaluate once
   on `svamp70_holdout_source`.
6. Scale-up rung: strict-small tooling/gate implementation; scientific run
   blocked until the stuck MPS process is cleared.

## Motivation

Historical `rotalign`, `latent_bridge`, and results-folder rows showed useful
source/C2C headroom, but the recent audit killed the live query-memory and
Perceiver rows. The strongest remaining exact-ID surfaces are:

- `svamp70_live_source`: target `21/70`, source `13/70`, source-only `9`,
  oracle `30/70`
- `svamp70_holdout_source`: target `8/70`, source `8/70`, source-only `6`,
  oracle `14/70`

The new branch avoids another decoded source-text guard. It asks whether the
source model can communicate a compact preference over candidates that the
target already has available. This matches the side-information coding
interpretation in `references/465_source_likelihood_syndrome_sidecar_refs.md`.

## Implementation

Added:

- `scripts/collect_source_likelihood_sketch.py`
  - loads a source model
  - scores target/text/source candidate predictions as continuations of the
    source prompt
  - writes `candidate_scores` JSONL rows with source-model mean logprob,
    token count, and candidate correctness
- `scripts/analyze_svamp70_source_likelihood_sketch_gate.py`
  - converts candidate scores into a top-label plus quantized-margin sketch
  - fits a live-CV decision stump with an acceptance penalty
  - evaluates zero-source, shuffled-source, label-shuffle, target-only, and
    slots-only controls
  - freezes the full-live rule and evaluates holdout exactly once
- focused tests:
  - `tests/test_collect_source_likelihood_sketch.py`
  - `tests/test_analyze_svamp70_source_likelihood_sketch_gate.py`

## Pass Rule

Live CV must satisfy:

- matched correct `>=25/70`
- clean source-necessary IDs `>=3`
- clean control union `0`
- accepted target-correct harm `<=1`

Frozen holdout must satisfy:

- matched correct `>=10/70`
- clean source-necessary IDs `>=1`
- clean control union `0`
- accepted target-correct harm `<=1`

If the live gate passes but holdout fails, weaken this branch and do not scale
it upward without a new diagnostic. If both pass, move to SVAMP70 medium
confirmation with seeds and paired bootstrap uncertainty.

## Exact Commands

First confirm the MPS blocker is gone:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Proceed only if PID `31103` is absent or no longer a `scripts/calibrate.py`
process using `--device mps`.

Collect live sketches:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_source_likelihood_sketch.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --reference-label target \
  --candidate-text-field prediction \
  --prompt-mode direct \
  --source-use-chat-template \
  --source-enable-thinking false \
  --device mps \
  --dtype float32 \
  --output-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_sketch.jsonl \
  --output-md results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_sketch.md
```

Collect holdout sketches:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_source_likelihood_sketch.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --eval-file results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl \
  --candidate target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --candidate text=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --reference-label target \
  --candidate-text-field prediction \
  --prompt-mode direct \
  --source-use-chat-template \
  --source-enable-thinking false \
  --device mps \
  --dtype float32 \
  --output-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/holdout_sketch.jsonl \
  --output-md results/qwen25math_svamp70_source_likelihood_sketch_20260427/holdout_sketch.md
```

Analyze live/holdout:

```bash
./venv_arm64/bin/python scripts/analyze_svamp70_source_likelihood_sketch_gate.py \
  --live-sketch-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_sketch.jsonl \
  --live-candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --live-candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --live-candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --live-target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-sketch-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/holdout_sketch.jsonl \
  --holdout-candidate target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --holdout-candidate text=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --holdout-candidate source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --holdout-target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --max-sidecar-bits 8 \
  --outer-folds 5 \
  --accept-penalty 0.10 \
  --output-json results/qwen25math_svamp70_source_likelihood_sketch_20260427/sketch_gate.json \
  --output-md results/qwen25math_svamp70_source_likelihood_sketch_20260427/sketch_gate.md \
  --output-predictions-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/sketch_gate_predictions.jsonl
```

## Tests

Passed:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp70_source_likelihood_sketch_gate.py tests/test_collect_source_likelihood_sketch.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp70_source_likelihood_sketch_gate.py scripts/collect_source_likelihood_sketch.py
```

## Current Stop Condition

Scientific execution is blocked by the orphaned MPS process:

```text
PID 31103, PPID 1, STAT UE, scripts/calibrate.py ... --device mps --dtype float32 --seed 1
```

Earlier `SIGTERM` and `SIGKILL` did not terminate it. The exact next action is
to restart the machine or otherwise clear PID `31103`; do not launch additional
MPS jobs while it remains stuck.
