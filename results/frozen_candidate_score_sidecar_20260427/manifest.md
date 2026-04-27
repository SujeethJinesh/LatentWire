# Frozen Candidate Score Sidecar Results

- date: `2026-04-27`
- status: `frozen_candidate_score_sidecar_live_failed`
- branch: frozen source-model-scored sidecar over target-side candidate values
- scale-up rung: `strict-small live smoke`

## Live Collection

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_svamp_frozen_candidate_score_sidecar.py \
  --scorer-model Qwen/Qwen2.5-Math-1.5B \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --continuation-template 'Answer: {text}' \
  --device cpu \
  --dtype float32 \
  --sidecar-bits 32 \
  --scorer-use-chat-template \
  --scorer-enable-thinking false \
  --resume \
  --date 2026-04-27 \
  --output-jsonl results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl \
  --output-md results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.md
```

Result:

- rows: `70`
- elapsed: `351.12s`
- candidate pool: `target_side_only`
- sidecar bits: `32`
- top labels: `target=44`, `t2t=26`

## Live Gate

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --live-sidecar-jsonl results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl \
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
  --output-dir results/frozen_candidate_score_sidecar_20260427/live_only_decoder_gate \
  --output-predictions-jsonl results/frozen_candidate_score_sidecar_20260427/live_only_decoder_gate/predictions.jsonl
```

Result:

- status: `semantic_predicate_decoder_fails_smoke`
- live matched correct: `21/70`
- live accepted: `1`
- live clean source-necessary: `0`
- live accepted harm: `0`
- live control clean union: `0`

The holdout section in this gate is not a model-scored holdout run because no
holdout sidecar was passed. It is included only because the decoder CLI requires
a holdout target set.

## Decision

Kill this frozen target-side candidate-score sidecar producer on the canonical
live SVAMP70 surface. It is cleaner than the older source-likelihood sketch,
but it recovers no clean source-necessary examples and does not improve target
accuracy.

Next branch should not be another threshold sweep on this target-side candidate
pool. The useful next move is source-surface discovery or a qualitatively new
sidecar with a larger but controlled candidate surface.

## Hashes

- `live_candidate_score_sidecar_cpu.jsonl`:
  `3734e4884c87bc14d3bc74317a47c195bbac85253927ae799e3eaa717cf2e771`
- `live_candidate_score_sidecar_cpu.md`:
  `4bff722fcda7579918d95b01f4b01471e52aae53f3b06236b786913054202cdc`
- `live_only_decoder_gate/semantic_predicate_decoder.json`:
  `7491d8c4e6e63d088ae208e1e01496b8712855940a7c5df8fccd246e3fbcd498`
- `live_only_decoder_gate/semantic_predicate_decoder.md`:
  `0a56cd0c02c7ad1a54dbf68e1fdf1d0d791b7a2c587b0916b87ba529d29bf28d`
- `live_only_decoder_gate/predictions.jsonl`:
  `90421fa78c019d7b2fa927bce3f3719bd083792383b585c0a571186107ac521c`
