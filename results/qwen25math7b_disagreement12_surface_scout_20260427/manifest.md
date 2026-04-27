# Qwen2.5-Math-7B Disagreement12 Surface Scout

- date: `2026-04-27`
- status: `fails_answer_unexplained_gate`
- scale rung: selected micro discovery
- source model: `Qwen/Qwen2.5-Math-7B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- eval file: `results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl`
- materialized eval: `results/qwen25math7b_disagreement12_surface_scout_20260427/_artifacts/disagreement12_eval_12.jsonl`
- device: `mps`
- max new tokens: `64`
- git base during audit: `68aae9003216437ec128bdfe36a672631f2bf420`

## Result

| Method | Correct | Accuracy | Numeric Coverage | Exact ID Parity |
|---|---:|---:|---:|---|
| `target_alone` | 0/12 | 0.0000 | 12/12 | true |
| `source_alone` | 5/12 | 0.4167 | 12/12 | true |
| `text_to_text` | 1/12 | 0.0833 | 12/12 | true |

Source-only over target: `5/12`.

Clean source-only after text relay: `5/12`.

Clean in target-side pool: `3`.

Answer-unexplained clean in target-side pool: `0`.

## Decision

Reject for positive-method promotion. The math-specialized 7B source creates a
stronger raw disagreement surface than the prior selected 7B run, but every
reachable clean target-pool answer is still explained by source final or
verified numeric answers. This is source-surface evidence, not communication
evidence.

## Commands

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl \
  --results-dir results/qwen25math7b_disagreement12_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 12 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error

./venv_arm64/bin/python scripts/build_source_contrastive_target_set.py \
  --target target=path=results/qwen25math7b_disagreement12_surface_scout_20260427/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math7b_disagreement12_surface_scout_20260427/source_alone.jsonl,method=source_alone \
  --control text=path=results/qwen25math7b_disagreement12_surface_scout_20260427/text_to_text.jsonl,method=text_to_text \
  --min-source-only 1 \
  --date 2026-04-27 \
  --output-json results/qwen25math7b_disagreement12_surface_scout_20260427/source_contrastive_target_set.json \
  --output-md results/qwen25math7b_disagreement12_surface_scout_20260427/source_contrastive_target_set.md

./venv_arm64/bin/python scripts/audit_source_surface_answer_masking.py \
  --results-root results/qwen25math7b_disagreement12_surface_scout_20260427 \
  --date 2026-04-27 \
  --output-json results/qwen25math7b_disagreement12_surface_scout_20260427/answer_masking_audit.json \
  --output-md results/qwen25math7b_disagreement12_surface_scout_20260427/answer_masking_audit.md
```

## Artifacts

- `manifest.json`
- `source_alone.jsonl`
- `target_alone.jsonl`
- `text_to_text.jsonl`
- `source_contrastive_target_set.json`
- `source_contrastive_target_set.md`
- `answer_masking_audit.json`
- `answer_masking_audit.md`
- `logs/source.log`
- `logs/target.log`
- `logs/t2t.log`
