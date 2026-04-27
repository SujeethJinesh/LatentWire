# Qwen2.5-Math-7B Disagreement12 Surface Scout

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready; no source-derived positive method survives
   answer masking and source-destroying controls.
2. Current paper story: stronger sources can expose raw source/target
   disagreement, but stored wins keep reducing to final-answer relay or
   target-side candidate artifacts.
3. Exact blocker: a source surface must have
   `answer_unexplained_clean_in_pool > 0` before another receiver, connector,
   or JEPA-style latent objective is worth promoting.
4. Current live branches: `none`; top candidates are Math-7B source-surface
   discovery and JEPA-style answer-masked latent prediction as a design
   constraint after non-leaky headroom exists.
5. Highest-priority gate: selected-disagreement Math-7B source-surface smoke.
6. Scale-up rung: selected micro discovery.

## JEPA / LeJEPA / V-JEPA Committee Update

The JEPA subagent returned the same discipline as the latest reference memo:
JEPA-style predictive latent objectives should be used as an anti-collapse and
answer-masking harness constraint, not as evidence by themselves. The concrete
design implications are:

- use answer-masked dual source views
- predict frozen target latent/KV summaries rather than final tokens
- add matched-source margins over zero-source, shuffled-source, target-only, and
  slots-only controls
- preserve target-correct IDs with a no-harm loss or abstention gate
- log sideinfo variance floor, effective rank, covariance off-diagonal mass,
  matched-vs-answer-only cosine, and Barlow/VICReg-style collapse telemetry

This does not change the next experiment until a surface exists with
answer-unexplained clean target-pool headroom.

## Surface Gate

Command:

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
```

The first fetch attempt stalled visibly at `Fetching 4 files: 0%`. The retry
with `HF_HUB_DISABLE_XET=1` downloaded the model into the repo-local
`.hf_home/hub` cache and completed the run.

Result:

- target: `0/12`
- source: `5/12`
- text relay: `1/12`
- exact ID parity: true for all methods
- numeric extraction coverage: `12/12` for all methods
- source-only over target: `5`
- clean source-only after text relay: `5`
- target/source oracle: `5/12`

Target-set/audit commands:

```bash
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

Answer-masking audit:

- clean in target-side pool: `3`
- answer-unexplained clean in target-side pool: `0`
- clean in-pool IDs: `561daa750422c0e4`, `ab1e71e8928661d0`,
  `daea537474de16ac`

## Decision

Fail for promotion. The selected Math-7B source is a stronger disagreement
generator than the previous selected 7B run, but it still does not create a
non-leaky receiver target. Do not train a learned receiver or JEPA-style
connector on this selected surface.

## Next Gate

Run the full SVAMP70 Math-7B scout now that the model is local only if we are
willing to spend the MPS time. Promotion still requires
`answer_unexplained_clean_in_pool > 0`; otherwise reject before receiver work.
If full SVAMP70 also fails, stop source-scorer/receiver variants and switch to a
new candidate-pool generator that creates target-side candidates without source
final-answer leakage.
