# Qwen2.5-7B SVAMP70 Surface And Query-Bottleneck Smoke

- date: `2026-04-27`
- readiness: not ICLR-ready
- rung: smoke / source-surface discovery
- live branch: none

## Cycle Start

1. Current ICLR readiness: not ready; still missing a source-derived method that
   survives answer-only and source-destroying controls.
2. Current paper story: receiver likelihood on the 7B disagreement slice was
   pruned because matched equals answer-only; JEPA-style objectives are design
   constraints, not evidence.
3. Exact blocker: find answer-unexplained source headroom in a target-side pool
   or show answer-free source latents can predict useful side information.
4. Candidate branches: fresh stronger-source surface discovery; answer-free
   query-bottleneck syndrome diagnostic.
5. Highest-priority gate: run the cheapest MPS-clear stronger-source scout, then
   a CPU answer-free latent diagnostic if the surface still fails.
6. Scale-up rung: smoke.

## Fresh Cached-7B SVAMP70 Scout

I did not run `Qwen/Qwen2.5-Math-7B-Instruct` because it was not cached locally.
The MPS-clear scout used cached `Qwen/Qwen2.5-7B-Instruct` to avoid turning this
cycle into a model-download gate.

Command:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25_7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Result:

- target: `21/70`
- source: `15/70`
- text relay: `12/70`
- numeric coverage: `70/70` for all three methods
- exact ID parity: true
- source-only over target: `8`
- clean source-only after text relay: `7`
- target/source oracle: `29/70`

Source-set and answer-masking commands:

```bash
./venv_arm64/bin/python scripts/build_source_contrastive_target_set.py \
  --target target=path=results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_alone.jsonl,method=source_alone \
  --control text=path=results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/text_to_text.jsonl,method=text_to_text \
  --min-source-only 1 \
  --date 2026-04-27 \
  --output-json results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json \
  --output-md results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.md

./venv_arm64/bin/python scripts/audit_source_surface_answer_masking.py \
  --results-root results/qwen25_7b_qwen3_svamp70_surface_scout_20260427 \
  --date 2026-04-27 \
  --output-json results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/answer_masking_audit.json \
  --output-md results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/answer_masking_audit.md
```

Answer-masking result:

- clean source-only: `7`
- clean in target-side pool: `3`
- answer-unexplained clean in target-side pool: `0`
- clean in-pool IDs: `4c84ebf42812703b`, `d64f6e35083ffe8c`,
  `de1bf4d142544e5b`

Decision: fail for positive-method promotion. The fresh surface has stronger
raw source headroom than the selected 7B disagreement slices, but every
reachable clean answer is explained by source final or verified numeric values.

## Answer-Free Query-Bottleneck Syndrome Smoke

Planner rationale: if no stored or fresh surface has answer-unexplained
target-pool headroom, test whether answer-free source hidden states can predict
the previously observed C2C candidate-syndrome bound under strict controls.

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_latent_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --moduli 2,3,5,7 \
  --probe-model query_bottleneck \
  --query-slots 8 \
  --query-epochs 80 \
  --query-lr 0.01 \
  --query-weight-decay 0.001 \
  --query-seed 0 \
  --shuffle-offset 1 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --feature-layers mid,last \
  --device cpu \
  --dtype float32 \
  --date 2026-04-27 \
  --output-json results/svamp32_query_bottleneck_syndrome_jepa_smoke_20260427/query_bottleneck_mid_last.json \
  --output-md results/svamp32_query_bottleneck_syndrome_jepa_smoke_20260427/query_bottleneck_mid_last.md
```

Result:

- status: `source_latent_syndrome_probe_fails_gate`
- matched: `10/32`
- zero-source: `13/32`
- shuffled-source: `10/32`
- label-shuffled: `12/32`
- target-only: `14/32`
- slots-only: `8/32`
- clean source-necessary IDs: `0`
- control clean union: `0`
- teacher numeric coverage: `32/32`
- candidate-pool clean-gold count: `2`

Decision: fail. Answer-free prompt hidden states with an 8-query bottleneck do
not predict useful C2C residue side information on this SVAMP32 bound; matched
falls below the target-only floor and recovers no clean source-necessary IDs.

## Overall Decision

- pruned: cached-7B full SVAMP70 surface as a source for answer-masked
  target-pool communication.
- pruned: answer-free query-bottleneck syndrome diagnostic as the immediate
  JEPA precursor.
- still alive only as design: JEPA-style source-memory connector with frozen
  target latents, preservation loss, answer-masked source views, and collapse
  telemetry, but it needs either a stronger surface or a different supervision
  target.

## Next Exact Gate

Run the originally recorded Math-7B source-surface scout once the model is
available locally, or run a selected-disagreement Math-7B slice if download time
is acceptable:

- require exact ID parity and numeric coverage `>= 69/70`
- require source-only over target `>= 6/70`
- require answer-unexplained clean-in-pool `> 0`
- reject immediately if clean reachable IDs are explained by source final or
  verified numeric answers

Do not run another normalized-answer receiver or answer-free syndrome probe on
these same surfaces without a new diagnostic reason.
