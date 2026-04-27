# Math-7B SVAMP70 Surface And Target-Only Sampling

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready; no source-derived positive method
   survives answer masking and source-destroying controls.
2. Current paper story: stronger sources expose some disagreement, but useful
   wins collapse to final-answer relay unless the target-side pool is made
   reachable without source leakage.
3. Exact blocker: no surface has `answer_unexplained_clean_in_pool > 0`.
4. Current live branch: target-only/no-source candidate-pool generation on
   residual clean IDs; source selectors are not live unless answer-masked
   source evidence survives controls.
5. Highest-priority gate: full SVAMP70 Math-7B source-surface scout, followed
   by target-only sampling on residual clean IDs if the scout fails.
6. Scale-up rung: medium surface discovery plus micro candidate-pool smoke.

## Full SVAMP70 Math-7B Scout

Command:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
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
- source: `5/70`
- text relay: `8/70`
- exact ID parity: true for all methods
- numeric coverage: `70/70` for all methods
- source-only over target: `3`
- clean source-only after text relay: `3`
- target/source oracle: `24/70`

Answer-masking audit:

- clean IDs: `3`
- clean in target-side pool: `1`
- answer-unexplained clean in target-side pool: `0`
- clean in-pool ID: `a07cd6cc8f1c832e`

Decision: fail. Math-7B is weaker than target-alone on this frozen SVAMP70
surface and does not expose non-leaky target-pool headroom. Do not run another
source-scorer or receiver on this surface.

## Target-Only Candidate-Pool Smoke

Residual clean IDs from the Math-7B scout:

- `14bfbfc94f2c2e7b`
- `a07cd6cc8f1c832e`
- `d64f6e35083ffe8c`

Command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file data/svamp_eval_70.jsonl \
  --ids 14bfbfc94f2c2e7b a07cd6cc8f1c832e d64f6e35083ffe8c \
  --output-jsonl results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.jsonl \
  --output-meta-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.meta.json

HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 16 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 17 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl \
  --output-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.json \
  --output-md results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.md
```

Result:

- target-only samples alone: candidate oracle `1/3`
- combined sampled target-side pool: clean in pool `2/3`
- clean IDs in combined pool: `14bfbfc94f2c2e7b`,
  `a07cd6cc8f1c832e`

Decision: generator-positive but not communication. Target-only/no-source
sampling improves candidate reachability from `1/3` to `2/3`, so the next
method should work on candidate-pool generation and answer-masked selection.

## Answer-Masked Selector Control

Code update: `scripts/materialize_svamp_source_candidate_sidecars.py` now has
`--profile-mode full|answer_only|answer_masked`.

Selector result on the sampled pool:

| Profile Mode | Matched Clean Correct | Source-Necessary Clean | Decision |
|---|---:|---:|---|
| `full` | 2/3 | 2 | answer relay |
| `answer_only` | 2/3 | 2 | answer relay |
| `answer_masked` | 0/3 | 0 | fails |

Decision: prune the current source-candidate score selector as communication.
The positive-looking selector is exactly reproduced by answer-only source
information and disappears when final/verified answer values are masked.

## Next Gate

Do not train a connector on these selectors. The next exact gate should create
a larger target-only/no-source sampled candidate pool on a stricter slice
(`SVAMP32` or all target-wrong/source-disagreement IDs), then test only
answer-masked source signals against answer-only, shuffled-source, random
sidecar, target-only, and slots-only controls.
