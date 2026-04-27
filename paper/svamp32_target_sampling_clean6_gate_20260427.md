# SVAMP32 Target-Only Clean6 Sampling Gate

Date: `2026-04-27`

## Cycle State

1. Current ICLR readiness: not ready; no deployable source-derived method
   survives answer masking and source-destroying controls.
2. Current paper story: target/no-source sampling can create target-side
   candidate reachability, but current source selectors either relay final
   answers or collapse to no accepted signal after answer masking.
3. Exact blocker: source-derived non-answer information must select newly
   reachable target candidates without source final/verified-answer leakage.
4. Current live branch: target-only/no-source candidate pools plus strictly
   answer-masked source selection; JEPA-style latent/process ranking only as
   the next design branch.
5. Highest-priority gate: SVAMP32 clean C2C residual IDs with target-only
   sampled candidates and full/answer-only/answer-masked selector controls.
6. Scale-up rung: strict small gate.

## Commands

Materialize the six clean C2C residual IDs:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --id-fields clean_residual_targets \
  --output-jsonl results/svamp32_target_sampling_clean6_20260427/clean6_eval.jsonl \
  --output-meta-json results/svamp32_target_sampling_clean6_20260427/clean6_eval.meta.json
```

Sample 16 no-source target candidates per example:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp32_target_sampling_clean6_20260427/clean6_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 16 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 31 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl \
  --output-json results/svamp32_target_sampling_clean6_20260427/target_only_samples.json \
  --output-md results/svamp32_target_sampling_clean6_20260427/target_only_samples.md
```

Extend the decoder-compatible target set with the sampled labels. The extender
now supports explicit IDs and a documented clean-residual override so this C2C
clean6 slice can reuse the loadable Qwen2.5-Math SVAMP32 source/target surface:

```bash
./venv_arm64/bin/python scripts/extend_target_set_candidate_labels.py \
  --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json \
  --ids 1d50b408c8f5cd2c 3e8a5691f5443495 47464cc0b064f172 575d7e83d84c1e67 6e9745b37ab6fc45 de1bf4d142544e5b \
  --override-clean-residual-ids 1d50b408c8f5cd2c 3e8a5691f5443495 47464cc0b064f172 575d7e83d84c1e67 6e9745b37ab6fc45 de1bf4d142544e5b \
  --candidate target_sample_s0=path=results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl,method=target_sample_s0 \
  ...
```

Materialize and evaluate source sidecars:

```bash
for mode in full answer_only answer_masked; do
  ./venv_arm64/bin/python scripts/materialize_svamp_source_candidate_sidecars.py \
    --live-target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json \
    --output-dir results/svamp32_target_sampling_clean6_20260427/source_candidate_sidecars_${mode} \
    --sidecar-bits 8 \
    --label-prior 0.0 \
    --profile-mode ${mode} \
    --date 2026-04-27

  ./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
    --target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json \
    --sidecar-jsonl results/svamp32_target_sampling_clean6_20260427/source_candidate_sidecars_${mode}/live_candidate_sidecars.jsonl \
    --min-confidence 2.0 \
    --min-source-necessary-clean 1 \
    --max-control-clean-union 0 \
    --max-accepted-harm 0 \
    --date 2026-04-27 \
    --output-json results/svamp32_target_sampling_clean6_20260427/top_selector_${mode}.json \
    --output-md results/svamp32_target_sampling_clean6_20260427/top_selector_${mode}.md
done
```

## Evidence

Target/no-source pool:

- clean C2C residual IDs: `6`
- samples per ID: `16`
- numeric coverage: `96/96`
- candidate-pool oracle: `2/6`
- reachable IDs: `3e8a5691f5443495`, `575d7e83d84c1e67`

Selector controls:

| Profile Mode | Matched Correct | Matched Accepted | Clean Correct | Source-Necessary Clean | Decision |
|---|---:|---:|---:|---:|---|
| `full` | 0/6 | 4 | 0 | 0 | fail |
| `answer_only` | 0/6 | 4 | 0 | 0 | fail |
| `answer_masked` | 0/6 | 0 | 0 | 0 | fail |

The generator passes the predefined strict-small floor (`>=2/6` clean gold in
the target-side pool). The source selector fails because no profile selects a
clean correct candidate, and answer-masked mode accepts nothing at the
confidence threshold.

## JEPA / Anti-Collapse Implication

The JEPA/LeJEPA/V-JEPA guidance remains useful, but only as the next non-numeric
design constraint. The local JEPA smoke already failed on SVAMP32
(`matched 10/32`, `target-only 14/32`, clean source-necessary `0`), and this
gate shows that numeric sidecars do not select the two newly reachable clean
candidates. A bounded next JEPA-style gate should operate only on the reachable
clean IDs with:

- answer-masked dual source views
- frozen target/candidate latent or KV summaries as stop-gradient targets
- matched-source ranking against zero-source, shuffled-source, answer-only,
  target-only, slots-only, and random same-byte sidecars
- no-harm target preservation
- variance floor, effective rank, covariance off-diagonal, Barlow/VICReg, and
  matched-vs-answer-only cosine telemetry

Do not tune this numeric source-candidate sidecar further.

## Decision

Strict small gate fails as a communication method. Promote only the
target/no-source candidate-pool generator as reusable headroom. The next exact
gate is an answer-masked process/latent ranking smoke on the two reachable
clean IDs, or a broader target/no-source candidate generator if that smoke
cannot use source information without answer leakage.
