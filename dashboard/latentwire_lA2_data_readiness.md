# LatentWire L-A2 Data Readiness

- status: `PARKED_NEEDS_STRONGER_GENERATOR`
- evidence surface: v3 materialized a 100-prompt x16 candidate pool with source, target, and verifier scores, but the pool is too weak to screen L-A2.
- corrected v3 artifact: `results/overnight_v3/20260604_corrected_reprobe/exp4_corrected_l_a2_rerank_ceiling/summary.json`.
- corrected v3 result: `1600` verifier-scored candidates, `36` prompts with at least one correct candidate, required threshold `80`; verdict `PARKED_NEEDS_STRONGER_GENERATOR`.
- existing smoke: `results/mac_continue/fresh_mmlu_pro/rerank_generation_rows.jsonl`, 3 generated rows with `row_id`, `split`, `answer_index`, and `generated_text` only.
- usable source/target MC score surfaces: `results/mac_continue/fresh_mmlu_pro/fresh_mmlu_pro_rows.jsonl`, but these are option-score rows, not generated-solution rerank pools.
- conclusion: L-A2 is not evidence yet; it is a stronger-generator cache/backfill job.
- receiver-conditioned rerank probe: `PARKED_POOL_TOO_WEAK`; do not estimate a gain until the correct-candidate subset has at least about `80` prompts.

## CPU Smoke Command

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
HF_HOME=.hf_home HF_DATASETS_CACHE=.hf_home/datasets TRANSFORMERS_CACHE=.hf_home/transformers \
venv_arm64/bin/python scripts/overnight_v3_corrected_probes.py \
  --device cpu \
  --run-exp exp4 \
  --exp4-prompts 300 \
  --exp4-candidates 16 \
  --exp4-max-new-tokens 96 \
  --require-verifier-score \
  --batch-size 1 \
  --no-confirm
```

This command is the v3 corrected cache/gate probe, not a paper claim. Completion requires source, target, and verifier score surfaces plus at least about `80` prompts with one correct candidate before any gain/MDE verdict; otherwise park with the stronger-generator command.

## Reusable Backfill Template

```yaml
- id: latentwire_l_a2_generated_solution_rerank_cache
  reason: missing_powered_generated_solution_candidate_pool_with_enough_correct_candidates
  command: "python -m pmc.cache_latentwire_scores --tasks generated_math --split dev,gate --method-id L_A2_verifier_rerank --candidate-counts 16,32,64 --source-model <SOURCE_MODEL> --target-model <TARGET_MODEL> --verifier-model <VERIFIER_MODEL> --out results/backfill/latentwire_l_a2_generated_solution_rerank/<RUN_ID> --checkpoint-every-prompts 20 --no-confirm"
  required_cache_or_model: "dev/gate generated-solution candidate pools plus source_score, target_score, verifier_score; require >=80 prompts with at least one correct candidate before screening"
  est_gpu_hours: 0-2
  promotion_allowed: false
```

Do not treat L-A2 as live evidence until the cache includes generated-solution candidates, source scores, target scores, verifier scores, split IDs, and a no-confirm manifest.

## Receiver-Conditioned MI Probe After Cache Exists

```bash
python -m pmc.cache_latentwire_scores \
  --tasks generated_math \
  --split dev,gate \
  --method-id L_A2_verifier_rerank \
  --candidate-counts 16,32,64 \
  --source-model <SOURCE_MODEL> \
  --target-model <TARGET_MODEL> \
  --verifier-model <VERIFIER_MODEL> \
  --out results/backfill/latentwire_l_a2_generated_solution_rerank/<RUN_ID> \
  --checkpoint-every-prompts 20 \
  --no-confirm

python scripts/stage1_row_safe_screens.py \
  --root experimental --root results \
  --dashboard-dir dashboard \
  --results-dir results/stage1 \
  --queues-dir queues
```

The MI probe must answer whether generated-solution rerank has source-private information beyond receiver/verifier scores. If it is near the MMLU-Pro receiver-conditioned `0.281933` bits and does not convert to a deployable win, lock LatentWire negative.
