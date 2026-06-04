# LatentWire L-A2 Data Readiness

- status: `PARKED_NEEDS_CACHE`
- evidence surface: no real generated-solution candidate pool with verifier/source/target score surfaces exists beyond the 3-row smoke.
- existing smoke: `results/mac_continue/fresh_mmlu_pro/rerank_generation_rows.jsonl`, 3 generated rows with `row_id`, `split`, `answer_index`, and `generated_text` only.
- usable source/target MC score surfaces: `results/mac_continue/fresh_mmlu_pro/fresh_mmlu_pro_rows.jsonl`, but these are option-score rows, not generated-solution rerank pools.
- conclusion: L-A2 is not evidence yet; it is a reusable cache/backfill job.
- receiver-conditioned MI probe: `NOT_RUN_NO_CACHE`; cannot estimate until generated-solution candidates have source, target, and verifier scores.

## CPU Smoke Command

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
HF_HOME=.hf_home HF_DATASETS_CACHE=.hf_home/datasets TRANSFORMERS_CACHE=.hf_home/transformers \
venv_arm64/bin/python scripts/mac_continue_latentwire.py \
  --device cpu \
  --max-scan-rows 360 \
  --max-score-rows 128 \
  --max-generation-rows 128 \
  --generation-timebox-seconds 21600 \
  --max-new-tokens 96 \
  --batch-size 1
```

## Reusable Backfill Template

```yaml
- id: latentwire_l_a2_generated_solution_rerank_cache
  reason: missing_real_generated_solution_candidate_pools_and_verifier_source_target_surfaces
  command: "python -m pmc.cache_latentwire_scores --tasks generated_math --split dev,gate --method-id L_A2_verifier_rerank --candidate-counts 16,32,64 --source-model <SOURCE_MODEL> --target-model <TARGET_MODEL> --verifier-model <VERIFIER_MODEL> --out results/backfill/latentwire_l_a2_generated_solution_rerank/<RUN_ID> --checkpoint-every-prompts 20 --no-confirm"
  required_cache_or_model: "dev/gate generated-solution candidate pools plus verifier/source/target score surfaces; repro+leakage review for exact code hash"
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
