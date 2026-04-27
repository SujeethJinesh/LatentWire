# Clean3 Source-Answer Ablation Gate

- date: `2026-04-27`
- readiness: not ICLR-ready
- scale-up rung: smoke falsification
- decision: kill the clean3 source candidate-score sidecar as a positive-method
  branch

## Question

Does the clean3 top-sidecar smoke still recover the source-necessary ID after
removing exact source-final and verified-answer numeric evidence, or is the
effect explained by source-answer copying into a target-only sampled candidate
pool?

## Commands

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/full_top_selector_rerun.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/full_top_selector_rerun.md
```

```bash
./venv_arm64/bin/python scripts/materialize_sidecar_counterfactuals.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --mode source_answer_masked \
  --date 2026-04-27 \
  --output-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.jsonl \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.md
```

```bash
./venv_arm64/bin/python scripts/materialize_sidecar_counterfactuals.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --mode source_final_only \
  --date 2026-04-27 \
  --output-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.jsonl \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.md
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_top_selector.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_top_selector.md
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 0 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_top_selector.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_top_selector.md
```

## Evidence

| Selector | Status | Source-Necessary Clean | Control Clean Union | Matched Accepted Harm |
|---|---|---:|---:|---:|
| full sidecar | `top_sidecar_selector_passes_smoke` | `1` | `0` | `0` |
| source-answer masked | `top_sidecar_selector_fails_smoke` | `0` | `0` | `0` |
| source-final only | `top_sidecar_selector_passes_smoke` | `1` | `0` | `0` |

The only recovered ID is `14bfbfc94f2c2e7b`. The full sidecar selects the gold
value `3`, but that win disappears when values matching the source final or
verified equation results are masked. A source-final-only sidecar recovers the
same ID with the same clean-control behavior.

## Decision

Kill this branch as a positive communication method. The clean3 smoke is
explained by source-final answer evidence over a target-only sampled candidate
pool, not by a richer interpretable communication interface. It remains useful
as a diagnostic: target-only sampling can expose receiver-side headroom, but
future source sidecars must pass source-answer masking or avoid direct numeric
answer channels.

## Artifacts

Scratch artifacts are intentionally under `.debug/` and are not tracked:

- `.debug/clean3_sidecar_counterfactuals_20260427/full_top_selector_rerun.json`
- `.debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.jsonl`
- `.debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_top_selector.json`
- `.debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.jsonl`
- `.debug/clean3_sidecar_counterfactuals_20260427/source_final_only_top_selector.json`

Hashes:

- `.debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.jsonl`:
  `d8fcde23ea05c1f989974925467c769565f11cdfa5fda61c3de852667fb4a7a2`
- `.debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.jsonl`:
  `e1fe88457fdd74defd94008fd22178aa355a2d54278fac44671d5e9817b655c3`

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_materialize_sidecar_counterfactuals.py \
  tests/test_analyze_candidate_score_sidecar_top_select.py -q
```

Result: `5 passed`.

## Next Gate

Switch the live branch away from clean3 candidate-score sidecars. The next
highest-value CPU-feasible branch is source-surface discovery over existing
artifacts: identify exact-ID surfaces where target-only/no-source candidate
pools contain source-necessary answers that are not explainable by direct
source-final numeric evidence. If no existing artifact has that property, wait
for MPS cleanup and generate a richer same-family surface before testing a new
selector.
