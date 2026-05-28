# RiskGuard Granite M11b Top-5 Calibration

## Method

Offline-only calibration on cached Granite M11b top-5 data. Candidate guards set predicted recovery to the static-1pct baseline value (0.0) when triggered, then recompute the preregistered bootstrap median CI with seed 20260527. This tests whether a cheap guard could cap negative tails without using recovery labels at deployment time.

## Available Data

- Final per-trace score caches are present for BF16, static_1pct, m11b_top1/top5/top10, and static_top10.
- Per-step or streaming PPL/loss ratios are not present; final loss ratios are diagnostic only and were not used for deployable promotion.
- Source activation magnitudes are present for all 12 prompts, 40 layers, every 100 decode positions from 100 to 10000.
- Protected-set trajectories are present, but the M11b policy builds global EMA sets from mean activation magnitudes across traces; per-trace churn here is recomputed from cached source activations.
- No target-side per-step activations, logits, top-k margins, or loss deltas are cached.

## Baseline Tail

- Included recoverable traces: 8/12.
- M11b top-5 median: 0.449284.
- M11b top-5 bootstrap CI: [-1.300900, 1.000791].
- Worst-quartile CVaR: -2.500238.

## Trigger Candidates

Best early source-activation trigger:
- Rule: `early_churn_mean >= 0.811005948`.
- Flagged recoverable traces: [4, 7] (2 negative, 0 positive).
- Predicted post-guard median: 0.449284.
- Predicted post-guard CI: [0.060213, 1.000791].
- Predicted worst-quartile CVaR: 0.000000.

Best full-source activation trigger, using position 10000 source activations but no target outcome:
- Rule: `early_churn_mean >= 0.811005948`.
- Flagged recoverable traces: [4, 7] (2 negative, 0 positive).
- Predicted post-guard median: 0.449284.
- Predicted post-guard CI: [0.060213, 1.000791].
- Predicted worst-quartile CVaR: 0.000000.

Robustness check:
- Leave-one-trace-out nested calibration gives median 0.449284, CI [0.000000, 1.000791], CVaR -1.139125.
- LOO-triggered traces: [4, 8]; remaining negative traces after LOO guard: [7].
- Interpretation: the positive in-sample CI depends on a threshold selected from the same 8 recoverable outcomes; when each trace is withheld, the guard misses one catastrophic trace and adds positive-trace fallbacks.

Oracle diagnostics, not deployable:
- Replacing exactly negative M11b top-5 traces with static baseline gives median 0.449284, CI [0.060213, 1.000791], CVaR 0.000000.
- Replacing exactly negative traces with BF16/full-precision gives median 1.000000, CI [0.194523, 1.000791], CVaR 0.157474.

## Decision

Do not promote to GPU confirmation. A source-activation churn trigger has positive in-sample CI, but the cutoff is outcome-fit on 8 traces and fails the leave-one-trace-out robustness check. This is not defensible enough to spend GPU confirmation time.

## Caveats

- All threshold search is in-sample over 8 recoverable traces; it is suitable for falsification/triage, not a claim.
- Source activation features may be deployable only in protocols that already run the source/BF16 pass before target scoring.
- Final score-cache loss ratios are available but intentionally excluded from promotion because they require outcome scoring and leak the quantity the guard is supposed to predict.
- No GPU was used.

## Source Files

- `/workspace/LatentWire/artifacts/external_review_pack/om_positive_method_pack_20260528_0536/EXECUTIVE_SUMMARY.md`
- `/workspace/LatentWire/artifacts/external_review_pack/om_positive_method_pack_20260528_0536/DECISIONS_SNAPSHOT.md`
- `/workspace/LatentWire/artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/05_m11b_granite/decision.json`
- `/workspace/LatentWire/artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/05_m11b_granite/per_trace.csv`
- `/workspace/LatentWire/experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/per_trace_metrics.json`
- `/workspace/LatentWire/experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/score_cache`
- `/workspace/LatentWire/experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/activation_magnitudes.jsonl.gz`
- `/workspace/LatentWire/experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/protected_sets.json`
- `/workspace/LatentWire/experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/protected_trajectories.json`
