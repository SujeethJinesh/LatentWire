# TRACE-ROUTER Offline Classifier

## Decision

Partial promotion: use TRACE-ROUTER only for the next Granite and DeepSeek gate. Do not promote Nemotron or Falcon. Falcon is marked NOT_ROUTABLE.

This is not paper-ready evidence. It is a tiny-n offline routing screen over cached AIME traces, suitable only for deciding whether a preregistered larger frozen-slice router run is worth doing.

## Features found

- Available leak-free first-1k features: early top-1/top-5/top-10 set drift from decode position 100 to 1000, adjacent early churn over positions 100..1000, and normalized top-k activation margins at position 1000.
- Unavailable or excluded: static-gap proxy, logit margin, early-KL slope, and no-gap predictor. Final static gap/no-gap labels are included in the feature table only as audit labels and were not used for routing.
- Source activations come from cached `activation_magnitudes.jsonl.gz`; no GPU work was run.

## CV protocol

- Per-model leave-one-trace-out CV over recoverable traces only. No-gap traces are excluded from gain because positive-static recovery is undefined there.
- Candidate policies: `static` = recovery 0 against `static_1pct`, `m11b_top5`, and `m11b_top10`.
- Train-fold baseline: best fixed policy by mean training recovery.
- Router: either a constant policy or one single-feature threshold stump, selected inside the train fold only. Held-out `oracle_best` is reported for audit and is not used to fit or route.

## Gains

| Model | n recoverable | CV mean gain | CV median gain | + / 0 / - folds | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Granite | 8 | 0.533548 | 0.449284 | 5 / 3 / 0 | PROMOTE_ROUTER_WEAK_CV |
| DeepSeek | 11 | 0.087906 | 0.000000 | 3 / 8 / 0 | PROMOTE_ROUTER_WEAK_CV |
| Nemotron | 10 | -0.064625 | 0.000000 | 1 / 6 / 3 | DO_NOT_PROMOTE_FIXED_POLICY_STRONGER |
| Falcon-H1 | 12 | 0.009051 | 0.000000 | 2 / 6 / 4 | NOT_ROUTABLE |

## Overfitting risk

Risk is high: n is 8-12 per model, there are no seed repeats, and the stump family still searches several early features. Granite has the clearest held-out signal, while DeepSeek is weaker: positive mean gain but zero median gain, driven by a few folds. Treat both as alive branches, not claims.

Nemotron has oracle headroom but the learned router underperforms the train-fold fixed `m11b_top10` baseline. Falcon remains not routable: the verified prior says Falcon is not routable, the CV median is zero, and held-out folds include negative gains.

## Saturated vs alive

- Alive: Granite TRACE-ROUTER and DeepSeek TRACE-ROUTER, only for a preregistered larger frozen-slice validation.
- Saturated: Nemotron fixed `m11b_top10` on this slice; routing did not improve it.
- Ruled out here: Falcon TRACE-ROUTER.

## Source files

- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/DECISIONS_SNAPSHOT.md`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/EXECUTIVE_SUMMARY.md`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/RUN_LEDGER_SNAPSHOT.md`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/05_m11b_granite/per_trace.csv`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/05_m11b_granite/source_files/per_trace_metrics.json`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/06_m11b_nemotron/per_trace.csv`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/06_m11b_nemotron/source_files/per_trace_metrics.json`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/07_m11b_deepseek/per_trace.csv`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/07_m11b_deepseek/source_files/per_trace_metrics.json`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/08_m11b_falcon/per_trace.csv`
- `artifacts/external_review_pack/om_positive_method_pack_20260528_0536/experiments/08_m11b_falcon/source_files/per_trace_metrics.json`
- `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/activation_magnitudes.jsonl.gz`
- `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z/activation_magnitudes.jsonl.gz`
- `experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z/activation_magnitudes.jsonl.gz`
- `experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z/activation_magnitudes.jsonl.gz`
- `paper/reviewer_feedback.md`
