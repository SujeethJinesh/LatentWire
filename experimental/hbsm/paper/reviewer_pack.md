# HBSM Reviewer Pack

- status: preregistered Mac-gate scaffold; novelty wounded by adjacent
  sensitivity work
- current decision: blocked on real hybrid layer-sensitivity rows
- camera-readiness: not submittable as a standalone paper until B1--B3 separate
  the mechanism from existing sensitivity tools

## Paper Link

- Draft PDF: `experimental/hbsm/paper/hbsm_colm2026.pdf`
- Draft TeX: `experimental/hbsm/paper/hbsm_colm2026.tex`
- Preregistration: `experimental/hbsm/phase2/preregister_hbsm_20260506.md`

## Current Claim

HBSM asks whether sensitivity in current hybrid reasoners can be explained by
boundary mechanisms and predicted more cheaply than repeated quantized forward
passes. The current artifacts do not claim novelty or a positive predictor.
They define the controls needed to decide whether HBSM survives or folds into
HORN.

## Promotion Ladder Boundary

Only B1 sensitivity heterogeneity has a real trace packet builder/checker
today. B2/B3 now have follow-up contract checks in
`experimental/shared/followup_gate_contracts.py`, but they are not current
evidence. A real B1 pass may authorize predictor-split and matched-noise
mechanism packets; it does not authorize a cheap-predictor, no-forward-pass, or
mechanism claim.
Status shorthand: B2/B3 are not current evidence.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | B1 should measure KL/NLL drift on current hybrid reasoners, but no live sensitivity sweep exists yet. | Gate pending. |
| Ablations | Required baselines are perturbation-off, random flags, layer index, parameter count/norm, boundary-only, KL-style ranking, activation/outlier ranking, and train/test or leave-one-model-out splits. | Adequate before real B1 and B2. |
| Correctness | The checker scores only primary `boundary_only` rows after aggregating prompt rows to `(model_id, layer)`, derives measured top deciles from aggregated `kl_or_nll_drift`, rejects supplied `top_decile_flag` disagreements on any prompt row, and requires prompt-level boundary/non-boundary coverage, finite metrics, true scoring-layer top-decile cardinality, a same-count non-enriched random baseline, train/test coverage, prompt hash provenance, architecture-map hash provenance tied to the claimed model, a `trace_plan_path` whose file hash matches the registered `trace_plan_hash`, every comparator control on the same `(model_id, layer)` scoring set as `boundary_only`, copied source sensitivity row-packet provenance, recomputed B1 `summary.json` enrichment/p-value/Spearman aggregates, and near-zero drift for perturbation-off rows. The shared builder now automatically marks resource-limited packets non-promotable. B2/B3 follow-up contracts additionally reject missing predictor registry hashes, hyperparameter hashes, train-only selection, held-out Spearman/baseline margins, HORN-alignment signs, noise-off controls, and direction-flip controls. | Artifact path is hardened. |
| Reproducibility | Synthetic B1 real-schema rehearsal is deterministic, passes the real checker in non-promoting mode, and shared architecture maps fix boundary flags. | Not model evidence. |
| Novelty | Broad forward-only sensitivity is crowded; the defensible wedge is mechanism plus cheaper predictor on current hybrid reasoners. | Narrow and fragile. |
| Camera-readiness | The draft is a preregistration shell. It needs real B1/B2/B3 evidence before submission as a standalone paper. | Not camera-ready. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic B1 schema rehearsal | 720 real-schema rows, 480 primary prompt rows, 240 layer-aligned control rows, 40 scoring layers after aggregation, real checker passes with `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1` | validates packet contract only |
| architecture provenance | shared boundary flags and model hashes exist | packet provenance ready |
| trace collection plan | `experimental/shared/results/hybrid_trace_plan_20260507/hbsm_trace_plan.jsonl` enumerates 2,304 layer-aligned B1 sensitivity/control rows | execution checklist only |
| model eligibility | live targets are identified, but weights are not cached locally | blocked on model load |
| real-packet checker | rejects missing no-op rows, missing primary-row prompt coverage, stale summary fields, non-finite metrics, split omissions, missing trace-plan hash pinning, missing source sensitivity artifacts, source row-packet hash mismatches, non-layer-aligned comparator controls, promotable resource-limited decisions, supplied top-decile flags that disagree with measured drift, unmatched random/top-decile counts, and random baselines that reproduce boundary enrichment | ready for real B1 |
| follow-up contract checker | `experimental.shared.followup_gate_contracts --gate hbsm_b2/hbsm_b3` enforces frozen predictor splits and matched-noise mechanism controls before later evidence can be cited | contract ready, no model rows |

## Reviewer Risks

- No real forward-sensitivity row exists.
- KL Lens-style forward sensitivity for mixed-precision hybrid models narrows
  the novelty claim; see arXiv:2604.13440
  (`https://arxiv.org/abs/2604.13440`).
- Cheap predictors may only learn depth, size, or norm baselines.
- If B3 matches HORN H2, HBSM should be folded into HORN rather than kept
  separate.

## Next Exact Gate

Run B1 on the smallest available live hybrid model. The first packet must pass:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/hbsm/phase2/results/hbsm_gate_b1_<YYYYMMDD>_<model_slug> \
  --mode real --project hbsm
```

Continue only if real layer-sensitivity heterogeneity passes the preregistered
B1 rule.
If the run is resource-limited, record it with
`RESOURCE_LIMITED_NOT_PROMOTABLE` and do not treat it as B1 promotion.
