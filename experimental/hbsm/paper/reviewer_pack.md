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

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | B1 should measure KL/NLL drift on current hybrid reasoners, but no live sensitivity sweep exists yet. | Gate pending. |
| Ablations | Required baselines are perturbation-off, random flags, layer index, parameter count/norm, boundary-only, and train/test or leave-one-model-out splits. | Adequate before real B1/B2. |
| Correctness | The checker requires true and false boundary flags, finite metrics, matched top-decile/random counts, train/test coverage, hash-shaped prompt/architecture provenance, decision-grade `summary.json` aggregates, and near-zero drift for perturbation-off rows. | Artifact path is hardened. |
| Reproducibility | Synthetic B1/B2 packet is deterministic, and shared architecture maps fix boundary flags. | Not model evidence. |
| Novelty | Broad forward-only sensitivity is crowded; the defensible wedge is mechanism plus cheaper predictor on current hybrid reasoners. | Narrow and fragile. |
| Camera-readiness | The draft is a preregistration shell. It needs real B1/B2/B3 evidence before submission as a standalone paper. | Not camera-ready. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic B1/B2 | kurtosis-vs-sensitivity Spearman rho 0.657, two boundary top-decile hits | validates readout only |
| architecture provenance | shared boundary flags and model hashes exist | packet provenance ready |
| model eligibility | live targets are identified, but weights are not cached locally | blocked on model load |
| real-packet checker | rejects missing no-op rows, missing boundary-flag coverage, non-finite metrics, split omissions, promotable resource-limited decisions, and unmatched random/top-decile counts | ready for real B1 |

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
