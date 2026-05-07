# HORN Reviewer Pack

- status: demoted Mac-gate control branch; no positive asymmetry result
- current decision: H1a and H2 scouts both fail on Granite Tiny
- camera-readiness: Not camera-ready; needs real H1/H2/H3 tables before any
  standalone method or measurement submission

## Paper Link

- Draft PDF: `experimental/horn/paper/horn_colm2026.pdf`
- Draft TeX: `experimental/horn/paper/horn_colm2026.tex`
- Preregistration: `experimental/horn/phase2/preregister_horn_20260506.md`

## Current Claim

HORN tests whether `attention->ssm` and `ssm->attention` boundaries differ in
activation-outlier magnitude and FP4-equivalent noise sensitivity. The current
artifacts define the gate and controls; the available Granite Tiny evidence
does not support a directional-asymmetry claim.

## Promotion Ladder Boundary

H1a/H1 boundary statistics have a real trace packet builder/checker, and H2/H3
have follow-up contract checks in
`experimental/shared/followup_gate_contracts.py`. The current H2 scout is
contract-valid but resource-limited and failing; it supports demotion, not a
noise-propagation, precision-allocation, or cross-architecture claim.
Status shorthand: H1a failed; H2 scout failed and is not current evidence for a
positive claim; H3 has no model evidence.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | H1a used real Granite Tiny boundary hooks over 12 short reasoning prompts; H2 used two short prompts, three seeds, and paired boundary-direction noisy-continuation rows. This is not enough for a positive paper but is enough to demote the standalone branch. | Demoted. |
| Ablations | Required controls are explicit boundary maps, non-boundary adjacent pairs, matched normalization placement, direction-label permutation, and pure-architecture controls. | Adequate before real H1. |
| Correctness | The checker now requires at least 12 prompts unless resource-limited, prompt-cluster IDs, both boundary directions, per-prompt both-direction non-boundary controls through `matched_boundary_direction`, finite numeric fields, prompt hash provenance, architecture-map hash provenance tied to the claimed model, a `trace_plan_path` whose file hash matches the registered `trace_plan_hash`, saved tensor artifact provenance, recomputed H1a `summary.json` gate aggregates, and every observed boundary tuple to have a permuted-direction row that reuses the same metrics/source/hash while flipping the actual `direction` label. Non-rehearsal packets must include the copied `.pt` activation tensors; the checker reloads them and recomputes max-abs, RMS, and kurtosis row metrics. The selected H1a lower bound is recomputed as a deterministic prompt-cluster bootstrap, so aggregate-only rows cannot fabricate the interval. Near-boundary non-boundary controls block promotion. The shared builder now automatically marks resource-limited packets non-promotable. H2/H3 follow-up contracts additionally reject missing fixed H1 direction, noise side/basis, paired clean/noisy NLL rows, hook-off controls, pure-attention controls, and pure-Mamba controls. | Artifact path is hardened. |
| Reproducibility | The 72-row synthetic H1a real-schema rehearsal is deterministic, passes the real checker in non-promoting mode, and shared architecture maps fix boundary IDs. The H1a real packet and H2 scout are both reproducible from local cached Granite Tiny weights with repo-local commands. | Strong enough for demotion; not positive evidence. |
| Novelty | The proposed wedge is directional propagation through hybrid boundaries, not generic activation-outlier measurement. | Plausible only if H1/H2/H3 pass. |
| Camera-readiness | The draft is a preregistration shell and needs real H1/H2/H3 tables before any standalone method/measurement submission. As a standalone paper it should be stopped unless reopened with a new gate; the current value is a negative/control appendix for SSQ-LR/HBSM. | Not camera-ready standalone. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic H1a schema rehearsal | 72 real-schema rows, selected max-abs ratio 4.044, non-boundary control ratio 1.042, permuted control ratio 0.247, real checker passes with `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A` | validates packet contract only |
| Granite Tiny H1a real screen | 288 rows, 12 prompts, all 8 planned boundaries, right-layer input hook tensors, selected ratio `1.06`, cluster-bootstrap low `1.06`, checker-passing decision `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN` | weak single-model screen; no H1a/H1 promotion |
| Granite Tiny H2 scout | 20 rows, 2 prompts, 3 seeds, paired units `6/6`, hook-off max delta `0.0`, H1-selected direction preserved, directional drift ratio `1.037`, paired lower bound `1.072`, contract-valid decision `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION` | demotes HORN standalone |
| architecture provenance | shared boundary IDs and direction counts exist | packet provenance ready |
| trace collection plan | `experimental/shared/results/hybrid_trace_plan_20260507/horn_trace_plan.jsonl` enumerates 1,404 boundary/control capture rows, including a matched non-boundary control for each observed boundary row | execution checklist only |
| model eligibility | live targets are identified, but weights are not cached locally | blocked on model load |
| real-packet checker | rejects missing directions, stale summary fields, too few prompts, non-finite rows, promotable resource-limited decisions, missing trace-plan hash pinning, missing tensor artifacts, tensor hash mismatches, row metrics that do not recompute from saved activation tensors, unpaired or independently measured permuted controls, near-boundary non-boundary controls, and permuted controls that preserve the selected directional effect | ready for real H1 |
| follow-up contract checker | `experimental.shared.followup_gate_contracts --gate horn_h2/horn_h3` enforces noisy replay and cross-architecture control fields before later evidence can be cited | contract ready, no model rows |

Non-boundary rows may retain their true architecture direction while using
`matched_boundary_direction` to identify the boundary direction they control
for, and this pairing is required for both directions on every prompt. The required matched flipped controls are the `permuted_direction` rows: they
must reuse an observed boundary tuple with the same prompt ID and normalization
positions, then invert the actual `direction` label. The H1a evaluator records
selected-direction control ratios, so it rejects controls that keep the
high-magnitude signal on the same direction label. A faithful label flip that
preserves unsigned max/min asymmetry but moves the signal to the opposite label
is treated as an acceptable null.

## Reviewer Risks

- The available directional ratios already look near-null on short Granite Tiny
  prompts.
- H3 pure-architecture controls have not been run, but the branch is too weak
  to justify them without a new reason.
- HORN should not be presented as a precision-allocation recipe.
- Any future revival must beat uniform precision at matched memory after
  clearing H1--H3.

## Next Exact Gate

Do not spend GPU on HORN standalone. Fold the H1a/H2 failures into SSQ-LR/HBSM
as negative/control evidence. Reopen HORN only with a new preregistered full
H2/H3 scope on more prompts/models and a clear reason why the `1.037` H2 scout
should be overturned.
