# HORN Reviewer Pack

- status: preregistered Mac-gate scaffold; no positive asymmetry result
- current decision: blocked on real hybrid boundary activation dumps
- camera-readiness: not submittable as a paper result until H1--H3 have real
  trace evidence

## Paper Link

- Draft PDF: `experimental/horn/paper/horn_colm2026.pdf`
- Draft TeX: `experimental/horn/paper/horn_colm2026.tex`
- Preregistration: `experimental/horn/phase2/preregister_horn_20260506.md`

## Current Claim

HORN tests whether `attention->ssm` and `ssm->attention` boundaries differ in
activation-outlier magnitude and FP4-equivalent noise sensitivity. The current
artifacts define the gate and controls; they do not claim directional
asymmetry.

## Promotion Ladder Boundary

Only H1a/H1 boundary statistics have a real trace packet builder/checker today.
H2/H3 now have follow-up contract checks in
`experimental/shared/followup_gate_contracts.py`, but they are not current
evidence. A real H1a/H1 pass may authorize noisy replay and
pure-architecture control packets; it does not authorize a noise-propagation,
precision-allocation, or cross-architecture claim.
Status shorthand: H2/H3 are not current evidence.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | H1/H2 should use reasoning traces plus WikiText/GSM-style drift controls, but no live hybrid activations have been dumped yet. | Gate pending. |
| Ablations | Required controls are explicit boundary maps, non-boundary adjacent pairs, matched normalization placement, direction-label permutation, and pure-architecture controls. | Adequate before real H1. |
| Correctness | The checker now requires at least 12 prompts unless resource-limited, prompt-cluster IDs, both boundary directions, per-prompt both-direction non-boundary controls through `matched_boundary_direction`, finite numeric fields, prompt hash provenance, architecture-map hash provenance tied to the claimed model, a `trace_plan_path` whose file hash matches the registered `trace_plan_hash`, saved tensor artifact provenance, recomputed H1a `summary.json` gate aggregates, and every observed boundary tuple to have a permuted-direction row that reuses the same metrics/source/hash while flipping the actual `direction` label. Non-rehearsal packets must include the copied `.pt` activation tensors; the checker reloads them and recomputes max-abs, RMS, and kurtosis row metrics. The selected H1a lower bound is recomputed as a deterministic prompt-cluster bootstrap, so aggregate-only rows cannot fabricate the interval. Near-boundary non-boundary controls block promotion. The shared builder now automatically marks resource-limited packets non-promotable. H2/H3 follow-up contracts additionally reject missing fixed H1 direction, noise side/basis, paired clean/noisy NLL rows, hook-off controls, pure-attention controls, and pure-Mamba controls. | Artifact path is hardened. |
| Reproducibility | The 72-row synthetic H1a real-schema rehearsal is deterministic, passes the real checker in non-promoting mode, and shared architecture maps fix boundary IDs. | Not model evidence. |
| Novelty | The proposed wedge is directional propagation through hybrid boundaries, not generic activation-outlier measurement. | Plausible only if H1/H2/H3 pass. |
| Camera-readiness | The draft is a preregistration shell. It needs real H1/H2/H3 tables before submission as a method or measurement paper. | Not camera-ready. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic H1a schema rehearsal | 72 real-schema rows, selected max-abs ratio 4.044, non-boundary control ratio 1.042, permuted control ratio 0.247, real checker passes with `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A` | validates packet contract only |
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

- No real boundary activations have been measured.
- Directional ratios may vanish under pure-architecture controls.
- HORN may collapse into HBSM if the effect is just layer sensitivity.
- A precision-allocation recipe must eventually beat uniform precision at
  matched memory.

## Next Exact Gate

Run the H1a single-model screen on the smallest available live hybrid model.
The first packet must pass:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_gate_h1_<YYYYMMDD>_<model_slug> \
  --mode real --project horn
```

Continue only if real boundary-direction asymmetry passes the preregistered H1a
screen. Do not call this H1 promotion until the same selected direction appears
on at least two hybrid models and H3 controls are clean.
If the run is resource-limited, record it with
`RESOURCE_LIMITED_NOT_PROMOTABLE` and do not treat it as H1 promotion.
