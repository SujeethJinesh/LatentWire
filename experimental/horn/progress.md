# HORN Progress

## 2026-05-06

Status: **NEW / Mac gates pending**.

Added and ran a deterministic synthetic H1 packet. It has now been upgraded to
a non-promoting real-schema H1a rehearsal:

- script: `phase2/horn_synthetic_h1_gate.py`
- packet: `phase2/results/horn_synthetic_h1/`
- decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A`
- rows: `72`
- selected H1a max-abs ratio: `4.044`
- non-boundary selected-direction control ratio: `1.042`
- permuted selected-direction control ratio: `0.247`
- real checker: passes `--mode real --project horn`

Interpretation: synthetic-only schema validation. It exercises the real H1a
row schema, paired controls, hash provenance, and recomputed evaluator summary,
but does not promote the branch or replace real hybrid boundary dumps.

Next exact gate: H1 activation magnitude and kurtosis characterization using
shared boundary-inspection utilities.

## 2026-05-06 Architecture Provenance Update

Added shared config-derived architecture maps at
`../shared/results/hybrid_architecture_maps_20260506/`. Real H1 packets must use
these explicit boundary IDs and include `architecture_map_hash`; substring-only
module classification is no longer admissible for promotion.

## 2026-05-06 Model Eligibility Update

Added metadata-only model eligibility at
`../shared/results/hybrid_model_eligibility_20260506/`. No live hybrid target is
cached repo-locally, and GPU-sized targets should wait for the 5090. HORN cannot
produce real boundary activation rows until a live hybrid model is loaded and
hooked.

## 2026-05-06 Real Packet Admissibility Update

The shared checker now rejects real HORN packets unless boundary rows cover both
`attention->ssm` and `ssm->attention`, and every `permuted_direction` control
matches an observed boundary tuple while flipping its direction. This prevents a
syntactically valid packet from skipping the directional asymmetry claim.
It also requires prompt coverage and finite numeric fields before H1 can be
interpreted.

## 2026-05-06 Reviewer Pack Update

Added `paper/reviewer_pack.md` and wired the stricter directional packet blocker
into the COLM shell. The paper now states that HORN is not camera-ready as a
method or measurement paper until real H1--H3 evidence exists.

## 2026-05-06 Decision-Grade Packet Hardening

Tightened the real H1 contract after COLM-style review. A real HORN packet now
must include hash-shaped prompt and architecture provenance, aggregate H1 fields
in `summary.json`, and `permuted_direction` controls paired by prompt ID,
boundary index, layer IDs, and normalization positions. Resource-limited runs
are diagnostic only and must use `RESOURCE_LIMITED_NOT_PROMOTABLE`.

Decision: **H1 PROMOTION NOW REQUIRES PROMPT-PAIRED DIRECTIONAL CONTROLS**. The
next exact gate remains a live boundary-activation dump on the smallest
available hybrid model.

## 2026-05-06 Recomputed Gate Evaluator Update

Added `../shared/hybrid_gate_evaluators.py` and wired the HORN real-packet
checker to recompute H1 directional ratios, selected metric/direction,
support fraction, non-boundary selected-direction control ratio, and
permuted-direction selected-direction control ratio from `raw_rows.jsonl`. A
real H1 packet now cannot pass by copying synthetic summary fields without
matching boundary rows. The shared Mac smoke prompt
manifest is `../shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl` with
SHA-256 `48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`.

Decision: **NEXT H1 MUST BE GENERATED FROM RAW BOUNDARY ROWS**. The blocker is
still a live hybrid activation dump.

## 2026-05-06 Permuted-Control Semantics Fix

After COLM-style review, the H1 evaluator no longer treats unsigned max/min
asymmetry in `permuted_direction` rows as automatic failure. A faithful
permutation flips only the direction label while preserving the observed
activation value, so the unsigned ratio can remain large. The gate now checks
whether the selected high-magnitude direction is erased by non-boundary and
permuted controls. Controls that keep the signal on the same selected direction
label still block H1.

The latest hardening also requires permuted rows to reuse the observed boundary
metrics, not independently measured tensors, and requires non-boundary controls
to stay below the selected H1 threshold rather than only below the boundary
effect size.

Decision: **H1 NULL CONTROLS MUST ERASE THE SELECTED DIRECTION, NOT MERELY
UNSIGNED ASYMMETRY**. The blocker remains a real prompt-paired boundary dump.

## 2026-05-06 H1a Promotion Boundary

After COLM-style review, the shared HORN evaluator now labels a single-model
directional asymmetry packet as an H1a screen rather than full H1 promotion. A
real single-model packet can emit `PASS_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN`,
but the project still needs cross-model consistency before claiming H1.

Decision: **SINGLE-MODEL BOUNDARY ASYMMETRY IS A SCREEN, NOT A PAPER RESULT**.
The next exact gate remains a real prompt-paired H1a dump, followed by
cross-model H1 aggregation only if the screen passes.

## 2026-05-06 H1a Schema-Rehearsal Upgrade

After COLM-style review, the HORN synthetic packet now validates as a
non-promoting real-schema rehearsal with `schema_rehearsal: true` and decision
`SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A`. The shared checker now
requires `matched_boundary_direction` so non-boundary controls can keep their
true architecture direction while still being paired against both boundary
directions for H1a scoring.
It also requires every prompt to have both matched non-boundary directions and
requires permuted controls to flip the actual `direction` label, not only a
matched-control metadata field.

Decision: **HORN PACKET PLUMBING IS READY FOR A REAL BOUNDARY DUMP**. The
blocker is still live hybrid activations with prompt-paired boundary,
non-boundary, and permuted controls.

## 2026-05-07 Resource-Limited Builder Guard

Updated the shared trace packet builder so any HORN tensor packet whose metadata
contains `resource_limit_note` writes a `RESOURCE_LIMITED_NOT_PROMOTABLE_...`
decision automatically. This lets a small Mac hook smoke validate prompt-paired
boundary/non-boundary/permuted rows without accidentally looking like an H1a
promotion if the recomputed directional metric passes on too few prompts.

Decision: **RESOURCE-LIMITED H1A SMOKE PACKETS CAN TEST HOOKS BUT CANNOT
PROMOTE H1A**. The blocker remains the full prompt-paired real boundary dump.

## 2026-05-07 Architecture Hash Provenance Guard

The shared real-packet checker now verifies that a non-rehearsal HORN packet's
`model_id` and `architecture_map_hash` match the shared architecture map
artifact. This prevents a boundary packet from citing a hash-shaped but
unrelated architecture provenance field.

Decision: **H1A BOUNDARY ROWS MUST BE TIED TO A KNOWN HYBRID ARCHITECTURE MAP**.
The blocker remains the same prompt-paired real activation dump.
