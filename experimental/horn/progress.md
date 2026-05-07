# HORN Progress

## Current Supersession Note

Older "next gate" entries below are historical. The controlling evidence as of
2026-05-07 is the checker-passing but resource-limited H1a scout plus the H2
noisy-continuation scout; both fail their gates. HORN is demoted to
control/negative evidence and should not be GPU-promoted unless a new
preregistered full H2/H3 reopening explains why the current near-null,
direction-flipping scout should not generalize.

## 2026-05-06 Historical Setup Entry

Status: **HISTORICAL / superseded Mac-gate setup**.

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

## 2026-05-07 Prompt-Cluster Bootstrap Alignment

The H1a evaluator now computes `selected_h1_ci_low` and
`selected_h1_cluster_bootstrap_low` with the same deterministic prompt-cluster
bootstrap implied by the preregistration, rather than a raw prompt-ratio order
statistic. Real H1a rows must carry `prompt_cluster_id`; the trace plan derives
it from the prompt manifest's task/cluster metadata. This keeps the uncertainty
unit at the prompt cluster and prevents aggregate-only packets from
hand-filling the lower bound.

Decision: **H1A LOWER BOUNDS ARE NOW PROMPT-CLUSTER BOOTSTRAP READOUTS**. The
blocker remains the real prompt-paired hybrid activation dump.

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

## 2026-05-07 Trace-Plan Artifact

Added `../shared/hybrid_trace_plan.py` and generated
`../shared/results/hybrid_trace_plan_20260507/`. For HORN, the plan enumerates
1,404 H1a/H1 capture rows across frozen prompts, shared architecture-map
boundaries, observed boundary rows, metric-reused permuted-direction rows, and
matched non-boundary controls for both boundary directions. The real-packet
checker now requires a `trace_plan_hash` for non-rehearsal packets, so future
H1a rows must cite the exact plan JSONL used during capture.

Decision: **H1A TRACE CAPTURE IS NOW OPERATIONALLY SPECIFIED BUT STILL NOT RUN**.
The next exact gate remains a real boundary tensor packet built from those
planned rows and checked with `check_gate_packet --mode real --project horn`.

## 2026-05-07 Capture-Manifest Templates

Added `../shared/hybrid_trace_capture_manifest.py` and generated
`../shared/results/hybrid_capture_manifests_20260507/`. For HORN, the artifact
provides per-model fill-in metadata templates with observed-boundary,
metric-reused permuted-direction, and matched non-boundary rows. Granite
templates contain 288 planned entries each, while the Qwen3-Next template
contains 828 entries because its architecture map exposes more hybrid
boundaries.

Decision: **H1A CAPTURE NOW HAS A FILL-IN TEMPLATE BUT STILL NO MODEL
EVIDENCE**. The next exact gate is to fill one HORN template from a real
boundary-activation capture, build the packet, and validate it with
`check_gate_packet --mode real --project horn`.

## 2026-05-07 Model-Alias and Permuted-Tensor Guard

The shared architecture maps now carry canonical model IDs and registered
served/HF aliases, so real H1a packets can preserve a served ID as
`served_model_id` while validating rows against the canonical map ID. The
capture manifests also now encode HORN `permuted_direction` rows with
`tensor_alias_of`, and the builder reuses the observed boundary tensor for the
flipped-label control. This prevents a real capture from dumping independent
permuted tensors that the checker would later reject as metric-mismatched.

Decision: **H1A PACKETS NOW REPRESENT PERMUTED CONTROLS AS METADATA ALIASES**.
The blocker remains a real boundary-activation capture.

## 2026-05-07 Complete Permuted-Pair Guard

The shared real-packet checker now requires every observed HORN boundary tuple
to have its paired `permuted_direction` row. The earlier checker verified each
supplied permuted row but did not reject omitted permuted rows, which could let
a real packet skip hard boundary tuples while still passing aggregate direction
coverage. A new negative test removes one permuted pair and confirms the packet
is rejected.

Decision: **H1A PERMUTED CONTROLS ARE NOW REQUIRED PER OBSERVED BOUNDARY, NOT
JUST IN AGGREGATE**. The blocker remains a real prompt-paired boundary dump.

## 2026-05-07 Trace-Plan Path Guard

The shared real-packet checker now rejects non-rehearsal HORN packets that omit
`trace_plan_path`. The H1 row set must be loaded from a cited frozen trace plan
before the checker will accept boundary, non-boundary, or permuted-control
coverage.

Decision: **H1A REAL BOUNDARY ROWS MUST BE TRACE-PLAN-CHECKABLE**. The blocker
remains the first real boundary-activation packet.

## 2026-05-07 Tensor Provenance Guard

`activation_dumper.py` now writes a tensor manifest with original hook names,
storage names, SHA-256 hashes, dtypes, shapes, and element counts. HORN builder
rows copy that provenance into every boundary/control row, and
`permuted_direction` rows must carry `tensor_alias_of` while reusing the
observed boundary tensor source and hash.

Decision: **H1A BOUNDARY METRICS MUST BE HASHED BACK TO THEIR SAVED TENSORS**.
The blocker remains the first real prompt-paired boundary dump.

## 2026-05-07 Promotable Trace-Plan Hash Guard

After reviewer audit, H1a/H1 real packets cannot promote by pointing
`trace_plan_path` at a caller-created subset plan. Non-resource-limited packets
must cite trace-plan rows whose file SHA-256 equals the registered shared HORN
`trace_plan_hash`. Resource-limited subset packets remain allowed only as
`RESOURCE_LIMITED_NOT_PROMOTABLE` diagnostics.

Decision: **H1A PROMOTION MUST USE THE REGISTERED FROZEN PLAN CONTENT**. The
blocker remains the first real prompt-paired boundary dump.

## 2026-05-07 Saved Tensor Metric Guard

After COLM-style artifact review, the shared packet builder now copies the
saved HORN tensor manifest and `.pt` files into every built real packet. The
real checker reloads each cited tensor and recomputes `max_abs`, `rms`, and
`kurtosis`; a row whose metrics do not match the saved activation bytes is
rejected even if its hash provenance fields are syntactically valid. Permuted
controls must still reuse the observed boundary tensor source, hash, and
metrics.

Decision: **H1A ROW METRICS MUST BE RECOMPUTABLE FROM SAVED BOUNDARY TENSORS**.
The blocker remains the first real prompt-paired boundary dump.

## 2026-05-07 H1a / H2 Scout Decision

The first real Granite Tiny H1a screen is checker-passing but
resource-limited and fails the magnitude-asymmetry gate: 288 rows, 12 prompts,
all 8 planned boundaries, right-layer input hook tensors, selected ratio
`1.06`, and cluster-bootstrap low `1.06`. This is too close to null to promote
H1a/H1.

The H2 noisy-continuation scout also fails: 20 rows over 2 prompts and 3 seeds,
paired units `6/6`, hook-off max delta `0.0`, fixed H1-selected direction
preserved in the aggregate, directional drift ratio `1.037`, signed
selected-direction lower bound `0.324`, and selected-direction support fraction
`0.5`.
The packet is contract-valid but its decision is
`FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`.

Decision: **HORN IS A DEMOTED CONTROL BRANCH**. Do not spend GPU time on a
standalone HORN claim. Reopen only with a new preregistered full H2/H3 scope
and a concrete reason the current near-null, direction-flipping H2 scout should
reverse.
